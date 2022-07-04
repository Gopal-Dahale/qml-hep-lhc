from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Flatten
import sympy
import cirq
from qml_hep_lhc.models.quantum.utils import one_qubit_unitary
from qml_hep_lhc.models.base_model import BaseModel
import tensorflow_quantum as tfq
import numpy as np
from qml_hep_lhc.encodings import AngleMap
from qml_hep_lhc.utils import _import_class
from tensorflow import Variable, random_uniform_initializer, constant, shape, repeat, tile, concat, gather
from qml_hep_lhc.models.quantum.utils import symbols_in_expr_map, resolve_formulas, get_count_of_qubits, get_num_in_symbols
from tensorflow import pad, constant

CONV_SYMBOLS = 15
POOL_SYMBOLS = 6


class QCNN(Layer):

    def __init__(self, input_dim, fm_class):
        super(QCNN, self).__init__()

        self.fm_class = fm_class

        # Prepare qubits
        self.n_qubits = get_count_of_qubits(self.fm_class, input_dim)
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)

        # Observables
        Z = cirq.PauliString(cirq.Z(self.qubits[-1]))
        I = cirq.PauliString(cirq.I(self.qubits[-1]))
        self.observables = [-0.5 * Z + 0.5 * I]

        # Sympy symbols for variational angles
        self.num_layers = int(np.log2(self.n_qubits))

        var_symbols = sympy.symbols(
            f'θ0:{self.num_layers*(CONV_SYMBOLS + POOL_SYMBOLS)}')
        self.var_symbols = np.asarray(var_symbols).reshape(self.num_layers, -1)

        # Sympy symbols for encoding angles
        self.num_in_symbols = get_num_in_symbols(self.fm_class, input_dim)
        in_symbols = sympy.symbols(f'x0:{self.num_in_symbols}')
        self.in_symbols = np.asarray(in_symbols).reshape((self.num_in_symbols))

    def build(self, input_shape):

        # Define data and model circuits
        data_circuit = cirq.Circuit()
        model_circuit = cirq.Circuit()

        # Prepare data circuit
        data_circuit += cluster_state_circuit(self.qubits)
        self.fm = _import_class(f"qml_hep_lhc.encodings.{self.fm_class}")()
        data_circuit += self.fm.build(self.qubits, self.in_symbols)

        # Prepare model circuit
        # Alternating convolutional and pooling layer
        n = self.n_qubits
        for i in range(self.num_layers):
            conv_start = n - (n // 2**i)
            pool_source_start = conv_start
            pool_source_end = n - (n // 2**(i + 1))
            pool_sink_start = pool_source_end

            model_circuit += quantum_conv_circuit(
                self.qubits[conv_start:], self.var_symbols[i, :CONV_SYMBOLS])
            model_circuit += quantum_pool_circuit(
                self.qubits[pool_source_start:pool_source_end],
                self.qubits[pool_sink_start:], self.var_symbols[i,
                                                                CONV_SYMBOLS:])

        # Convert symbols to list
        self.var_symbols = list(self.var_symbols.flat)
        self.in_symbols = list(self.in_symbols.flat)

        # Initalize variational angles
        var_init = random_uniform_initializer(minval=-np.pi / 2,
                                              maxval=np.pi / 2)
        self.theta = Variable(initial_value=var_init(
            shape=(1, len(self.var_symbols)), dtype="float32"),
                              trainable=True,
                              name="thetas")

        # Flatten circuit
        data_circuit, expr_map = cirq.flatten(data_circuit)
        self.raw_symbols = symbols_in_expr_map(expr_map)
        self.expr = list(expr_map)

        # Define explicit symbol order and expression resolver
        symbols = [str(symb) for symb in self.var_symbols + self.expr]
        self.indices = constant([symbols.index(a) for a in sorted(symbols)])
        self.input_resolver = resolve_formulas(self.expr, self.raw_symbols)

        # Define computation layer
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.ansatz = model_circuit
        self.computation_layer = tfq.layers.ControlledPQC(
            data_circuit + model_circuit, self.observables)

    def call(self, input_tensor):

        # Pad input tensor to nearest power of 2 in case of amplitude encoding
        # Padded with one to avoid division by zero
        padding = self.num_in_symbols - input_tensor.shape[1]
        if padding:
            input_tensor = pad(input_tensor,
                               constant([[0, 0], [0, padding]]),
                               constant_values=1.0)

        resolved_inputs = self.input_resolver(input_tensor)
        batch_dim = shape(input_tensor)[0]

        tiled_up_circuits = repeat(self.empty_circuit,
                                   repeats=batch_dim,
                                   name="tiled_up_circuits")
        tiled_up_thetas = tile(self.theta,
                               multiples=[batch_dim, 1],
                               name="tiled_up_thetas")
        joined_vars = concat([tiled_up_thetas, resolved_inputs], axis=1)
        joined_vars = gather(joined_vars,
                             self.indices,
                             axis=1,
                             name='joined_vars')
        out = self.computation_layer([tiled_up_circuits, joined_vars])
        return out


class QCNNCong(BaseModel):
    """
		Quantum Convolutional Neural Network.
		This implementation is based on https://www.tensorflow.org/quantum/tutorials/qcnn
		"""

    def __init__(self, data_config, args=None):
        super(QCNNCong, self).__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]

        self.fm_class = self.args.get("feature_map")
        if self.fm_class is None:
            self.fm_class = "AngleMap"

        self.qcnn = QCNN(self.input_dim, self.fm_class)

    def call(self, input_tensor):
        """
        `call` takes in an input tensor, adds the cluster circuit to it, and then passes the result to
        the expectation layer
        
        Args:
            input_tensor: The input tensor to the layer.
        
        Returns:
            The expectation value of the cluster state.
        """
        x = Flatten()(input_tensor)
        out = self.qcnn(x)
        return out

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="QCNNCong")

    def get_ansatz(self):
        return [self.qcnn.ansatz]

    @staticmethod
    def add_to_argparse(parser):
        return parser


def cluster_state_circuit(bits):
    """
		Return a cluster state on the qubits in `bits`
		
		Args:
			bits: The qubits to use in the circuit.
		
		Returns:
			A circuit that creates a cluster state.
		"""
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(bits))
    for this_bit, next_bit in zip(bits, bits[1:] + [bits[0]]):
        circuit.append(cirq.CZ(this_bit, next_bit))
    return circuit


def two_qubit_unitary(bits, symbols):
    """Make a Cirq circuit that creates an arbitrary two qubit unitary."""
    circuit = cirq.Circuit()
    circuit += one_qubit_unitary(bits[0], symbols[0:3])
    circuit += one_qubit_unitary(bits[1], symbols[3:6])
    circuit += [cirq.ZZ(*bits)**symbols[6]]
    circuit += [cirq.YY(*bits)**symbols[7]]
    circuit += [cirq.XX(*bits)**symbols[8]]
    circuit += one_qubit_unitary(bits[0], symbols[9:12])
    circuit += one_qubit_unitary(bits[1], symbols[12:])
    return circuit


def two_qubit_pool(source_qubit, sink_qubit, symbols):
    """
		Make a Cirq circuit to do a parameterized 'pooling' operation, which
		attempts to reduce entanglement down from two qubits to just one.
		
		Args:
			source_qubit: the qubit that is being measured
			sink_qubit: the qubit that will be measured
			symbols: a list of 6 symbols, each of which is either 'X', 'Y', or 'Z'.
		
		Returns:
			A circuit that performs a two-qubit pooling operation.
		"""
    pool_circuit = cirq.Circuit()
    sink_basis_selector = one_qubit_unitary(sink_qubit, symbols[0:3])
    source_basis_selector = one_qubit_unitary(source_qubit, symbols[3:6])
    pool_circuit.append(sink_basis_selector)
    pool_circuit.append(source_basis_selector)
    pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
    pool_circuit.append(sink_basis_selector**-1)
    return pool_circuit


def quantum_conv_circuit(bits, symbols):
    """
		Quantum Convolution Layer. Return a Cirq circuit with the 
		cascade of `two_qubit_unitary` applied to all pairs of 
		qubits in `bits`.
		
		Args:
			bits: a list of qubits
			symbols: a list of symbols that will be used to represent the qubits.
		
		Returns:
			A circuit with the two qubit unitary applied to the first two qubits, then the second two qubits,
		then the third two qubits, then the first and last qubits.
		"""
    circuit = cirq.Circuit()
    for first, second in zip(bits[0::2], bits[1::2]):
        circuit += two_qubit_unitary([first, second], symbols)
    for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
        circuit += two_qubit_unitary([first, second], symbols)
    return circuit


def quantum_pool_circuit(source_bits, sink_bits, symbols):
    """
		A layer that specifies a quantum pooling operation.
		A Quantum pool tries to learn to pool the relevant information from two
		qubits onto 1.
		
		Args:
			source_bits: the qubits that will be used as the input to the pooling layer
			sink_bits: the qubits that will be measured at the end of the circuit
			symbols: a list of symbols that will be used to label the qubits in the circuit.
		
		Returns:
			A circuit with the two qubit pool gates applied to each pair of source and sink bits.
		"""
    circuit = cirq.Circuit()
    for source, sink in zip(source_bits, sink_bits):
        circuit += two_qubit_pool(source, sink, symbols)
    return circuit
