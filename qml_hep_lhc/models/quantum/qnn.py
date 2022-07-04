from tensorflow.keras import Model, Input
import sympy
import tensorflow_quantum as tfq
import cirq
from qml_hep_lhc.models.base_model import BaseModel
from tensorflow.keras.layers import Layer, Flatten
import numpy as np
from tensorflow import random_uniform_initializer, Variable, constant, shape, repeat, tile, concat, gather
from qml_hep_lhc.utils import _import_class
from qml_hep_lhc.models.quantum.utils import symbols_in_expr_map, resolve_formulas, get_count_of_qubits, get_num_in_symbols
from tensorflow import pad, constant


class QLinear(Layer):

    def __init__(self, input_dim, fm_class, num_layers=1, layer_type='alt-xz'):
        super(QLinear, self).__init__()

        self.fm_class = fm_class

        self.num_layers = num_layers
        self.layer_type = layer_type

        # Prepare qubits
        self.n_qubits = get_count_of_qubits(self.fm_class, input_dim)
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)
        self.readout = cirq.GridQubit(-1, -1)

        # Observables
        Z = cirq.PauliString(cirq.Z(self.readout))
        I = cirq.PauliString(cirq.I(self.readout))
        self.observables = [-0.5 * Z + 0.5 * I]

        # Sympy symbols for variational angles
        var_symbols = sympy.symbols(f'θ0:{self.num_layers*self.n_qubits}')
        self.var_symbols = np.asarray(var_symbols).reshape(
            (self.n_qubits, self.num_layers))

        # Sympy symbols for encoding angles
        self.num_in_symbols = get_num_in_symbols(self.fm_class, input_dim)
        in_symbols = sympy.symbols(f'x0:{self.num_in_symbols}')
        self.in_symbols = np.asarray(in_symbols).reshape((self.num_in_symbols))

    def _generate_gates(self):
        if self.layer_type == 'alt-xz':
            return [cirq.XX, cirq.ZZ]
        if self.layer_type == 'all-x':
            return [cirq.XX, cirq.XX]
        if self.layer_type == 'all-z':
            return [cirq.ZZ, cirq.ZZ]

    def build(self, input_shape):

        # Define data and model circuits
        data_circuit = cirq.Circuit()
        model_circuit = cirq.Circuit()

        # Prepare the readout qubit
        model_circuit.append(cirq.X(self.readout))
        model_circuit.append(cirq.H(self.readout))

        # Prepare data circuit
        self.fm = _import_class(f"qml_hep_lhc.encodings.{self.fm_class}")()
        data_circuit += self.fm.build(self.qubits, self.in_symbols)

        # Prepare model circuit
        gates = self._generate_gates()
        num_gates = len(gates)
        for i in range(self.num_layers):
            for index, qubit in enumerate(self.qubits):
                model_circuit.append(gates[i % num_gates](
                    qubit, self.readout)**self.var_symbols[index, i])

        # Finally, prepare the readout qubit.
        model_circuit.append(cirq.H(self.readout))
        self.ansatz = model_circuit

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


class QNN(BaseModel):
    """
    Quantum Neural Network.
    This implementation is based on https://www.tensorflow.org/quantum/tutorials/mnist
    """

    def __init__(self, data_config, args=None):
        super(QNN, self).__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]
        self.n_qubits = np.prod(self.input_dim)

        self.fm_class = self.args.get("feature_map")
        if self.fm_class is None:
            self.fm_class = "AngleMap"

        self.num_layers = self.args.get("--num-controlled-layers", 2)
        self.layer_type = self.args.get("--controlled-layers-type", "alt-xz")

        self.qlinear = QLinear(self.input_dim, self.fm_class, self.num_layers,
                               self.layer_type)

    def call(self, input_tensor):
        """
        The function takes in an input tensor and returns the expectation of the input tensor
        
        Args:
          input_tensor: The input tensor to the layer.
        
        Returns:
          The expectation of the input tensor.
        """
        x = Flatten()(input_tensor)
        out = self.qlinear(x)
        return out

    def build_graph(self):
        # x = Input(shape=(), dtype=string)
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="QNN")

    def get_ansatz(self):
        return [self.qlinear.ansatz]

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--num-controlled-layers", type=int, default=2)
        parser.add_argument("--controlled-layers-type",
                            type=str,
                            default="alt-xz",
                            choices=["all-x", "all-z"])
        return parser