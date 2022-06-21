from tensorflow.keras import Model, Input
import sympy
from tensorflow import string
import tensorflow_quantum as tfq
import cirq
from qml_hep_lhc.models.base_model import BaseModel
import tensorflow_quantum
from tensorflow.keras.layers import Layer, Flatten
import numpy as np
from qml_hep_lhc.encodings import AngleMap
from tensorflow import random_uniform_initializer, Variable, constant, shape, repeat, tile, concat, gather
from qml_hep_lhc.utils import _import_class


class QLinear(Layer):

    def __init__(self, input_dim, fm_class):
        super(QLinear, self).__init__()

        self.dim = np.prod(input_dim)
        self.fm_class = fm_class

        # Prepare qubits
        self.n_qubits = self.dim

        if self.fm_class == 'AmplitudeMap':
            self.n_qubits = int(np.log2(self.dim))

        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)
        self.readout = cirq.GridQubit(-1, -1)
        self.observables = [cirq.Z(self.readout)]

        var_symbols = sympy.symbols(f'qnn0:{2*self.n_qubits}')
        self.var_symbols = np.asarray(var_symbols).reshape((self.n_qubits, 2))

        in_symbols = sympy.symbols(f'x0:{self.dim}')
        self.in_symbols = np.asarray(in_symbols).reshape((self.dim))

    def build(self, input_shape):
        circuit = cirq.Circuit()

        # Prepare the readout qubit
        circuit.append(cirq.X(self.readout))
        circuit.append(cirq.H(self.readout))

        fm = _import_class(f"qml_hep_lhc.encodings.{self.fm_class}")()
        circuit += fm.build(self.qubits, self.in_symbols)

        for i, qubit in enumerate(self.qubits):
            circuit.append(cirq.XX(qubit, self.readout)**self.var_symbols[i, 0])
        for i, qubit in enumerate(self.qubits):
            circuit.append(cirq.ZZ(qubit, self.readout)**self.var_symbols[i, 1])

        # Finally, prepare the readout qubit.
        circuit.append(cirq.H(self.readout))

        self.var_symbols = list(self.var_symbols.flat)
        self.in_symbols = list(self.in_symbols.flat)

        var_init = random_uniform_initializer(minval=-np.pi / 2,
                                              maxval=np.pi / 2)
        self.theta = Variable(initial_value=var_init(
            shape=(1, len(self.var_symbols)), dtype="float32"),
                              trainable=True,
                              name="thetas")

        # Define explicit symbol order
        symbols = [str(symb) for symb in self.var_symbols + self.in_symbols]
        self.indices = constant([symbols.index(a) for a in sorted(symbols)])

        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(
            circuit, self.observables)

    def call(self, input_tensor):
        batch_dim = shape(input_tensor)[0]

        tiled_up_circuits = repeat(self.empty_circuit,
                                   repeats=batch_dim,
                                   name="tiled_up_circuits")
        tiled_up_thetas = tile(self.theta,
                               multiples=[batch_dim, 1],
                               name="tiled_up_thetas")
        joined_vars = concat([tiled_up_thetas, input_tensor], axis=-1)
        joined_vars = gather(joined_vars,
                             self.indices,
                             axis=-1,
                             name='joined_vars')
        out = self.computation_layer([tiled_up_circuits, joined_vars])
        return out


class QNN(BaseModel):
    """
    Quantum Neural Network.
    This implementation is based on https://www.tensorflow.org/quantum/tutorials/mnist
    """

    def __init__(self, data_config, args=None):
        super().__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]
        self.n_qubits = np.prod(self.input_dim)
        fm_class = self.args.get("feature_map")
        if fm_class is None:
            fm_class = "AngleMap"
        self.qlinear = QLinear(self.input_dim, fm_class)

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

    @staticmethod
    def add_to_argparse(parser):
        return parser