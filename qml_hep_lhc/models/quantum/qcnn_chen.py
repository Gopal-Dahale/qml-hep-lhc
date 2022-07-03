from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer, Flatten, Dense, Concatenate, Reshape
import sympy
import cirq
import numpy as np
from qml_hep_lhc.models.quantum.utils import one_qubit_unitary
from qml_hep_lhc.models.base_model import BaseModel
from tensorflow import random_uniform_initializer, Variable, constant, repeat, tile, shape, concat, gather
import tensorflow_quantum as tfq
from qml_hep_lhc.models.quantum.utils import symbols_in_expr_map, resolve_formulas, get_count_of_qubits, get_num_in_symbols
from qml_hep_lhc.utils import _import_class
from tensorflow import pad, constant


def entangling_circuit(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops


class QuantumConv(Layer):

    def __init__(self,
                 name,
                 fm_class,
                 kernel_size=3,
                 strides=1,
                 activation='tanh',
                 n_layers=1):

        super(QuantumConv, self).__init__(name=name)

        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.n_layers = n_layers
        self.fm_class = fm_class

        # Prepare qubits
        self.n_qubits = get_count_of_qubits(self.fm_class,
                                            kernel_size * kernel_size)
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)

        # Observables
        Z = cirq.PauliString(cirq.Z(self.qubits[0]))
        I = cirq.PauliString(cirq.I(self.qubits[0]))
        self.observables = [-0.5 * Z + 0.5 * I]

        # Sympy symbols for variational angles
        var_symbols = sympy.symbols(f'qconv0:{3*self.n_qubits*self.n_layers}')
        self.var_symbols = np.asarray(var_symbols).reshape(
            (self.n_layers, self.n_qubits, 3))

        # Sympy symbols for encoding angles
        self.num_in_symbols = get_num_in_symbols(self.fm_class,
                                                 kernel_size * kernel_size)
        in_symbols = sympy.symbols(f'x0:{self.num_in_symbols}')
        self.in_symbols = np.asarray(in_symbols).reshape((self.num_in_symbols))

    def build(self, input_shape):
        # Define data and model circuits
        data_circuit = cirq.Circuit()
        model_circuit = cirq.Circuit()

        # Prepare data circuit
        self.fm = _import_class(f"qml_hep_lhc.encodings.{self.fm_class}")()
        data_circuit += self.fm.build(self.qubits, self.in_symbols)

        # Prepare model circuit
        for layer in range(self.n_layers):
            model_circuit += entangling_circuit(self.qubits)
            for bit in range(self.n_qubits):
                model_circuit += one_qubit_unitary(self.qubits[bit],
                                                   self.var_symbols[layer, bit])

        # Convert symbols to list
        self.var_symbols = list(self.var_symbols.flat)
        self.in_symbols = list(self.in_symbols.flat)

        # Initalize variational angles
        var_init = random_uniform_initializer(minval=-np.pi / 2,
                                              maxval=np.pi / 2)
        self.theta = Variable(initial_value=var_init(
            shape=(1, len(self.var_symbols)), dtype="float32"),
                              trainable=True,
                              name=self.name + "_thetas")

        # Flatten circuit
        data_circuit, expr_map = cirq.flatten(data_circuit)
        self.raw_symbols = symbols_in_expr_map(expr_map)
        self.expr = list(expr_map)

        print(data_circuit)

        # Define explicit symbol order and expression resolver
        symbols = [str(symb) for symb in self.var_symbols + self.expr]
        self.indices = constant([symbols.index(a) for a in sorted(symbols)])
        self.input_resolver = resolve_formulas(self.expr, self.raw_symbols)

        # Define computation layer
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.circuit = data_circuit + model_circuit
        self.computation_layer = tfq.layers.ControlledPQC(
            self.circuit, self.observables)

    def call(self, input_tensor):
        batch_dim = shape(input_tensor)[0]
        strides = self.strides
        kernel_size = self.kernel_size

        n = (input_tensor.shape[1] - kernel_size) // strides + 1
        m = (input_tensor.shape[2] - kernel_size) // strides + 1

        conv_out = []

        for i in range(n):
            for j in range(m):
                x = input_tensor[:, i * strides:i * strides + kernel_size,
                                 j * strides:j * strides + kernel_size]
                x = Flatten()(x)

                # Pad input tensor to nearest power of 2 in case of amplitude encoding
                # Padded with one to avoid division by zero
                padding = self.num_in_symbols - x.shape[1]
                if padding:
                    x = pad(x,
                            constant([[0, 0], [0, padding]]),
                            constant_values=1.0)
                resolved_inputs = self.input_resolver(x)

                tiled_up_circuits = repeat(self.empty_circuit,
                                           repeats=batch_dim,
                                           name=self.name +
                                           "_tiled_up_circuits")
                tiled_up_thetas = tile(self.theta,
                                       multiples=[batch_dim, 1],
                                       name=self.name + "_tiled_up_thetas")
                joined_vars = concat([tiled_up_thetas, resolved_inputs], axis=1)
                joined_vars = gather(joined_vars,
                                     self.indices,
                                     axis=1,
                                     name=self.name + '_joined_vars')
                out = self.computation_layer([tiled_up_circuits, joined_vars])
                conv_out += [out]

        conv_out = Concatenate(axis=1)(conv_out)
        conv_out = Reshape((n, m))(conv_out)
        return conv_out


class QCNNChen(BaseModel):
    """
    Quantum Convolutional Neural Network.
    This implementation is based on https://arxiv.org/abs/2012.12177
    """

    def __init__(self, data_config, args=None):
        super().__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]
        self.fm_class = self.args.get("feature_map")
        if self.fm_class is None:
            self.fm_class = "DoubleAngleMap"

        self.conv2d_1 = QuantumConv(kernel_size=3,
                                    strides=1,
                                    n_layers=2,
                                    fm_class=self.fm_class,
                                    name='conv2d_1')
        self.conv2d_2 = QuantumConv(kernel_size=2,
                                    strides=1,
                                    n_layers=2,
                                    fm_class=self.fm_class,
                                    name='conv2d_2')
        self.dense1 = Dense(8, activation='relu')
        self.dense2 = Dense(2, activation='softmax')

    def call(self, input_tensor):
        x = self.conv2d_1(input_tensor)
        x = self.conv2d_2(x)
        x = Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="QCNNChen")

    @staticmethod
    def add_to_argparse(parser):
        return parser