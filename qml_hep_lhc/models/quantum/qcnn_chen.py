from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Flatten
import tensorflow_quantum as tfq
import tensorflow as tf
import sympy
import cirq
import numpy as np
from qml_hep_lhc.models.quantum.utils import one_qubit_unitary
from tensorflow.math import atan, square


def entangling_circuit(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops


def encoding_circuit(qubits, symbols):
    e_ops = [
        cirq.ry(symbols[index, 0]).on(bit) for index, bit in enumerate(qubits)
    ]
    e_ops += [
        cirq.rz(symbols[index, 1]).on(bit) for index, bit in enumerate(qubits)
    ]
    return e_ops


class QuantumConv(Layer):

    def __init__(self,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='linear',
                 n_layers=1,
                 name='quantum_conv'):
        super(QuantumConv, self).__init__(name=name)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.n_layers = n_layers
        self.name = name

        self.n_qubits = num_filters * num_filters
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)
        self.observables = [cirq.Z(self.qubits[0])]

        var_symbols = sympy.symbols(f'qconv0:{3*self.n_qubits*self.n_layers}')
        self.var_symbols = np.asarray(var_symbols).reshape(
            (self.n_layers, self.n_qubits, 3))

        in_symbols = sympy.symbols(f'x0:{2*self.n_qubits}')
        self.in_symbols = np.asarray(in_symbols).reshape((self.n_qubits, 2))

    def build(self, input_shape):
        circuit = cirq.Circuit()

        circuit += encoding_circuit(self.qubits, self.in_symbols)
        for layer in range(self.n_layers):
            circuit += entangling_circuit(self.qubits)
            for bit in range(self.n_qubits):
                circuit += one_qubit_unitary(self.qubits[bit],
                                             self.var_symbols[layer, bit])

        self.var_symbols = list(self.var_symbols.flat)
        self.in_symbols = list(self.in_symbols.flat)

        var_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(initial_value=var_init(
            shape=(1, len(self.var_symbols)), dtype="float32"),
                                 trainable=True,
                                 name=self.name + "_thetas")

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(initial_value=lmbd_init,
                                dtype="float32",
                                trainable=True,
                                name="lambdas")

        # Define explicit symbol order
        symbols = [str(symb) for symb in self.var_symbols + self.in_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(
            circuit, self.observables)

    def call(self, input_tensor):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(input_tensor[0]), 0)

        x = Flatten()(input_tensor)

        x1 = atan(x)
        x2 = atan(square(x))

        tiled_up_circuits = tf.repeat(self.empty_circuit,
                                      repeats=batch_dim,
                                      name=self.name + "_tiled_up_circuits")
        tiled_up_thetas = tf.tile(self.theta,
                                  multiples=[batch_dim, 1],
                                  name=self.name + "_tiled_up_thetas")
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(
            self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])


# def quantum_conv_circuit(kernel_size):
#     num_qubits = kernel_size*kernel_size
#     symbols = sympy.symbols(f"qconv0:{3*num_qubits}")
#     symbols= np.asarray(symbols).reshape(num_qubits, 3)

#     qubits = cirq.GridQubit.rect(1, num_qubits)
#     circuit = cirq.Circuit()

#     circuit += entangling_circuit(qubits)
#     for bit in range(num_qubits):
#         circuit += one_qubit_unitary(qubits[bit], symbols[bit])

#     return circuit


class QCNNChen(Model):
    """
    Quantum Convolutional Neural Network.
    This implementation is based on https://arxiv.org/abs/2012.12177
    """

    def __init__(self, data_config, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]

    def call(self, input_tensor):
        pass