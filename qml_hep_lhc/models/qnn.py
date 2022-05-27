from tensorflow.keras import Model, Input
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


class QNN(Model):
    def __init__(self,data_config, args = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.input_dim = data_config["input_dims"]

        data_qubits = cirq.GridQubit.rect(self.input_dim[0], self.input_dim[1])
        readout = cirq.GridQubit(-1, -1)
        circuit = cirq.Circuit()

        builder = CircuitLayerBuilder(
            data_qubits = data_qubits,
            readout=readout)

    def call(self,input_tensor):
        pass
        # tfq.layers.PQC()

    def build_graph(self):
        x = Input(shape=self.input_dim ,dtype = tf.string)
        print(x)
        # return Model(inputs=[x], outputs=self.call(x))
