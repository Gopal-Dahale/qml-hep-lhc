from tensorflow.keras import Model, Input
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq


class CircuitLayerBuilder():
    """
    Circuit Layer Builder
    """

    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


class QNN(Model):
    """
    Quantum Neural Network.
    This implementation is based on https://www.tensorflow.org/quantum/tutorials/mnist
    """

    def __init__(self, data_config, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]

        # Prepare qubits
        data_qubits = cirq.GridQubit.rect(self.input_dim[0], self.input_dim[1])
        readout = cirq.GridQubit(-1, -1)
        circuit = cirq.Circuit()

        # Prepare the readout qubit
        circuit.append(cirq.X(readout))
        circuit.append(cirq.H(readout))

        builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)

        # Then add layers (experiment by adding more).
        builder.add_layer(circuit, cirq.XX, "xx1")
        builder.add_layer(circuit, cirq.ZZ, "zz1")

        # Finally, prepare the readout qubit.
        circuit.append(cirq.H(readout))

        self.model_circuit = circuit
        self.model_readout = cirq.Z(readout)

        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        self.expectation_layer = tfq.layers.PQC(self.model_circuit,
                                                operators=self.model_readout)

    def call(self, input_tensor):
        """
        The function takes in an input tensor and returns the expectation of the input tensor
        
        Args:
          input_tensor: The input tensor to the layer.
        
        Returns:
          The expectation of the input tensor.
        """
        expectation = self.expectation_layer(input_tensor)
        return expectation

    def build_graph(self):
        x = Input(shape=(), dtype=tf.string)
        return Model(inputs=[x], outputs=self.call(x), name="QNN")
