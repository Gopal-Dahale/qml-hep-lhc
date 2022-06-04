"""
This implementation is based on https://www.tensorflow.org/quantum/tutorials/qcnn
"""
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer
import tensorflow_quantum as tfq
import tensorflow as tf
import sympy
import cirq


class QCNNCong(Model):

    def __init__(self, data_config, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]

        # Prepare qubits
        qubits = cirq.GridQubit.rect(1, self.input_dim[0] * self.input_dim[1])
        self.model_readout = cirq.Z(qubits[-1])

        circuit = cirq.Circuit()
        symbols = sympy.symbols('qconv0:63')

        # First convolution layer with pooling layer
        # Reduces 16 qubits to 8 qubits
        circuit += quantum_conv_circuit(qubits, symbols[0:15])
        circuit += quantum_pool_circuit(qubits[:8], qubits[8:], symbols[15:21])

        # Second convolution layer with pooling layer
        # Reduces 8 qubits to 4 qubits
        circuit += quantum_conv_circuit(qubits[8:], symbols[21:36])
        circuit += quantum_pool_circuit(qubits[8:12], qubits[12:],
                                        symbols[36:42])

        # Final convoluation layer with pooling layer
        # Reduces 4 qubits to 1
        circuit += quantum_conv_circuit(qubits[12:], symbols[42:57])
        circuit += quantum_pool_circuit(qubits[12:15], [qubits[15]],
                                        symbols[57:63])

        self.cluster_circuit = cluster_state_circuit(qubits)
        self.model_circuit = circuit

        self.expectation_layer = tfq.layers.PQC(self.model_circuit,
                                                operators=self.model_readout)

    def call(self, input_tensor):
        """
        `call` takes in an input tensor, adds the cluster circuit to it, and then passes the result to
        the expectation layer
        
        Args:
          input_tensor: The input tensor to the layer.
        
        Returns:
          The expectation value of the cluster state.
        """
        cluster_state = tfq.layers.AddCircuit()(input_tensor,
                                                prepend=self.cluster_circuit)
        expectation = self.expectation_layer(cluster_state)
        return expectation

    def build_graph(self):
        x = Input(shape=(), dtype=tf.string)
        return Model(inputs=[x], outputs=self.call(x), name="QCNNCong")


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


def one_qubit_unitary(bit, symbols):
    """Make a Cirq circuit enacting a rotation of the bloch sphere about the X,
    Y and Z axis, that depends on the values in `symbols`.
    """
    return cirq.Circuit(
        cirq.X(bit)**symbols[0],
        cirq.Y(bit)**symbols[1],
        cirq.Z(bit)**symbols[2])


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
