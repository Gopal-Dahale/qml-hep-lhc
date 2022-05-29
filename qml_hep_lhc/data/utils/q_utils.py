import numpy as np
import cirq



def binary_encoding(image, size):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(size[0], size[1])
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit

def angle_encoding(image, size):
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(1,size[0]*size[1])
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        circuit.append(cirq.rx(value)(qubits[i]))
    return circuit