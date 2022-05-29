import numpy as np
import cirq


def binary_encoding(x, threshold=0.5):
    return np.array(x > threshold, dtype=np.float32)


def convert_to_circuit(image, size):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(size[0], size[1])
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit
