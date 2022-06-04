import numpy as np
import cirq


def binary_encoding(image, size):
    """
    It takes an image and a size, and returns a circuit that encodes the image as a binary string
    
    Args:
      image: the image to be encoded
      size: the size of the image, in pixels.
    
    Returns:
      A circuit with the X gate applied to the qubits that correspond to the pixels that are white.
    """
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(size[0], size[1])
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


def angle_encoding(image, size):
    """
    It takes an image and returns a circuit that encodes the image into a quantum state
    
    Args:
      image: the image you want to encode
      size: the size of the image
    
    Returns:
      A circuit with the values of the image encoded in the angles of the rotations.
    """
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(1, size[0] * size[1])
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        circuit.append(cirq.rx(value)(qubits[i]))
    return circuit
