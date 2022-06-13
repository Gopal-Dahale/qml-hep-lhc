import numpy as np
import cirq
from tensorflow_quantum import convert_to_tensor


class BinaryMap():

    def __init__(self, threshold=0):
        self.threshold = threshold

    def _encode(self, image):
        values = np.ndarray.flatten(image)
        qubits = cirq.GridQubit.rect(1, len(values))
        circuit = cirq.Circuit()
        for i, value in enumerate(values):
            if value:
                circuit.append(cirq.X(qubits[i]))
        return circuit

    def build(self, x):
        # Convert the data to binary representation
        x = np.array(x > self._threshold, dtype=np.float32)
        return convert_to_tensor([self._encode(image) for image in x])
