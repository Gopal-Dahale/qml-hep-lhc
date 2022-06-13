import numpy as np
import cirq
from tensorflow_quantum import convert_to_tensor


class AngleMap():

    def __init__(self, gate='rx'):
        if gate not in ['rx', 'ry', 'rz']:
            raise ValueError('gate must be one of rx, ry, rz')
        self.gate = gate

    def _encode(self, image):
        values = np.ndarray.flatten(image)
        qubits = cirq.GridQubit.rect(1, len(values))
        circuit = cirq.Circuit()
        for i, value in enumerate(values):
            circuit.append(cirq.rx(value)(qubits[i]))
        return circuit

    def build(self, x):
        return convert_to_tensor([self._encode(image) for image in x])
