import numpy as np
import cirq
from tensorflow_quantum import convert_to_tensor


class DoubleAngleMap:

    def __init__(self):
        super().__init__()

    def build(self, qubits, symbols):
        e_ops = [cirq.ry(symbols[i, 0])(bit) for i, bit in enumerate(qubits)]
        e_ops += [cirq.rz(symbols[i, 1])(bit) for i, bit in enumerate(qubits)]
        return e_ops
