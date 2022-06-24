import cirq
from qml_hep_lhc.utils import _import_class
import sympy as sp


class AngleMap:

    def __init__(self, gate='rx'):
        valid_gates = ['rx', 'ry', 'rz']
        if gate not in valid_gates:
            raise ValueError('gate must be one of rx, ry, rz')
        self.gate = _import_class("cirq.{}".format(gate))

    def build(self, qubits, symbols):
        e_ops = [
            self.gate(sp.pi * symbols[index])(bit)
            for index, bit in enumerate(qubits)
        ]
        return e_ops
