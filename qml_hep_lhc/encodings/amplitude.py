import cirq
from qml_hep_lhc.utils import _import_class
import sympy as sp
from functools import reduce


class AmplitudeMap:

    def __init__(self):
        super().__init__()

    def _beta(self, s, j, x):
        index_num = (2 * j - 1) * (2**(s - 1))
        index_den = (j - 1) * (2**s)

        num_start = index_num
        num_end = index_num + 2**(s - 1)

        den_start = index_den
        den_end = index_den + 2**(s)

        if ((den_end <= den_start) or (num_end <= num_start)):
            return 0

        res = [sp.Abs(x[i])**2 for i in range(num_start, num_end, 1)]
        coeff = reduce(lambda m, n: m + n, res)
        num_coeff = sp.sqrt(coeff)

        res = [sp.Abs(x[i])**2 for i in range(den_start, den_end, 1)]
        coeff = reduce(lambda m, n: m + n, res)
        den_coeff = sp.sqrt(coeff)

        beta = 2 * sp.asin(num_coeff / den_coeff)

        return beta

    def _locate_x(self, curr_j, prev_j, length):
        curr_bin = bin(curr_j)[2:].zfill(length)
        prev_bin = bin(prev_j)[2:].zfill(length)
        return [i for i, (x, y) in enumerate(zip(curr_bin, prev_bin)) if x != y]

    def build(self, qubits, symbols):
        n = len(qubits)
        ae_ops = []
        ae_ops += [cirq.ry(self._beta(n, 1, symbols))(qubits[0])]

        for i in range(1, n):
            # We can have at most i control bits
            # Total possibilities is therefore 2^i
            controls = 2**i

            control_qubits = [qubits[c] for c in range(i + 1)]
            ae_ops += [
                cirq.ControlledGate(
                    sub_gate=cirq.ry(self._beta(n - i, controls, symbols)),
                    num_controls=len(control_qubits) - 1)(*control_qubits)
            ]

            for j in range(1, controls):
                for loc in self._locate_x(controls - j - 1, controls - j, i):
                    ae_ops += [cirq.X(qubits[loc])]

                    ae_ops += [
                        cirq.ControlledGate(sub_gate=cirq.ry(
                            self._beta(n - i, controls - j, symbols)),
                                            num_controls=len(control_qubits) -
                                            1)(*control_qubits)
                    ]

            for k in range(i):
                ae_ops += [cirq.X(qubits[k])]

        return ae_ops
