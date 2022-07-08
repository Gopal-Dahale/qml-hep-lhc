from .utils import cnot_entangling_circuit, one_qubit_unitary
import cirq
import sympy as sp
import numpy as np


class Chen:
    """
	Ansatz based on
	
	S. Y. C. Chen, T. C. Wei, C.Zhang, H. Yu and S. Yoo, 
	Quantum convolutional neural networks for high energy 
	physics data analysis, Phys. Rev. Res. \textbf{4} (2022) no.1, 013231
	doi:10.1103/PhysRevResearch.4.013231 
	"""

    def __init__(self):
        super().__init__()

    def build(self, qubits, feature_map, n_layers, drc, in_symbols=None):
        """
		Builds the circuit for the Chen ansatz.
		
		Args:
			qubits: the qubits to use
			feature_map: the feature map to use.
			n_layers: number of layers in the circuit
			drc: boolean, whether to use the re-encoding layer
			in_symbols: the input symbols to the circuit.
		
		Returns:
			The circuit, the symbols, and the observable.
				"""
        # Observables
        Z = cirq.PauliString(cirq.Z(qubits[0]))
        I = cirq.PauliString(cirq.I(qubits[0]))
        observable = [-0.5 * Z + 0.5 * I]

        n_qubits = len(qubits)
        # Sympy symbols for variational angles
        var_symbols = sp.symbols(f'qconv0:{3*n_qubits*n_layers}')
        var_symbols = np.asarray(var_symbols).reshape((n_layers, n_qubits, 3))

        circuit = cirq.Circuit()
        for l in range(n_layers):
            circuit += cnot_entangling_circuit(qubits)
            circuit += cirq.Circuit([
                one_qubit_unitary(q, var_symbols[l, i])
                for i, q in enumerate(qubits)
            ])
            # Re-encoding layer
            if drc and (l < n_layers - 1):
                circuit += feature_map.build(qubits, in_symbols[l])

        return circuit, list(var_symbols.flat), observable
