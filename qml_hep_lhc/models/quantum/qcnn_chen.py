from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense
import cirq
from qml_hep_lhc.layers.qconv2d import QConv2D
from qml_hep_lhc.models.base_model import BaseModel


def entangling_circuit(qubits):
    """
	Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
	"""
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops


class QCNNChen(BaseModel):
    """
	Quantum Convolutional Neural Network.
	This implementation is based on https://arxiv.org/abs/2012.12177
	"""

    def __init__(self, data_config, args=None):
        super(QCNNChen, self).__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]

        self.conv2d_1 = QConv2D(
            filters=1,
            kernel_size=3,
            strides=1,
            n_layers=1,
            padding="same",
            cluster_state=False,
            fm_class="DoubleAngleMap",
            ansatz_class='Chen',
            drc=False,
            name='conv2d_1',
        )

        self.conv2d_2 = QConv2D(
            filters=1,
            kernel_size=2,
            strides=1,
            n_layers=1,
            padding="same",
            cluster_state=False,
            fm_class="DoubleAngleMap",
            ansatz_class='Chen',
            drc=False,
            name='conv2d_2',
        )

        self.dense1 = Dense(8, activation='relu')
        self.dense2 = Dense(2, activation='softmax')

    def call(self, input_tensor):
        x = self.conv2d_1(input_tensor)
        x = self.conv2d_2(x)
        x = Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="QCNNChen")

    @staticmethod
    def add_to_argparse(parser):
        return parser
