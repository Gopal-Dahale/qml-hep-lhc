from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
from qml_hep_lhc.layers.qconv2d import QConv2D
from qml_hep_lhc.models.base_model import BaseModel


class QCNNCong(BaseModel):
    """
	Quantum Convolutional Neural Network.
	This implementation is based on https://arxiv.org/abs/2012.12177
	"""

    def __init__(self, data_config, args=None):
        super(QCNNCong, self).__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]
        kernel_size = (self.input_dim[0], self.input_dim[1])
        n_layers = 3
        self.fm_class = "AngleMap"
        self.ansatz_class = "Cong"

        self.conv2d = QConv2D(
            filters=1,
            kernel_size=kernel_size,
            strides=1,
            n_layers=n_layers,
            padding="same",
            cluster_state=False,
            fm_class=self.fm_class,
            ansatz_class=self.ansatz_class,
            drc=False,
            name='conv2d',
        )

    def call(self, input_tensor):
        return self.conv2d(input_tensor)

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="QCNNCong")

    @staticmethod
    def add_to_argparse(parser):
        return parser
