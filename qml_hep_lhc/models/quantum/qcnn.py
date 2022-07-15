from email.policy import default
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense, MaxPool2D
from qml_hep_lhc.models.base_model import BaseModel
from qml_hep_lhc.layers import QConv2D, TwoLayerPQC
import numpy as np
from qml_hep_lhc.layers.utils import get_count_of_qubits, get_num_in_symbols
from qml_hep_lhc.utils import _import_class


class QCNN(BaseModel):
    """
	General Quantum Convolutional Neural Network
	"""

    def __init__(self, data_config, args=None):
        super(QCNN, self).__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]
        self.cluster_state = self.args.get("cluster_state", False)
        self.fm_class = self.args.get("feature_map", None)
        self.ansatz_class = self.args.get("ansatz", None)
        self.n_layers = self.args.get("n_layers", 1)

        if self.fm_class is None:
            self.fm_class = "AngleMap"
        if self.ansatz_class is None:
            self.ansatz_class = "Chen"

        self.drc = self.args.get("drc", False)

        input_shape = [None] + list(self.input_dim)

        self.conv2d_1 = QConv2D(
            filters=1,
            kernel_size=3,
            strides=1,
            n_layers=self.n_layers,
            padding="same",
            cluster_state=self.cluster_state,
            fm_class=self.fm_class,
            ansatz_class=self.ansatz_class,
            drc=self.drc,
            name='conv2d_1',
        )

        input_shape = self.conv2d_1.compute_output_shape(input_shape)

        self.conv2d_2 = QConv2D(
            filters=1,
            kernel_size=2,
            strides=1,
            n_layers=self.n_layers,
            padding="same",
            cluster_state=self.cluster_state,
            fm_class=self.fm_class,
            ansatz_class=self.ansatz_class,
            drc=self.drc,
            name='conv2d_2',
        )

        input_shape = self.conv2d_2.compute_output_shape(input_shape)
        n_qubits = get_count_of_qubits(self.fm_class, np.prod(input_shape[1:]))
        n_inputs = get_num_in_symbols(self.fm_class, np.prod(input_shape[1:]))

        feature_map = _import_class(f"qml_hep_lhc.encodings.{self.fm_class}")()
        ansatz = _import_class(f"qml_hep_lhc.ansatzes.{self.ansatz_class}")()

        self.vqc = TwoLayerPQC(
            n_qubits,
            n_inputs,
            feature_map,
            ansatz,
            self.cluster_state,
            None,
            self.n_layers,
            self.drc,
        )

    def call(self, input_tensor):
        x = self.conv2d_1(input_tensor)
        x = self.conv2d_2(x)
        x = Flatten()(x)
        x = self.vqc(x)
        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x],
                     outputs=self.call(x),
                     name=f"QCNN-{self.fm_class}-{self.ansatz_class}")

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--cluster-state",
                            action="store_true",
                            default=False)
        parser.add_argument("--feature-map", "-fm", type=str)
        parser.add_argument("--ansatz", type=str)
        parser.add_argument("--n-layers", type=int, default=1)
        parser.add_argument("--drc", action="store_true", default=False)
        return parser
