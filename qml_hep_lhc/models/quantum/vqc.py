from email.policy import default
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense, MaxPool2D
from qml_hep_lhc.models.base_model import BaseModel
from qml_hep_lhc.layers import TwoLayerPQC
from qml_hep_lhc.layers.utils import get_count_of_qubits, get_num_in_symbols
import numpy as np
from qml_hep_lhc.utils import _import_class


class VQC(BaseModel):
    """
	General Quantum Convolutional Neural Network
	"""

    def __init__(self, data_config, args=None):
        super(VQC, self).__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]
        self.cluster_state = self.args.get("cluster_state")
        self.fm_class = self.args.get("feature_map")
        self.ansatz_class = self.args.get("ansatz")
        self.n_layers = self.args.get("n_layers")

        if self.fm_class is None:
            self.fm_class = "AngleMap"
        if self.ansatz_class is None:
            self.ansatz_class = "Chen"

        self.drc = self.args.get("drc")
        self.n_qubits = get_count_of_qubits(self.fm_class,
                                            np.prod(self.input_dim))
        self.n_inputs = get_num_in_symbols(self.fm_class,
                                           np.prod(self.input_dim))

        self.feature_map = _import_class(
            f"qml_hep_lhc.encodings.{self.fm_class}")()
        self.ansatz = _import_class(
            f"qml_hep_lhc.ansatzes.{self.ansatz_class}")()

        self.vqc = TwoLayerPQC(
            self.n_qubits,
            self.n_inputs,
            self.feature_map,
            self.ansatz,
            self.cluster_state,
            None,
            self.n_layers,
            self.drc,
        )

    def call(self, input_tensor):
        return self.vqc(input_tensor)

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x],
                     outputs=self.call(x),
                     name=f"VQC-{self.fm_class}-{self.ansatz_class}")

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--cluster-state",
                            action="store_true",
                            default=False)
        parser.add_argument("--feature-map", "-fm", type=str)
        parser.add_argument("--n-layers", type=int, default=1)
        parser.add_argument("--ansatz", type=str)
        parser.add_argument("--drc", action="store_true", default=False)
        return parser
