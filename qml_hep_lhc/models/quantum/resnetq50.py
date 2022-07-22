from tensorflow.keras.applications import ResNet50
from qml_hep_lhc.models.base_model import BaseModel
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from qml_hep_lhc.layers.utils import get_count_of_qubits, get_num_in_symbols
from qml_hep_lhc.utils import _import_class
from qml_hep_lhc.layers import TwoLayerPQC


class ResnetQ50(BaseModel):

    def __init__(self, data_config, args=None):
        super(ResnetQ50, self).__init__(args)
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

        self.base_model = ResNet50(include_top=False,
                                   weights='imagenet',
                                   input_shape=(self.input_dim))
        self.base_model.trainable = False
        input_shape = self.base_model.compute_output_shape(input_shape)

        self.flatten = Flatten()
        input_shape = self.flatten.compute_output_shape(input_shape)

        self.droput = Dropout(0.25)
        self.dense1 = Dense(512, activation='relu')
        input_shape = self.dense1.compute_output_shape(input_shape)

        self.dense2 = Dense(16, activation='relu')
        input_shape = self.dense2.compute_output_shape(input_shape)

        if ((np.prod(input_shape[1:]) > 16) and
            (self.fm_class != "AmplitudeMap")):
            print(
                f"Will use Amplitude Map since n_qubits = {np.prod(input_shape[1:])} > 16"
            )
            self.fm_class = "AmplitudeMap"

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
        x = self.base_model(input_tensor)
        x = self.flatten(x)
        x = self.droput(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.vqc(x)

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="ResnetQ50")

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