from email.policy import default
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense, MaxPool2D
from qml_hep_lhc.models.base_model import BaseModel
from qml_hep_lhc.layers import QConv2D


class QCNN(BaseModel):
    """
	General Quantum Convolutional Neural Network
	"""

    def __init__(self, data_config, args=None):
        super(QCNN, self).__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]
        self.cluster_state = self.args.get("cluster_state")
        self.fm_class = self.args.get("feature_map")
        self.ansatz_class = self.args.get("ansatz")

        if self.fm_class is None:
            self.fm_class = "AngleMap"
        if self.ansatz_class is None:
            self.ansatz_class = "Chen"

        self.drc = self.args.get("drc")

        self.conv2d_1 = QConv2D(
            filters=1,
            kernel_size=3,
            strides=2,
            n_layers=1,
            padding="same",
            cluster_state=self.cluster_state,
            fm_class=self.fm_class,
            ansatz_class=self.ansatz_class,
            drc=self.drc,
            name='conv2d_1',
        )

        self.conv2d_2 = QConv2D(
            filters=1,
            kernel_size=3,
            strides=2,
            n_layers=1,
            padding="same",
            cluster_state=self.cluster_state,
            fm_class=self.fm_class,
            ansatz_class=self.ansatz_class,
            drc=self.drc,
            name='conv2d_2',
        )

        self.dense1 = Dense(8, activation='relu')
        self.dense2 = Dense(2, activation='softmax')
        self.pooling = MaxPool2D(pool_size=(2, 2))

    def call(self, input_tensor):
        x = self.conv2d_1(input_tensor)
        x = self.conv2d_2(x)
        x = self.pooling(x)
        x = Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="QCNN")

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--cluster-state",
                            action="store_true",
                            default=False)
        parser.add_argument("--feature-map", "-fm", type=str)
        parser.add_argument("--ansatz", type=str)
        parser.add_argument("--drc", action="store_true", default=False)
        return parser
