from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense
from qml_hep_lhc.models import QCNN
from qml_hep_lhc.layers import QConv2D


class QCNNHybrid(QCNN):
    """
	General Quantum Convolutional Neural Network
	"""

    def __init__(self, data_config, args=None):
        super(QCNNHybrid, self).__init__(data_config, args)
        self.args = vars(args) if args is not None else {}

        self.num_qconv_layers = self.args.get('num_qconv_layers', 1)
        self.qconv_dims = self.args.get('qconv_dims', [1])

        assert len(
            self.qconv_dims
        ) == self.num_qconv_layers, 'qconv_dims must be a list of length num_qconv_layers'

        self.qconvs = []
        for i, num_filters in enumerate(self.qconv_dims):
            self.qconvs.append(
                QConv2D(
                    filters=num_filters,
                    kernel_size=3,
                    strides=1,
                    n_layers=self.n_layers,
                    padding="valid",
                    cluster_state=self.cluster_state,
                    fm_class=self.fm_class,
                    ansatz_class=self.ansatz_class,
                    drc=self.drc,
                    name=f'qconv2d_{i}',
                ))

        self.num_fc_layers = self.args.get('num_fc_layers', 1)
        self.fc_dims = self.args.get('fc_dims', [8])

        assert len(
            self.fc_dims
        ) == self.num_fc_layers, 'fc_dims must be a list of length num_fc_layers'

        self.fcs = []
        for units in self.fc_dims:
            self.fcs.append(Dense(units, activation='relu'))

        self.fcs.append(Dense(self.num_classes, activation='softmax'))

    def call(self, input_tensor):
        x = input_tensor
        for qconv in self.qconvs:
            x = qconv(x)
        x = Flatten()(x)
        for fc in self.fcs:
            x = fc(x)
        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x],
                     outputs=self.call(x),
                     name=f"QCNNHybrid-{self.fm_class}-{self.ansatz_class}")
