from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten
from qml_hep_lhc.models import QCNN
from qml_hep_lhc.layers import QConv2D, TwoLayerPQC, NQubitPQC
import numpy as np
from qml_hep_lhc.layers.utils import get_count_of_qubits, get_num_in_symbols
from qml_hep_lhc.utils import _import_class


class FQCNN(QCNN):
    """
	General Quantum Convolutional Neural Network
	"""

    def __init__(self, data_config, args=None):
        super(FQCNN, self).__init__(data_config, args)
        self.args = vars(args) if args is not None else {}

        input_shape = [None] + list(self.input_dim)

        self.qconv2d_1 = QConv2D(
            filters=1,
            kernel_size=3,
            strides=1,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            sparse=self.sparse,
            padding="same",
            cluster_state=self.cluster_state,
            fm_class=self.fm_class,
            ansatz_class=self.ansatz_class,
            drc=self.drc,
            name='qconv2d_1',
        )

        input_shape = self.qconv2d_1.compute_output_shape(input_shape)

        self.qconv2d_2 = QConv2D(
            filters=1,
            kernel_size=3,
            strides=1,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            sparse=self.sparse,
            padding="same",
            cluster_state=self.cluster_state,
            fm_class=self.fm_class,
            ansatz_class=self.ansatz_class,
            drc=self.drc,
            name='qconv2d_2',
        )

        input_shape = self.qconv2d_2.compute_output_shape(input_shape)

        if self.ansatz_class == 'NQubit':
            self.vqc = NQubitPQC(
                n_qubits=self.n_qubits,
                cluster_state=self.cluster_state,
                n_layers=self.n_layers,
                sparse=self.sparse,
            )
        else:
            n_qubits = get_count_of_qubits(self.fm_class,
                                           np.prod(input_shape[1:]))
            n_inputs = get_num_in_symbols(self.fm_class,
                                          np.prod(input_shape[1:]))

            feature_map = _import_class(
                f"qml_hep_lhc.encodings.{self.fm_class}")()
            ansatz = _import_class(
                f"qml_hep_lhc.ansatzes.{self.ansatz_class}")()

            self.vqc = TwoLayerPQC(
                n_qubits=n_qubits,
                n_inputs=n_inputs,
                feature_map=feature_map,
                ansatz=ansatz,
                cluster_state=self.cluster_state,
                n_layers=self.n_layers,
                drc=self.drc,
            )

    def call(self, input_tensor):
        x = self.qconv2d_1(input_tensor)
        x = self.qconv2d_2(x)
        x = Flatten()(x)
        x = self.vqc(x)
        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x],
                     outputs=self.call(x),
                     name=f"FQCNN-{self.fm_class}-{self.ansatz_class}")
