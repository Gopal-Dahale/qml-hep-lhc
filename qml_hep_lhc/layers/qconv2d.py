from tensorflow.keras.layers import Layer, Concatenate, Reshape, Add, Activation
from qml_hep_lhc.layers.utils import normalize_padding, normalize_tuple, convolution_iters, get_count_of_qubits, get_num_in_symbols
from qml_hep_lhc.utils import _import_class
import numpy as np
from tensorflow import pad
from qml_hep_lhc.layers import TwoLayerPQC
from qml_hep_lhc.layers import NQubitPQC


class QConv2D(Layer):
    """
    2D Quantum convolution layer (e.g. spatial convolution over images).
    This layer creates a convolution kernel that is convolved 
    with the layer input to produce a tensor of outputs. Finally,
    `activation` is applied to the outputs as well.
    """

    def __init__(
            self,
            filters=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            n_qubits=1,
            n_layers=1,
            sparse=False,
            padding='valid',
            activation='relu',
            cluster_state=False,
            fm_class='AngleMap',
            ansatz_class='Chen',
            observable=None,
            drc=False,
            name='QConv2D',
    ):

        super(QConv2D, self).__init__(name=name)

        # Filters
        if isinstance(filters, float):
            filters = int(filters)
        if filters is not None and filters <= 0:
            raise ValueError('Invalid value for argument `filters`. '
                             'Expected a strictly positive value. '
                             f'Received filters={filters}.')
        self.filters = filters

        # Num layers
        if isinstance(n_layers, float):
            n_layers = int(n_layers)
        if n_layers is not None and n_layers <= 0:
            raise ValueError('Invalid value for argument `n_layers`. '
                             'Expected a strictly positive value. '
                             f'Received n_layers={n_layers}.')
        self.n_layers = n_layers

        self.observable = observable
        self.kernel_size = normalize_tuple(kernel_size, 'kernel_size')
        self.strides = normalize_tuple(strides, 'strides')
        self.padding = normalize_padding(padding)
        self.activation = Activation(activation)
        self.cluster_state = cluster_state
        self.fm_class = fm_class
        self.ansatz_class = ansatz_class
        self.drc = drc
        self.n_qubits = n_qubits
        self.sparse = sparse

    def build(self, input_shape):

        self.iters, self.padding_constant = convolution_iters(
            input_shape[1:3], self.kernel_size, self.strides, self.padding)
        self.n_channels = input_shape[3]

        self.conv_pqcs = [[(filter, channel)
                           for channel in range(self.n_channels)]
                          for filter in range(self.filters)]

        if self.ansatz_class == 'NQubit':
            for filter in range(self.filters):
                for channel in range(self.n_channels):
                    name = f"{self.name}_{filter}_{channel}"
                    self.conv_pqcs[filter][channel] = NQubitPQC(
                        self.n_qubits, self.cluster_state, self.observable,
                        self.n_layers, self.sparse, name)
        else:
            self.n_qubits = get_count_of_qubits(self.fm_class,
                                                np.prod(self.kernel_size))
            self.n_inputs = get_num_in_symbols(self.fm_class,
                                               np.prod(self.kernel_size))

            self.feature_map = _import_class(
                f"qml_hep_lhc.encodings.{self.fm_class}")()
            self.ansatz = _import_class(
                f"qml_hep_lhc.ansatzes.{self.ansatz_class}")()

            for filter in range(self.filters):
                for channel in range(self.n_channels):
                    name = f"{self.name}_{filter}_{channel}"
                    self.conv_pqcs[filter][channel] = TwoLayerPQC(
                        self.n_qubits, self.n_inputs, self.feature_map,
                        self.ansatz, self.cluster_state, self.observable,
                        self.n_layers, self.drc, name)

    def _convolution(self, input_tensor, filter, channel):

        s = self.strides
        k = self.kernel_size

        conv_out = []
        for i in range(self.iters[0]):
            for j in range(self.iters[1]):
                x = input_tensor[:, i * s[0]:i * s[0] + k[0], j *
                                 s[1]:j * s[1] + k[1]]
                conv_out += [self.conv_pqcs[filter][channel](x)]

        conv_out = Concatenate(axis=1)(conv_out)
        conv_out = Reshape(
            (self.iters[0], self.iters[1], 3 * self.n_qubits))(conv_out)
        return conv_out

    def call(self, input_tensor):
        input_tensor = pad(input_tensor, self.padding_constant)

        if self.n_channels == 1:
            conv_out = [
                self._convolution(input_tensor[:, :, :, 0], filter, 0)
                for filter in range(self.filters)
            ]

        else:
            conv_out = [
                Add()([
                    self._convolution(input_tensor[:, :, :, c], filter, c)
                    for c in range(self.n_channels)
                ])
                for filter in range(self.filters)
            ]

        conv_out = Concatenate(axis=-1)(conv_out)
        return self.activation(conv_out)