"""
This implementation is based on https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/
"""
from qml_hep_lhc.models.resnet.bottleneck import BottleneckResidual
from tensorflow.keras.layers import Dense, Activation, AveragePooling2D, Flatten, Input, add
from tensorflow.keras import Model


class ResnetV1(Model):

    def __init__(self, data_config, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.depth = self.args.get("resnet_depth", 20)

        if (self.depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n + 2 (eg 20, 32, 44 in [a])')

        self.num_res_blocks = int((self.depth - 2) / 6)
        self.input_dim = data_config["input_dims"]
        self.num_classes = len(data_config["mapping"])

        num_filters = 16
        self.res_block1 = BottleneckResidual()
        self.res_blocks = []

        for stage in range(3):
            for res_block in range(self.num_res_blocks):
                block = []
                strides = 1
                if stage > 0 and res_block == 0:
                    strides = 2
                block.append(
                    BottleneckResidual(num_filters=num_filters,
                                       strides=strides))
                block.append(
                    BottleneckResidual(num_filters=num_filters,
                                       activation=None))

                if stage > 0 and res_block == 0:
                    block.append(
                        BottleneckResidual(num_filters=num_filters,
                                           kernel_size=1,
                                           strides=strides,
                                           activation=None,
                                           batch_normalization=False))
                self.res_blocks.append(block.copy())
                del block

            num_filters *= 2

        self.activation_layer = Activation('relu')
        self.pooling = AveragePooling2D(pool_size=(8, 8))
        self.flatten = Flatten()
        self.dense = Dense(self.num_classes,
                           activation='softmax',
                           kernel_initializer='he_normal')

    def call(self, input_tensor):
        num_filters = 16
        x = self.res_block1(input_tensor)
        for stage in range(3):

            for res_block in range(self.num_res_blocks):

                y = self.res_blocks[stage * self.num_res_blocks +
                                    res_block][0](x)

                y = self.res_blocks[stage * self.num_res_blocks +
                                    res_block][1](y)

                if stage > 0 and res_block == 0:
                    x = self.res_blocks[stage * self.num_res_blocks +
                                        res_block][2](x)

                x = add([x, y])
                x = self.activation_layer(x)
            num_filters *= 2

        x = self.pooling(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x), name="ResnetV1")
