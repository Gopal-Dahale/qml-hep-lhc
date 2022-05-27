import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, AveragePooling2D, Flatten, Input,  Layer
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
import numpy as np

class BottleneckResidual(Layer):
    def __init__(self,num_filters = 16,
                 kernel_size = 3,
                 strides = 1,
                 activation ='relu',
                 batch_normalization = True,conv_first=True):
        super(BottleneckResidual,self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = strides
        self.batch_normalization = batch_normalization
        self.conv_first = conv_first

        self.conv = Conv2D(num_filters,
                  kernel_size = kernel_size,
                  strides = strides,
                  padding ='same',
                  kernel_initializer ='he_normal',
                  kernel_regularizer = l2(1e-4))
        
        self.batch_norm = BatchNormalization() if batch_normalization else None
        self.activation = Activation(activation) if activation is not None else None

    def call(self, input_tensor):
        x = input_tensor
        if self.conv_first:
            x = self.conv(x)
            if self.batch_normalization:
                x = self.batch_norm(x)
            if self.activation is not None:
                x = self.activation(x)
        else:
            if self.batch_normalization:
                x = self.batch_norm(x)
            if self.activation is not None:
                x = self.activation(x)
            x = self.conv(x)
        return x

class Resnet(Model):
    def __init__(self,data_config, args = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.version = self.args.get("version", 1)
        self.depth = self.args.get("depth", 20)

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
                block.append(BottleneckResidual(num_filters = num_filters, strides = strides))
                block.append(BottleneckResidual(num_filters = num_filters, activation=None))
                
                if stage > 0 and res_block == 0:
                    block.append(BottleneckResidual(num_filters = num_filters, kernel_size=1, strides = strides, activation=None, batch_normalization=False))
                self.res_blocks.append(block.copy())
                del block

            num_filters *= 2

        self.activation_layer = Activation('relu')
        self.pooling = AveragePooling2D(pool_size=(8, 8))
        self.flatten = Flatten()
        self.dense = Dense(self.num_classes, activation='softmax', kernel_initializer='he_normal')


    def call(self,input_tensor):
        num_filters = 16
        x = self.res_block1(input_tensor)
  

        for stage in range(3):
            for res_block in range(self.num_res_blocks):
                y = self.res_blocks[stage*self.num_res_blocks + res_block][0](x)
                y = self.res_blocks[stage*self.num_res_blocks + res_block][1](y)
                if stage > 0 and res_block == 0:
                    x = self.res_blocks[stage*self.num_res_blocks + res_block][2](x)
                x = tf.keras.layers.add([x, y])
                x = self.activation_layer(x)
            num_filters *= 2    
        
        x = self.activation_layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    
    def build_graph(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x))

# class Mlp(tf.keras.Model):
#     def __init__(self, num_classes):
#         super(Mlp, self).__init__()
#         # define all layers in init
#         self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
#         self.dense1 = tf.keras.layers.Dense(128, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

#     def call(self, input_tensor, training=False):
#         x = self.flatten(input_tensor)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         return x

#     def build_graph(self):
#         x = tf.keras.layers.Input(shape=(28,28,1))
#         return tf.keras.Model(inputs=[x], outputs=self.call(x))