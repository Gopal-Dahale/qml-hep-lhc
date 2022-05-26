import tensorflow as tf

class Mlp(tf.keras.Model):
    def __init__(self, num_classes):
        super(Mlp, self).__init__()
        # define all layers in init
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, input_tensor, training=False):
        x = self.flatten(input_tensor)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(28,28,1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))