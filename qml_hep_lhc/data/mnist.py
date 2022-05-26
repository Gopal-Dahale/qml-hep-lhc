import tensorflow as tf

class Mnist:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes=10)