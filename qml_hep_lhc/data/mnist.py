import tensorflow as tf
from qml_hep_lhc.data.base_data_module import BaseDataModule
import numpy as np
import collections
from qml_hep_lhc.data.preprocessor import DataPreprocessor
from sklearn.utils import shuffle


class MNIST(BaseDataModule):
    """
    MNIST Data module
    """

    def __init__(self, args):
        super().__init__(args)

        self.dims = (28, 28, 1)
        self.output_dims = (1, )
        self.mapping = range(10)

        # Parse args
        self.args['is_binary_data'] = False
        self.percent_samples = self.args.get(
            'percent_samples', 1.0)  # percent of data to use for training
        self.filename = self.data_dir / 'mnist.npz'

    def prepare_data(self):
        # Load the data
        (self.x_train,
         self.y_train), (self.x_test,
                         self.y_test) = tf.keras.datasets.mnist.load_data(
                             self.filename)

        # Shuffle the data
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        self.x_test, self.y_test = shuffle(self.x_test, self.y_test)

        # Extract percent_samples of data from x_train and x_test
        self.x_train = self.x_train[:int(self.percent_samples *
                                         len(self.x_train))]
        self.y_train = self.y_train[:int(self.percent_samples *
                                         len(self.y_train))]
        self.x_test = self.x_test[:int(self.percent_samples *
                                       len(self.x_test))]
        self.y_test = self.y_test[:int(self.percent_samples *
                                       len(self.y_test))]

    def setup(self):

        # Remove images that are labeled as belonging to both classes
        self.x_train, self.y_train = remove_contradicting(
            self.x_train, self.y_train)
        self.x_test, self.y_test = remove_contradicting(
            self.x_test, self.y_test)

        # Preprocess the data
        preprocessor = DataPreprocessor(data={
            "x_train": self.x_train,
            "y_train": self.y_train,
            "x_test": self.x_test,
            "y_test": self.y_test
        },
                                        args=self.args,
                                        data_config=self.config())

        preprocessor.process()

        # Set the data
        self.x_train = preprocessor.x_train
        self.y_train = preprocessor.y_train
        self.x_test = preprocessor.x_test
        self.y_test = preprocessor.y_test

        # Set the configuration
        self.dims = preprocessor.dims
        self.output_dims = preprocessor.output_dims
        self.mapping = preprocessor.mapping

        # Set the quantum data
        self.encoding_data_to_quantum_circuit()

    def __repr__(self) -> str:
        return super().__repr__("MNIST")


def remove_contradicting(xs, ys):
    """
    It takes a list of images and a list of labels, and returns a new list of images and labels where
    each image is only associated with one label
    
    Args:
      xs: a list of images
      ys: The labels for each image.
    """
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x, y in zip(xs, ys):
        orig_x[tuple(x.flatten())] = x
        mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for flatten_x in mapping:
        x = orig_x[flatten_x]
        labels = mapping[flatten_x]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(next(iter(labels)))
        else:
            # Throw out images that match more than one label.
            pass

    return np.array(new_x), np.array(new_y)
