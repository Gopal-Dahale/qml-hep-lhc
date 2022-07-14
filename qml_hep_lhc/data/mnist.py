from tensorflow.keras.datasets import mnist
from qml_hep_lhc.data.base_data_module import BaseDataModule
from qml_hep_lhc.data.preprocessor import DataPreprocessor
from sklearn.utils import shuffle


class MNIST(BaseDataModule):
    """
    MNIST Data module
    """

    def __init__(self, args):
        super().__init__(args)

        self.classes = list(range(10))

        self.dims = (28, 28, 1)
        self.output_dims = (1,)
        self.mapping = list(range(10))

        # Parse args
        self.args['is_binary_data'] = False
        self.filename = self.data_dir / 'mnist.npz'

    def prepare_data(self):
        # Load the data
        (self.x_train,
         self.y_train), (self.x_test,
                         self.y_test) = mnist.load_data(self.filename)

        # Shuffle the data
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        self.x_test, self.y_test = shuffle(self.x_test, self.y_test)

        # Extract percent_samples of data from x_train and x_test
        self.x_train = self.x_train[:int(self.percent_samples *
                                         len(self.x_train))]
        self.y_train = self.y_train[:int(self.percent_samples *
                                         len(self.y_train))]

        self.x_test = self.x_test[:int(self.percent_samples * len(self.x_test))]
        self.y_test = self.y_test[:int(self.percent_samples * len(self.y_test))]

    def setup(self):
        # Preprocess the data
        preprocessor = DataPreprocessor(self.args)
        self.x_train, self.y_train, self.x_test, self.y_test = preprocessor.process(
            self.x_train, self.y_train, self.x_test, self.y_test, self.config(),
            self.classes)

        # Set the configuration
        self.dims = preprocessor.dims
        self.output_dims = preprocessor.output_dims
        self.mapping = preprocessor.mapping
        self.classes = preprocessor.classes

    def __repr__(self) -> str:
        return super().__repr__("MNIST")

    @staticmethod
    def add_to_argparse(parser):
        return parser