from qml_hep_lhc.data.base_data_module import BaseDataModule
import numpy as np
from qml_hep_lhc.data.preprocessor import DataPreprocessor
from sklearn.utils import shuffle
from qml_hep_lhc.data.utils import extract_samples


class QuarkGluon(BaseDataModule):

    def __init__(self, args=None) -> None:
        super().__init__(args)

        self.dims = (39, 39, 1)
        self.output_dims = (1,)
        self.mapping = list(range(2))

        self.classes = ['Quark', 'Gluon']

        # Parse args
        self.args['is_binary_data'] = True
        self.percent_samples = self.args.get("percent_samples", 1.0)
        self.filename = self.data_dir / 'quark_gluon_med.npz'

    def prepare_data(self):
        # Load the data

        # Extract the data
        data = np.load(self.filename, allow_pickle=True)
        self.x_train, self.y_train = data['x_train'], data['y_train']
        self.x_test, self.y_test = data['x_test'], data['y_test']

    def __repr__(self) -> str:
        return super().__repr__("Quark Gluon")

    @staticmethod
    def add_to_argparse(parser):
        return parser