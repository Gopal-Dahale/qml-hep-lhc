from qml_hep_lhc.data.base_data_module import BaseDataModule, _download_raw_dataset
import numpy as np
from qml_hep_lhc.data.preprocessor import DataPreprocessor
from sklearn.utils import shuffle
from qml_hep_lhc.data.constants import ELECTRON_PHOTON_DATASET_URL


class ElectronPhoton(BaseDataModule):
    """
    Electron Photon Data module
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)

        self.dims = (32, 32, 1)
        self.output_dims = (1,)
        self.mapping = range(2)

        # Parse args
        self.args['is_binary_data'] = True
        self.percent_samples = self.args.get("percent_samples", 1.0)
        self.filename = self.data_dir / 'electron_photon.npz'

    def prepare_data(self):
        # Load the data
        if not self.filename.exists():
            _download_raw_dataset(ELECTRON_PHOTON_DATASET_URL, self.filename)

        # Extract the data
        data = np.load(self.filename, allow_pickle=True)
        self.x_train, self.y_train = data['x_train'], data['y_train']
        self.x_test, self.y_test = data['x_test'], data['y_test']

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
        self.x_train, self.y_train = preprocessor.process(
            self.x_train, self.y_train, self.config())
        self.x_test, self.y_test = preprocessor.process(self.x_test,
                                                        self.y_test,
                                                        self.config())
        # Set the configuration
        self.dims = preprocessor.dims
        self.output_dims = preprocessor.output_dims
        self.mapping = preprocessor.mapping

    def __repr__(self) -> str:
        return super().__repr__("Electron Photon")
