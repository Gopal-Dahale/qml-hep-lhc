from qml_hep_lhc.data.base_data_module import BaseDataModule, _download_raw_dataset
import numpy as np
from qml_hep_lhc.data.preprocessor import DataPreprocessor
from sklearn.utils import shuffle
from qml_hep_lhc.data.utils.utils import ELECTRON_PHOTON_DATASET_URL

class ElectronPhoton(BaseDataModule):
    def __init__(self, args=None) -> None:
        super().__init__(args)

        self.dims = (32,32,1)
        self.output_dims = (1,)
        self.mapping = range(2)

        self.filename = self.data_dir / 'electron_photon.npz'
        self._binary_data = self.args.get("binary_data", None)

        self.percent_samples = self.args.get("percent_samples", 1.0)
        
        if self._binary_data is not None:
            raise ValueError("Binary data argument is not supported for electron_photon dataset as it is already binary")

    def prepare_data(self):
        if not self.filename.exists():
            _download_raw_dataset(ELECTRON_PHOTON_DATASET_URL, self.filename)
        
        data = np.load(self.filename, allow_pickle=True)
        self.x_train, self.y_train = data['x_train'], data['y_train']
        self.x_test, self.y_test = data['x_test'], data['y_test']

        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        
        # extract percent_samples of data from x_train and x_test
        self.x_train = self.x_train[:int(self.percent_samples * len(self.x_train))]
        self.y_train = self.y_train[:int(self.percent_samples * len(self.y_train))]
        self.x_test = self.x_test[:int(self.percent_samples * len(self.x_test))]
        self.y_test = self.y_test[:int(self.percent_samples * len(self.y_test))]

    def setup(self):

        preprocessor = DataPreprocessor(data={
            "x_train": self.x_train,
            "y_train": self.y_train,
            "x_test": self.x_test,
            "y_test": self.y_test
        }, args = self.args, data_config=self.config())

        preprocessor.process()

        self.x_train = preprocessor.x_train
        self.y_train = preprocessor.y_train
        self.x_test = preprocessor.x_test
        self.y_test = preprocessor.y_test

        self.dims = preprocessor.dims
        self.output_dims = preprocessor.output_dims
        self.mapping = preprocessor.mapping

        self.encoding_data_to_quantum_circuit()
                    

    def __repr__(self) -> str:
        return super().__repr__("Electron Photon")
            
        

    