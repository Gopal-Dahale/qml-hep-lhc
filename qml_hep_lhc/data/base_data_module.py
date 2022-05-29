from pathlib import Path
from tqdm import tqdm
from urllib.request import urlretrieve
import tensorflow_quantum as tfq
from qml_hep_lhc.data.utils.q_utils import binary_encoding, angle_encoding
import numpy as np

class BaseDataModule():

    def __init__(self, args=None) -> None:
        self.args = vars(args) if args is not None else {}

        self.data_dir = self.data_dirname() / "downloaded"
        self.processed_data_dir = self.data_dirname() / "processed"

        if not self.data_dir.exists():
            self.data_dir.mkdir()
        if not self.processed_data_dir.exists():
            self.processed_data_dir.mkdir()

        self.dims = None
        self.output_dims = None
        self.mapping = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # quantum data
        self.q_dims = None
        self.q_output_dims = None
        self.q_mapping = None

        self.qx_train = None
        self.qx_test = None

        self._quantum = self.args.get("quantum", False)
        self._binary_encoding = self.args.get("binary_encoding", True)
        self._angle_encoding = self.args.get("angle_encoding", False)
        self._threshold = self.args.get("threshold", 0.5)

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "datasets"

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {
            "input_dims": self.dims,
            "output_dims": self.output_dims,
            "mapping": self.mapping
        }

    def q_data_config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {
            "input_dims": self.q_dims,
            "output_dims": self.q_output_dims,
            "mapping": self.q_mapping
        }

    def prepare_data(self):
        pass

    def setup(self):
        pass

    def encoding_data_to_quantum_circuit(self):
        if self._quantum:
            image_size = self.x_train.shape[1:]
           
            # Encoding the data as quantum circuits
            if self._binary_encoding:
                self.qx_train = np.array(self.x_train > self._threshold, dtype=np.float32)
                self.qx_test = np.array(self.x_test > self._threshold, dtype=np.float32)

                self.qx_train = [
                    binary_encoding(x, image_size)
                    for x in self.qx_train
                ]
                self.qx_test = [
                    binary_encoding(x, image_size)
                    for x in self.qx_test
                ]

                self.q_dims = (image_size[0] , image_size[1]) 
            
            elif self._angle_encoding:
                self.qx_train = [angle_encoding(x, image_size) for x in self.x_train]
                self.qx_test = [angle_encoding(x, image_size) for x in self.x_test]
                self.q_dims = (1,image_size[0]*image_size[1])

           

            self.qx_train = tfq.convert_to_tensor(self.qx_train)
            self.qx_test = tfq.convert_to_tensor(self.qx_test)

            
            print("q_dims", self.q_dims)
            self.q_output_dims = (1, )
            self.q_mapping = self.mapping

    def __repr__(self, name) -> str:
        data = f"{name} dataset"+ "\n" + \
            f"Train/test sizes: {self.x_train.shape}, {self.x_test.shape}\n"+\
            f"Train/test labels: {self.y_train.shape}, {self.y_test.shape}\n"

        q_data = ""
        if self._quantum:
            q_data = "Quantum dataset"+ "\n" + \
                f"Train/test sizes: {self.qx_train.shape}, {self.qx_test.shape}\n"+\
                f"Train/test labels: {self.y_train.shape}, {self.y_test.shape}\n" \
                f"Quantum data config"+ "\n" + \
                f"input_dims: {self.q_dims}\n"+\
                f"output_dims: {self.q_output_dims}\n"+\
                f"mapping: {self.q_mapping}\n"

        return data + q_data


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * bsize -
                    self.n)  # will also set self.n = b * bsize


def _download_raw_dataset(url, filename):
    print(f"Downloading raw dataset from {url} to {filename}")
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024,
                  miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec
