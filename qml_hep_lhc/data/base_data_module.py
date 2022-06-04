from pathlib import Path
from tqdm import tqdm
from urllib.request import urlretrieve
import tensorflow_quantum as tfq
from qml_hep_lhc.data.utils.q_utils import binary_encoding, angle_encoding
import numpy as np
from tabulate import tabulate


class BaseDataModule():
    """
    The BaseDataModule class is a base class for all the datasets. It contains the basic functions that
    are common to all the datasets
    """

    def __init__(self, args=None) -> None:
        self.args = vars(args) if args is not None else {}

        # Set the data directories
        self.data_dir = self.data_dirname() / "downloaded"
        self.processed_data_dir = self.data_dirname() / "processed"

        # Create data directories if does not exist
        if not self.data_dir.exists():
            self.data_dir.mkdir()
        if not self.processed_data_dir.exists():
            self.processed_data_dir.mkdir()

        # Set the data files
        self.dims = None
        self.output_dims = None
        self.mapping = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # Set the quantum data files
        self.q_dims = None
        self.q_output_dims = None
        self.q_mapping = None
        self.qx_train = None
        self.qx_test = None

        # Parse arguments
        self._quantum = self.args.get("quantum", False)
        self._binary_encoding = self.args.get("binary_encoding", False)
        self._angle_encoding = self.args.get("angle_encoding", False)
        self._threshold = self.args.get("threshold", 0.5)

    @classmethod
    def data_dirname(cls):
        """
        It returns the path to the directory containing the datasets
        
        Args:
          cls: the class of the dataset.
        
        Returns:
          The path to the datasets folder.
        """
        return Path(__file__).resolve().parents[2] / "datasets"

    def config(self):
        """
        Return important settings of the classical dataset, which will be passed to instantiate models.
        """
        return {
            "input_dims": self.dims,
            "output_dims": self.output_dims,
            "mapping": self.mapping
        }

    def q_data_config(self):
        """
        Return important settings of the quantum dataset, which will be passed to instantiate models.
        """
        return {
            "input_dims": self.q_dims,
            "output_dims": self.q_output_dims,
            "mapping": self.q_mapping
        }

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """
        pass

    def setup(self):
        """
        Split into train, val, test, and set dims and other tasks.
        """
        pass

    def encoding_data_to_quantum_circuit(self):
        """
        The function takes the training and test data and converts it into quantum circuits.
        Two modes of operation: binary encoding and angle encoding. 
        """
        if self._quantum:
            image_size = self.x_train.shape[
                1:]  # (height, width, channels = 1)

            # Encoding the data as quantum circuits

            # The data is converted into a binary representation which
            # is then encoded into a quantum circuit using NOT gates.
            if self._binary_encoding:

                # Convert the data to binary representation
                self.qx_train = np.array(self.x_train > self._threshold,
                                         dtype=np.float32)
                self.qx_test = np.array(self.x_test > self._threshold,
                                        dtype=np.float32)

                # Encode the data into quantum circuits
                self.qx_train = [
                    binary_encoding(x, image_size) for x in self.qx_train
                ]
                self.qx_test = [
                    binary_encoding(x, image_size) for x in self.qx_test
                ]

                self.q_dims = (image_size[0], image_size[1])  # (height, width)

            # The data is is assumed to be in angle representation and then
            # encoded into a quantum circuit using Rx gates.
            elif self._angle_encoding:
                self.qx_train = [
                    angle_encoding(x, image_size) for x in self.x_train
                ]
                self.qx_test = [
                    angle_encoding(x, image_size) for x in self.x_test
                ]

                self.q_dims = (1, image_size[0] * image_size[1]
                               )  # (1, height*width)

            # Convert these Cirq circuits to tensors for tfq
            self.qx_train = tfq.convert_to_tensor(self.qx_train)
            self.qx_test = tfq.convert_to_tensor(self.qx_test)

            self.q_output_dims = (1, )  # Binary output
            self.q_mapping = self.mapping  # The mapping is the same as the classical one

    def __repr__(self, name) -> str:
        """
        Print info about the dataset.
        
        Args:
          name: The name of the dataset.
        """
        headers = ["Data", "Train size", "Test size", "Dims"]
        rows = [["X", self.x_train.shape, self.x_test.shape, self.dims],
                ["y", self.y_train.shape, self.y_test.shape, self.output_dims]]

        data = f"Dataset :{name}\n"
        data += tabulate(rows, headers, tablefmt="fancy_grid") + "\n"

        # Print quantum data if it exists
        if self._quantum:
            q_rows = [[
                "QX", self.qx_train.shape, self.qx_test.shape, self.q_dims
            ], [
                "Qy", self.y_train.shape, self.y_test.shape, self.q_output_dims
            ]]

            data += "Quantum Dataset\n"
            data += tabulate(q_rows, headers, tablefmt="fancy_grid") + "\n"

        return data


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
    """
    It downloads a file from a URL to a local file
    
    Args:
      url: The URL of the file to download.
      filename: The name of the file to download to.
    """
    print(f"Downloading raw dataset from {url} to {filename}")
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024,
                  miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec
