from pathlib import Path
from tqdm import tqdm
from urllib.request import urlretrieve
from tabulate import tabulate
import os
from tensorflow import argmax
import numpy as np


class BaseDataModule():
    """
    The BaseDataModule class is a base class for all the datasets. It contains the basic functions that
    are common to all the datasets
    """

    def __init__(self, args=None) -> None:
        self.args = vars(args) if args is not None else {}

        # Set the data directories
        self.data_dir = self.data_dirname() / "downloaded"
        if self.args.get("data_dir") is not None:
            self.data_dir = Path(self.args.get("data_dir"))

        self.processed_data_dir = self.data_dirname() / "processed"

        # Create data directories if does not exist
        if not self.data_dir.exists():
            os.makedirs(self.data_dir)
        if not self.processed_data_dir.exists():
            os.makedirs(self.processed_data_dir)

        # Set the data files
        self.dims = None
        self.output_dims = None
        self.mapping = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # Parse arguments
        self.batch_size = self.args.get("batch_size", 128)

        # Percent of data to use for training and testing
        self.percent_samples = self.args.get("percent_samples", 1.0)

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

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--batch-size", "-batch", type=int, default=128)
        parser.add_argument("--percent-samples",
                            "-per-samp",
                            type=float,
                            default=1.0)
        parser.add_argument("--data-dir", "-data-dir", type=str, default=None)
        return parser

    def config(self):
        """
        Return important settings of the classical dataset, which will be passed to instantiate models.
        """
        return {
            "input_dims": self.dims,
            "output_dims": self.output_dims,
            "mapping": self.mapping
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

    def __repr__(self, name) -> str:
        """
        Print info about the dataset.
        
        Args:
          name: The name of the dataset.
        """
        headers = ["Data", "Train size", "Test size", "Dims"]
        rows = [["X", self.x_train.shape, self.x_test.shape, self.dims],
                ["y", self.y_train.shape, self.y_test.shape, self.output_dims]]

        data = f"\nDataset :{name}\n"
        data += tabulate(rows, headers, tablefmt="fancy_grid") + "\n\n"

        headers = [
            "Type", "Min", "Max", "Mean", "Std", "Samples for each class"
        ]

        if len(self.y_train.shape) == 2:
            n_train_samples_per_class = [
                np.sum(argmax(self.y_train, axis=-1) == i)
                for i in (self.mapping)
            ]
            n_test_samples_per_class = [
                np.sum(argmax(self.y_test, axis=-1) == i)
                for i in (self.mapping)
            ]
        else:
            n_train_samples_per_class = [
                np.sum(self.y_train == i) for i in (self.mapping)
            ]
            n_test_samples_per_class = [
                np.sum(self.y_test == i) for i in (self.mapping)
            ]

        rows = [[
            "Train Images", f"{self.x_train.min():.2f}",
            f"{self.x_train.max():.2f}", f"{self.x_train.mean():.2f}",
            f"{self.x_train.std():.2f}", n_train_samples_per_class
        ],
                [
                    "Train Labels", f"{self.y_train.min():.2f}",
                    f"{self.y_train.max():.2f}", f"{self.y_train.mean():.2f}",
                    f"{self.y_train.std():.2f}"
                ],
                [
                    "Test Images", f"{self.x_test.min():.2f}",
                    f"{self.x_test.max():.2f}", f"{self.x_test.mean():.2f}",
                    f"{self.x_test.std():.2f}", n_test_samples_per_class
                ],
                [
                    "Test Labels", f"{self.y_test.min():.2f}",
                    f"{self.y_test.max():.2f}", f"{self.y_test.mean():.2f}",
                    f"{self.y_test.std():.2f}"
                ]]

        data += tabulate(rows, headers, tablefmt="fancy_grid") + "\n\n"

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
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


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
