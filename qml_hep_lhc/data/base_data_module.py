from lib2to3.pytree import Base
from pathlib import Path
BATCH_SIZE = 128

class BaseDataModule():
    def __init__(self, args = None) -> None:
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)   

        self.dims = None
        self.output_dims= None
        self.mapping= None
        self.x_train = None
        self.y_train = None
        self.x_val= None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        
    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "datasets"

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}