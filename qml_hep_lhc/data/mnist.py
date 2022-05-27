import tensorflow as tf
from qml_hep_lhc.data.base_data_module import BaseDataModule
from sklearn.model_selection import train_test_split
import numpy as np

DONWLOADED_DATA_DIRNAME = BaseDataModule.data_dirname()/'downloaded'

class Mnist(BaseDataModule):
    def __init__(self,args):
        super().__init__(args)
        self.data_dir = DONWLOADED_DATA_DIRNAME
        self.dims = (28,28,1)
        self.output_dims = (1,)
        self.mapping = range(10)
        
        # Get arguments from args
        self.labels_to_categorical = self.args.get("labels_to_categorical", False)
        self.normalize = self.args.get("normalize", False)
        self.resize = self.args.get("resize", None)
    

    def prepare_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data(self.data_dir/'mnist.npz')

        if self.labels_to_categorical:
            self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=10)
            self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes=10)
            
    def __repr__(self) -> str:
        data = "MNIST dataset"+ "\n" + \
            f"Train/val/test sizes: {self.x_train.shape}, {self.x_val.shape}, {self.x_test.shape}\n"+\
            f"Train/val/test labels: {self.y_train.shape}, {self.y_val.shape}, {self.y_test.shape}\n"  
            
        return data

    def setup(self):
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=42)

        if self.normalize:
            self.x_train, self.x_test ,self.x_val= self.x_train[..., np.newaxis]/255.0, self.x_test[..., np.newaxis]/255.0, self.x_val[..., np.newaxis]/255.0
        
        if self.resize is not None and len(self.resize) == 2:
            self.x_train = tf.image.resize(self.x_train, self.resize).numpy()
            self.x_test = tf.image.resize(self.x_test, self.resize).numpy()
            self.x_val = tf.image.resize(self.x_val, self.resize).numpy()
            self.dims = self.x_train.shape[1:]