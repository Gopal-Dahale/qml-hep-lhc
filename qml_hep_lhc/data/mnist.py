import tensorflow as tf
from qml_hep_lhc.data.base_data_module import BaseDataModule
from sklearn.model_selection import train_test_split
import numpy as np
import cirq

import tensorflow_quantum as tfq

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

        self.quantum = self.args.get("quantum", False)
        self.binary_encoding = self.args.get("binary_encoding", False)
        self.threshold = self.args.get("threshold", 0.5)
        self.binary_data = self.args.get("binary_data", None)


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
        
        if self.binary_data is not None and len(self.binary_data) == 2:
            d1 = self.binary_data[0]
            d2 = self.binary_data[1]
            self.x_train, self.y_train = binary_filter(d1,d2, self.x_train, self.y_train)
            self.x_test, self.y_test = binary_filter(d1,d2, self.x_test, self.y_test)
            self.x_val, self.y_val = binary_filter(d1,d2, self.x_val, self.y_val)
            self.mapping = [d1,d2]
        
        if self.quantum:
            # Encoding the data as quantum circuits
            
            if self.binary_encoding:
                self.qx_train = np.array(self.x_train > self.threshold, dtype=np.float32)
                self.qx_test = np.array(self.x_test > self.threshold, dtype=np.float32)
                self.qx_val = np.array(self.x_val > self.threshold, dtype=np.float32)

            image_size = self.qx_train.shape[1:]
            self.qx_train = tfq.convert_to_tensor([convert_to_circuit(x, image_size) for x in self.qx_train])
            self.qx_test = tfq.convert_to_tensor([convert_to_circuit(x,image_size) for x in self.qx_test])
            self.qx_val = tfq.convert_to_tensor([convert_to_circuit(x,image_size) for x in self.qx_val])

            self.q_dims = (image_size[0],image_size[1])
            self.q_output_dims = (1,)
            self.q_mapping = self.mapping

def convert_to_circuit(image, size):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(size[0], size[1])
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit

def binary_filter(d1,d2, x,y):
    keep = (y == d1) | (y == d2)
    x,y = x[keep], y[keep]
    y = (y == d1)
    return x,y