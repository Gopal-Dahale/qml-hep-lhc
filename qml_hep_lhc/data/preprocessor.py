import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA


class DataPreprocessor():
    """
    Data Preprocessing Module
    """

    def __init__(self, data, args, data_config) -> None:

        # Load the data and arguments
        self.x_train = data["x_train"]
        self.y_train = data["y_train"]
        self.x_test = data["x_test"]
        self.y_test = data["y_test"]
        self.args = args

        # Load the data config
        self.dims = data_config["input_dims"]
        self.output_dims = data_config["output_dims"]
        self.mapping = data_config["mapping"]

        # Parse args
        self._labels_to_categorical = self.args.get("labels_to_categorical",
                                                    False)
        self._normalize = self.args.get("normalize", False)
        self._standardize = self.args.get("standardize", False)
        self._resize = self.args.get("resize", None)
        self._binary_data = self.args.get("binary_data", None)
        self._hinge_labels = self.args.get("hinge_labels", False)
        self._is_binary_data = self.args.get("is_binary_data", False)
        self._pca = self.args.get("pca", None)

        # Handling binary data case for hinge labels
        if self._is_binary_data is False:
            if self._hinge_labels and self._binary_data is None:
                raise ValueError("Hinge labels requires binary data")
        if self._is_binary_data:
            self._binary_data = None

    def standardize(self):
        """
        Standardize features by removing the mean and scaling to unit variance.
        """
        print("Standardizing data...")

        scaler = StandardScaler()
        img_size = self.dims

        self.x_train = scaler.fit_transform(
            self.x_train.reshape(-1, img_size[0] * img_size[1])).reshape(
                -1, img_size[0], img_size[1])
        self.x_test = scaler.transform(
            self.x_test.reshape(-1, img_size[0] * img_size[1])).reshape(
                -1, img_size[0], img_size[1])

    def normalize(self):
        """
        Scale input vectors individually to unit norm (vector length).
        """
        print("Normalizing data...")

        img_size = self.dims
        self.x_train = normalize(
            self.x_train.reshape(-1, img_size[0] * img_size[1])).reshape(
                -1, img_size[0], img_size[1])
        self.x_test = normalize(
            self.x_test.reshape(-1, img_size[0] * img_size[1])).reshape(
                -1, img_size[0], img_size[1])

    def resize(self):
        """
        It resizes the training and testing data to the size specified in the constructor
        """
        print("Resizing data...")

        self.x_train = tf.image.resize(self.x_train, self._resize).numpy()
        self.x_test = tf.image.resize(self.x_test, self._resize).numpy()
        self.dims = self.x_train.shape[1:]

    def labels_to_categorical(self):
        """
        It converts the labels to categorical data
        """
        print("Converting labels to categorical...")

        self.y_train = tf.keras.utils.to_categorical(self.y_train,
                                                     num_classes=len(
                                                         self.mapping))
        self.y_test = tf.keras.utils.to_categorical(self.y_test,
                                                    num_classes=len(
                                                        self.mapping))
        self.output_dims = len(self.mapping)

    def binary_data(self):
        """
        It takes the data and filters it so that only the data that contains binary classes
        """
        print("Binarizing data...")

        if self._is_binary_data is False:

            # Get the binary classes
            d1 = self._binary_data[0]
            d2 = self._binary_data[1]

            # Extract binary data
            self.x_train, self.y_train = binary_filter(d1, d2, self.x_train,
                                                       self.y_train)
            self.x_test, self.y_test = binary_filter(d1, d2, self.x_test,
                                                     self.y_test)
            self.mapping = [d1, d2]

    def hinge_labels(self):
        """
        It converts the labels to hinge labels
        """
        print("Converting labels to hinge labels...")

        self.y_train = 2 * self.y_train - 1
        self.y_test = 2 * self.y_test - 1
        self.output_dims = (1, )

    def pca(self, n_components=16):
        """
        Performs Principal component analysis (PCA) on the data.

        Args:
          n_components: Number of components to keep. If n_components is not set all components are
        kept:. Defaults to 16
        """
        print("Performing PCA on data...")

        pca_obj = PCA(n_components)
        pca_obj.fit(self.x_train.reshape(-1, self.dims[0] * self.dims[1]))
        print("Cumulative sum for train:",
              np.cumsum(pca_obj.explained_variance_ratio_ * 100)[-1])
        self.x_train = pca_obj.transform(
            self.x_train.reshape(-1, self.dims[0] * self.dims[1]))

        pca_obj.fit(self.x_test.reshape(-1, self.dims[0] * self.dims[1]))
        print("Cumulative sum for test:",
              np.cumsum(pca_obj.explained_variance_ratio_ * 100)[-1])
        self.x_test = pca_obj.transform(
            self.x_test.reshape(-1, self.dims[0] * self.dims[1]))
        self.dims = (n_components, 1)

    def process(self):
        """
        Data processing pipeline.
        """
        if self._binary_data and len(self._binary_data) == 2:
            self.binary_data()
        if self._resize is not None and len(self._resize) == 2:
            self.resize()
        if self._pca is not None:
            self.pca(self._pca)

        if self._standardize:
            self.standardize()
        if self._normalize:
            self.normalize()

        if self._labels_to_categorical:
            self.labels_to_categorical()
        if self._hinge_labels:
            self.hinge_labels()

        # Add new axis
        self.x_train, self.x_test = self.x_train[..., np.newaxis], self.x_test[
            ..., np.newaxis]


def binary_filter(d1, d2, x, y):
    """
    It takes a dataset and two labels, and returns a dataset with only those two labels
    
    Args:
      d1: the first digit to filter for
      d2: the second digit to keep
      x: the data
      y: the labels
    
    Returns:
      the x and y values that are either d1 or d2.
    """
    keep = (y == d1) | (y == d2)
    x, y = x[keep], y[keep]
    y = (y == d1)
    return x, y
