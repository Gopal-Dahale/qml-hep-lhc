import numpy as np
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.decomposition import PCA
from numba import njit, prange
from tensorflow import image
from tensorflow.keras.utils import to_categorical
from argparse import Action
from qml_hep_lhc.utils import ParseAction


class DataPreprocessor():
    """
    Data Preprocessing Module
    """

    def __init__(self, args=None) -> None:

        # Load the data and arguments
        self.args = args if args is not None else {}

        # Parse args
        self._labels_to_categorical = self.args.get("labels_to_categorical",
                                                    False)
        self._normalize = self.args.get("normalize", False)
        self._standardize = self.args.get("standardize", False)
        self._min_max = self.args.get("min_max", False)
        self._resize = self.args.get("resize", None)
        self._binary_data = self.args.get("binary_data", None)
        self._is_binary_data = self.args.get("is_binary_data", False)
        self._pca = self.args.get("pca", None)
        self._graph_conv = self.args.get("graph_conv", False)
        self._center_crop = self.args.get("center_crop", None)

        if self._is_binary_data:
            self._binary_data = None

    def standardize(self, x):
        """
        Standardize features by removing the mean and scaling to unit variance.
        """
        print("Standardizing data...")

        img_size = self.dims

        x = x.reshape(-1, np.prod(img_size))
        StandardScaler(copy=False).fit_transform(x)
        x = x.reshape([-1] + list(img_size))
        return x

    def normalize_data(self, x):
        """
        Scale input vectors individually to unit norm (vector length).
        """
        print("Normalizing data...")
        img_size = self.dims
        x = x.reshape(-1, np.prod(img_size))
        normalize(x, copy=False)
        x = x.reshape([-1] + list(img_size))
        return x

    def min_max_scale(self, x):
        print("Min-max scaling...")
        img_size = self.dims
        x = x.reshape(-1, np.prod(img_size))
        MinMaxScaler((-np.pi, np.pi), copy=False).fit_transform(x)
        x = x.reshape([-1] + list(img_size))
        return x

    def resize(self, x):
        """
        It resizes the training and testing data to the size specified in the constructor
        """
        print("Resizing data...")
        x = image.resize(x, self._resize).numpy()
        self.dims = x.shape[1:]
        return x

    def labels_to_categorical(self, y):
        """
        It converts the labels to categorical data
        """
        print("Converting labels to categorical...")

        y = to_categorical(y, num_classes=len(self.mapping))
        self.output_dims = (len(self.mapping),)
        return y

    def binary_data(self, x, y):
        """
        It takes the data and filters it so that only the data that contains binary classes
        """
        print("Binarizing data...")

        if self._is_binary_data is False:

            # Get the binary classes
            d1 = self._binary_data[0]
            d2 = self._binary_data[1]

            # Extract binary data
            x, y = binary_filter(d1, d2, x, y)
            self.mapping = [d1, d2]
            self.classes = [self.classes[d1], self.classes[d2]]
        return x, y

    def pca(self, x, n_components=16):
        """
        Performs Principal component analysis (PCA) on the data.

        Args:
          n_components: Number of components to keep. If n_components is not set all components are
        kept:. Defaults to 16
        """
        print("Performing PCA on data...")

        sq_root = int(np.sqrt(n_components))
        assert sq_root * sq_root == n_components, "Number of components must be a square"

        pca_obj = PCA(n_components)
        x = x.reshape(-1, np.prod(self.dims))

        pca_obj.fit(x)
        cumsum = np.cumsum(pca_obj.explained_variance_ratio_ * 100)[-1]
        print("Cumulative sum :", cumsum)
        x = pca_obj.transform(x)

        x = x.reshape(-1, sq_root, sq_root, 1)

        self.dims = (sq_root, sq_root, 1)
        return x

    def graph_convolution(self, x):
        print("Performing graph convolution...")
        m = self.dims[0]
        n = self.dims[1]
        N = m * n
        adj = np.zeros((N, N))
        sigma = np.pi

        # Create adjacency matrix
        @njit(parallel=True)
        def fill(adj, i):
            for j in prange(i, N):
                p1 = np.array([i // n, i % n])
                p2 = np.array([j // n, j % n])
                d = np.sqrt(np.sum(np.square(p1 - p2)))
                val = np.exp(-d / (sigma**2))
                adj[i][j] = val
                adj[j][i] = val

        def iterate(adj):
            for i in prange(N):
                fill(adj, i)

        iterate(adj)

        # Perfrom graph convolution
        x = x.reshape(-1, N, self.dims[2]).T
        x = np.dot(adj, x).T.reshape(-1, m, n, self.dims[2])
        return x

    def center_crop(self, x, fraction=0.2):
        print("Center cropping...")
        x = image.central_crop(x, fraction).numpy()
        self.dims = x.shape[1:]
        return x

    def process(self, x, y, config, classes):
        """
        Data processing pipeline.
        """

        self.dims = config['input_dims']
        self.output_dims = config['output_dims']
        self.mapping = config['mapping']
        self.classes = classes

        # Add new axis
        if len(x.shape) == 3:
            x = x[..., np.newaxis]  # For resizing we need to add one more axis

        if self._binary_data and len(self._binary_data) == 2:
            x, y = self.binary_data(x, y)

        if self._resize is not None and len(self._resize) == 2:
            x = self.resize(x)
        if self._pca is not None:
            x = self.pca(x, self._pca)
        if self._center_crop:
            x = self.center_crop(x, self._center_crop)

        if self._standardize:
            x = self.standardize(x)
        if self._normalize:
            x = self.normalize_data(x)
        if self._min_max:
            x = self.min_max_scale(x)

        if self._labels_to_categorical:
            y = self.labels_to_categorical(y)

        if self._graph_conv:
            x = self.graph_convolution(x)

        return x, y

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--labels-to-categorical",
                            "-to-cat",
                            action="store_true",
                            default=False)
        parser.add_argument("--normalize",
                            "-nz",
                            action="store_true",
                            default=False)
        parser.add_argument("--standardize",
                            "-std",
                            action="store_true",
                            default=False)
        parser.add_argument("--min-max",
                            "-mm",
                            action="store_true",
                            default=False)
        parser.add_argument("--resize", "-rz", action=ParseAction, default=None)
        parser.add_argument("--binary-data",
                            "-bd",
                            action=ParseAction,
                            default=None)
        parser.add_argument("--pca", "-pca", type=int, default=None)
        parser.add_argument("--graph-conv",
                            "-gc",
                            action="store_true",
                            default=False)
        parser.add_argument("--center-crop", "-cc", type=float, default=None)
        return parser


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
