from qml_hep_lhc.models.quantum.jax.base_model import BaseModel as JaxBaseModel
from qml_hep_lhc.utils import ParseAction


class QCNN(JaxBaseModel):
    """
	General Quantum Convolutional Neural Network
	"""

    def __init__(self, data_config, args=None):
        super(QCNN, self).__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]
        self.n_layers = self.args.get("n_layers", 1)
        self.n_qubits = self.args.get("n_qubits", 1)
        self.sparse = self.args.get("sparse", False)
        self.num_classes = len(data_config["mapping"])

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--n-layers", type=int, default=1)
        parser.add_argument("--n-qubits", type=int, default=1)
        parser.add_argument("--sparse", action="store_true", default=False)
        parser.add_argument('--num-fc-layers', type=int, default=1)
        parser.add_argument('--fc-dims', action=ParseAction, default=[8])
        parser.add_argument('--num-qconv-layers', type=int, default=1)
        parser.add_argument('--qconv-dims', action=ParseAction, default=[1])
        return parser
