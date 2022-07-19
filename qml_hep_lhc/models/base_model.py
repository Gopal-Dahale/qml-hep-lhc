from tensorflow.keras import Model, losses, optimizers
from tensorflow.keras.metrics import AUC
from qml_hep_lhc.utils import _import_class
from qml_hep_lhc.models.quantum.metrics import qAUC, custom_accuracy
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead


class BaseModel(Model):

    def __init__(self, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        # Loss function
        self.loss = self.args.get('loss', "CategoricalCrossentropy")
        self.loss_fn = getattr(losses, self.loss)()

        self.lr = self.args.get('learning_rate', 0.002)

        # Optimizer
        if self.args.get('optimizer', 'Adam') == 'Adam':
            self.optimizer = getattr(optimizers, 'Adam')(learning_rate=self.lr)
        elif self.args.get('optimizer', 'Adam') == 'Ranger':
            radam = RectifiedAdam(learning_rate=self.lr)
            self.optimizer = Lookahead(radam, sync_period=6, slow_step_size=0.5)

        # Learning rate scheduler
        self.batch_size = self.args.get('batch_size', 128)

        # Accuracy
        self.accuracy = [AUC(), 'accuracy']
        self.acc_metrics = ['accuracy,', 'AUC']

        if self.args.get('use_quantum', False):
            self.loss = "MeanSquaredError"
            self.loss_fn = getattr(losses, self.loss)()

    def compile(self):
        super(BaseModel, self).compile(loss=self.loss_fn,
                                       metrics=self.accuracy,
                                       optimizer=self.optimizer)

    def fit(self, data, callbacks):
        x = data.x_train
        y = data.y_train

        return super(BaseModel,
                     self).fit(x=x,
                               y=y,
                               batch_size=self.batch_size,
                               epochs=self.args.get('epochs', 3),
                               callbacks=callbacks,
                               validation_split=self.args.get(
                                   'validation_split', 0.2),
                               shuffle=True,
                               workers=self.args.get('num_workers', 4))

    def test(self, data, callbacks):
        x = data.x_test
        y = data.y_test

        return super(BaseModel,
                     self).evaluate(x=x,
                                    y=y,
                                    callbacks=callbacks,
                                    batch_size=self.batch_size,
                                    workers=self.args.get('num_workers', 4))

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", "-opt", type=str, default="Adam")
        parser.add_argument("--learning-rate", "-lr", type=float, default=0.01)
        parser.add_argument("--loss",
                            "-l",
                            type=str,
                            default="CategoricalCrossentropy")
        parser.add_argument("--use-quantum",
                            "-q",
                            action="store_true",
                            default=False)
        return parser
