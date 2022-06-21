from tensorflow.keras import Model, losses, optimizers
from tensorflow_addons.optimizers import CyclicalLearningRate
from tensorflow.keras.metrics import AUC
from qml_hep_lhc.utils import _import_class
from qml_hep_lhc.models.quantum.metrics import qAUC, custom_accuracy


class BaseModel(Model):

    def __init__(self, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        # Loss function
        loss = self.args.get('loss', "CategoricalCrossentropy")
        self.loss_fn = getattr(losses, loss)

        # Optimizer
        self.optimizer = getattr(optimizers, self.args.get('optimizer', 'Adam'))

        self.lr = self.args.get('learning_rate', 0.001)
        self.max_lr = 1e-2

        # Learning rate scheduler
        self.batch_size = self.args.get('batch_size', 128)
        steps_per_epoch = self.batch_size
        self.clr = CyclicalLearningRate(initial_learning_rate=self.lr,
                                        maximal_learning_rate=self.max_lr,
                                        scale_fn=lambda x: 1 / (2.**(x - 1)),
                                        step_size=2 * steps_per_epoch)

        if self.args.get('use_quantum', False):
            print("use quantum")
            self.loss_fn = getattr(losses, "MeanSquaredError")
            self.accuracy = [custom_accuracy, qAUC()]
        else:
            print("dont use quantum")
            self.accuracy = ['accuracy', AUC()]

    def compile(self):
        super(BaseModel, self).compile(loss=self.loss_fn(),
                                       metrics=self.accuracy,
                                       optimizer=self.optimizer(self.clr))

    def fit(self, data, callbacks):
        x = data.x_train
        y = data.y_train

        return super(BaseModel, self).fit(
            x=x,
            y=y,
            batch_size=self.batch_size,
            epochs=self.args.get('epochs'),
            callbacks=callbacks,
            validation_split=self.args.get('validation_split'),
            shuffle=True,
            workers=self.args.get('num_workers'))

    def test(self, data, callbacks):
        x = data.x_test
        y = data.y_test

        return super(BaseModel,
                     self).evaluate(x=x,
                                    y=y,
                                    callbacks=callbacks,
                                    batch_size=self.batch_size,
                                    workers=self.args.get('num_workers'))

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", "-opt", type=str, default="Adam")
        parser.add_argument("--learning-rate",
                            "-lr",
                            type=float,
                            default=0.0001)
        parser.add_argument("--loss",
                            "-l",
                            type=str,
                            default="CategoricalCrossentropy")
        parser.add_argument("--use-quantum",
                            "-q",
                            action="store_true",
                            default=False)
        parser.add_argument("--feature-map", "-fm", type=str)
        return parser
