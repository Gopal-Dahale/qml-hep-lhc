import optax
from qml_hep_lhc.models.quantum.jax.metrics import pauli_z_accuracy, custom_accuracy


class BaseModel:

    def __init__(self, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.lr = self.args.get('learning_rate', 0.002)

        # Optimizer
        optimizer = optax.adamw(learning_rate=self.lr)

        self.batch_size = self.args.get('batch_size', 128)

        if self.args.get('use_quantum', False):
            self.loss = "MeanSquaredError"
            self.accuracy = pauli_z_accuracy
        else:
            self.loss = "CategoricalCrossentropy"
            self.accuracy = custom_accuracy

    def compile(self):
        super(BaseModel, self).compile(loss=self.loss_fn,
                                       metrics=self.accuracy,
                                       optimizer=self.optimizer,
                                       run_eagerly=True)

    def fit(self, data, callbacks):
        return super(BaseModel,
                     self).fit(data.train_ds,
                               batch_size=self.batch_size,
                               epochs=self.args.get('epochs', 3),
                               callbacks=callbacks,
                               validation_data=data.val_ds,
                               shuffle=True,
                               workers=self.args.get('num_workers', 4))

    def test(self, data, callbacks):
        return super(BaseModel,
                     self).evaluate(data.test_ds,
                                    callbacks=callbacks,
                                    batch_size=self.batch_size,
                                    workers=self.args.get('num_workers', 4))

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--learning-rate", "-lr", type=float, default=0.01)
        parser.add_argument("--use-quantum",
                            "-q",
                            action="store_true",
                            default=False)
        return parser
