from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.train import latest_checkpoint
import wandb
from argparse import ArgumentParser, Namespace
from os import path, makedirs
from qml_hep_lhc.utils import _import_class
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow import concat
from tensorflow import map_fn
from cirq.contrib.svg import SVGCircuit
from cairosvg import svg2png


def _setup_parser():
    """
    It creates a parser object, and then adds arguments to it

    Returns:
      A parser object
    """
    parser = ArgumentParser()

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--data-class", "-dc", type=str, default="MNIST")
    parser.add_argument("--model-class", "-mc", type=str, default="ResnetV1")
    parser.add_argument("--load-checkpoint", "-lc", type=str, default=None)
    parser.add_argument("--load-latest-checkpoint",
                        "-llc",
                        action="store_true",
                        default=False)

    temp_args, _ = parser.parse_known_args()

    base_data_class = _import_class(f"qml_hep_lhc.data.BaseDataModule")
    data_class = _import_class(f"qml_hep_lhc.data.{temp_args.data_class}")
    base_model_class = _import_class(f"qml_hep_lhc.models.BaseModel")
    model_class = _import_class(f"qml_hep_lhc.models.{temp_args.model_class}")
    dp_class = _import_class(f"qml_hep_lhc.data.DataPreprocessor")

    # Get data, model, and LitModel specific arguments
    base_data_group = parser.add_argument_group("Base Data Args")
    base_data_class.add_to_argparse(base_data_group)

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    base_model_group = parser.add_argument_group("Base Model Args")
    base_model_class.add_to_argparse(base_model_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    dp_group = parser.add_argument_group("Data Preprocessing Args")
    dp_class.add_to_argparse(dp_group)

    # model.fit specific arguments
    parser.add_argument("--epochs", "-e", type=int, default=3)
    parser.add_argument("--validation-split",
                        "-val-split",
                        type=float,
                        default=0.2)
    parser.add_argument("--num-workers", "-workers", type=int, default=4)
    return parser


def _setup_callbacks(args, config, data):
    """
    This function initializes and returns a list of callbacks

    Args:
      args: This is the namespace object that contains all the parameters that we passed in from the
    command line.

    Returns:
      A list of callbacks.
    """

    callbacks = []

    # Wandb callback
    if args.wandb:
        wandb.init(project='qml-hep-lhc', config=config)
        # wandb.run.name = f"{args.data_class}-{args.model_class}"
        callbacks.append(wandb.keras.WandbCallback(save_weights_only=True))

        # ROC Plot callback for wandb
        class PRMetrics(Callback):

            def __init__(self, data, use_quantum):
                self.x = data.x_test
                self.y = data.y_test
                self.use_quantum = use_quantum
                self.classes = data.classes

            def on_train_end(self, logs=None):
                out = self.model.predict(self.x)
                if self.use_quantum:
                    preds = map_fn(lambda x: 1.0 if x >= 0.5 else 0, out)
                    probs = out
                    probs = concat((probs, 1 - probs), axis=1)
                else:
                    self.y = self.y.argmax(axis=1)
                    preds = out.argmax(axis=1)
                    probs = out

                roc_curve = wandb.sklearn.plot_roc(self.y, probs, self.classes)
                confusion_matrix = wandb.sklearn.plot_confusion_matrix(
                    self.y, preds, self.classes)

                wandb.log({"roc_curve": roc_curve})
                wandb.log({"confusion_matrix": confusion_matrix})

                method = 'get_circuit'
                model_has_circuit = hasattr(self.model, method) and callable(
                    getattr(self.model, method))

                if model_has_circuit:
                    circuit = self.model.get_circuit()
                    image = SVGCircuit(circuit)._repr_svg_()
                    svg2png(image, write_to='circuit.png')

                    circuit_log = wandb.Image('circuit.png',
                                              caption=f"Quantum Circuit")

                    wandb.log({"circuit": circuit_log})

        callbacks.append(PRMetrics(data, args.use_quantum))

    checkpoint_path = './checkpoints/'
    checkpoint_dir = path.dirname(checkpoint_path)
    if not path.exists(checkpoint_dir):
        makedirs(checkpoint_dir)

    # Create a callback that saves the model's weights
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path + f"{args.data_class}-{args.model_class}-" +
        "{epoch:03d}-{val_loss:.3f}.ckpt",
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_weights_only=True,
        save_freq='epoch')

    callbacks.append(model_checkpoint_callback)

    # LR Scheduler callback
    lr_scheduler_callback = ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.1,
                                              patience=5,
                                              min_delta=0.0001,
                                              min_lr=1e-6)

    callbacks.append(lr_scheduler_callback)

    # Early Stopping Callback
    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            mode="min",
                                            patience=5)

    callbacks.append(early_stopping_callback)
    return callbacks


def get_configuration(parser, args, data, model):
    arg_grps = {}
    for group in parser._action_groups:
        group_dict = {
            a.dest: getattr(args, a.dest, None) for a in group._group_actions
        }
        arg_grps[group.title] = group_dict

    # Add additional configurations
    arg_grps['Base Model Args']['loss'] = model.loss
    arg_grps['Base Model Args']['accuracy'] = model.acc_metrics
    arg_grps['Base Model Args']['scheduler'] = 'ReduceLROnPlateau'

    # Additional configurations for quantum model
    if arg_grps['Base Model Args']['use_quantum']:
        arg_grps['Base Model Args']['feature_map'] = model.fm_class
    return arg_grps


def main():

    # Parsing the arguments from the command line.
    parser = _setup_parser()
    args = parser.parse_args()

    # Importing the data class
    data_class = _import_class(f"qml_hep_lhc.data.{args.data_class}")

    # Creating a data object, and then calling the prepare_data and setup methods on it.
    data = data_class(args)
    data.prepare_data()
    data.setup()

    print(data)

    # Importing the model class
    model_class = _import_class(f"qml_hep_lhc.models.{args.model_class}")
    model = model_class(data.config(), args)  # Model

    if args.load_latest_checkpoint:
        latest = latest_checkpoint(path.dirname('./checkpoints/'))
        print(latest)
        model.load_weights(latest)

    elif args.load_checkpoint is not None:
        model.load_weights(args.load_checkpoint)

    config = get_configuration(parser, args, data, model)
    callbacks = _setup_callbacks(args, config, data)

    print(model.build_graph().summary())  # Print the Model summary

    # Training the model
    model.compile()
    model.fit(data, callbacks=callbacks)

    # Testing the model
    model.test(data, callbacks=callbacks)


if __name__ == "__main__":
    main()
