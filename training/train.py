from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.train import latest_checkpoint
import wandb
from argparse import ArgumentParser
from os import path, makedirs
from qml_hep_lhc.utils import _import_class
from tensorflow.keras.callbacks import ReduceLROnPlateau


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

    data_class = _import_class(f"qml_hep_lhc.data.{temp_args.data_class}")
    base_model_class = _import_class(f"qml_hep_lhc.models.BaseModel")
    model_class = _import_class(f"qml_hep_lhc.models.{temp_args.model_class}")
    dp_class = _import_class(f"qml_hep_lhc.data.DataPreprocessor")

    # Get data, model, and LitModel specific arguments
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


def _setup_callbacks(args):
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
        wandb.init(project='mnist',
                   config={
                       'dataset': "mnist-dataset",
                       'lr': 0.001,
                   })
        callbacks.append(wandb.keras.WandbCallback())
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
                                              min_lr=1e-6)
    # callbacks.append(lr_scheduler_callback)
    return callbacks


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

    callbacks = _setup_callbacks(args)
    print(model.build_graph().summary())  # Print the Model summary

    # Training the model
    model.compile()
    model.fit(data, callbacks=callbacks)

    # # Testing the model
    # model.test(data, callbacks=callbacks)


if __name__ == "__main__":
    main()
