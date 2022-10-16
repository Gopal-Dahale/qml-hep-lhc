from tensorflow import train
from argparse import ArgumentParser
from qml_hep_lhc.utils import _import_class
from callbacks import _setup_callbacks
import wandb


def _setup_parser():
    """
    It creates a parser object, and then adds arguments to it

    Returns:
      A parser object
    """
    parser = ArgumentParser()

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--data-class", "-dc", type=str, default="MNIST")
    parser.add_argument("--model-class", "-mc", type=str, default="FQCNN")

    temp_args, _ = parser.parse_known_args()

    base_data_class = _import_class(f"qml_hep_lhc.data.BaseDataModule")
    data_class = _import_class(f"qml_hep_lhc.data.{temp_args.data_class}")
    base_model_class = _import_class(f"qml_hep_lhc.models.JBaseModel")
    model_class = _import_class(f"qml_hep_lhc.models.J{temp_args.model_class}")
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
    parser.add_argument("--num-workers", "-workers", type=int, default=4)
    return parser


def get_configuration(parser, args, data, model):
    arg_grps = {}
    for group in parser._action_groups:
        group_dict = {
            a.dest: getattr(args, a.dest, None) for a in group._group_actions
        }
        arg_grps[group.title] = group_dict

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
    model_class = _import_class(f"qml_hep_lhc.models.J{args.model_class}")
    model = model_class(data.config(), args)  # Model

    config = get_configuration(parser, args, data, model)

    if args.wandb:
        wandb.init(project='qml-hep-lhc', config=config)

    print(model.summary())  # Print the Model summary

    # Training the model
    model.compile()
    model.fit(data)

    # Testing the model
    model.test(data)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
