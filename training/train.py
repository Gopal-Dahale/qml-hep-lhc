import importlib
import tensorflow as tf
import wandb
import argparse
import tensorflow_addons as tfa
from qml_hep_lhc.metrics import hinge_accuracy


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    parser = argparse.ArgumentParser()

    # Data parameters
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data-class", "-dc", type=str, default="MNIST")
    data_group.add_argument("--quantum", "-q", action="store_true")
    data_group.add_argument("--labels-to-categorical", "-to-cat",
                        action="store_true",
                        default=False)
    data_group.add_argument("--normalize", "-nz", action="store_true", default=False)
    data_group.add_argument("--resize", "-rz", nargs='+', type=int, default=None)
    data_group.add_argument("--binary-encoding", "-be",
                        action="store_true",
                        default=False)
    data_group.add_argument("--threshold", "-t", type=float, default=0.5)
    data_group.add_argument("--binary-data", "-bd", nargs='+', type=int, default=None)
    data_group.add_argument("--hinge-labels", "-hinge", action="store_true", default=False)
    data_group.add_argument("--batch-size", "-batch", type=int, default=128)
    data_group.add_argument("--percent-samples", "-per-samp", type=float, default=1.0)
    data_group.add_argument("--angle-encoding", "-ae", action="store_true", default=False)
    
    # Model parameters
    model_group = parser.add_argument_group("Model")    
    model_group.add_argument("--model-class", "-mc", type=str, default="ResnetV1")
    model_group.add_argument("--resnet-depth", "-rd" ,type=int, default=20)


    # Hyperparameters
    hyper_group = parser.add_argument_group("Hyperparameters")
    hyper_group.add_argument("--epochs", "-e" ,type=int, default=3)
    hyper_group.add_argument("--loss", "-l", type=str, default="CategoricalCrossentropy")
    hyper_group.add_argument("--optimizer", "-opt", type=str, default="Adam")
    hyper_group.add_argument("--accuracy", "-acc", type=str, default="accuracy")
    hyper_group.add_argument("--validation-split", "-val-split" ,type=float, default=0.2)
    hyper_group.add_argument("--num-workers", "-workers", type=int, default=2)
    hyper_group.add_argument("--learning-rate", "-lr" ,type=float, default=0.0001)
    hyper_group.add_argument("--wandb", action="store_true", default=False)

    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"qml_hep_lhc.data.{args.data_class}")

    data = data_class(args)
    data.prepare_data()
    data.setup()
    print(repr(data))

    if args.quantum:
        model_class = _import_class(f"qml_hep_lhc.models.{args.model_class}")
        model = model_class(data.q_data_config(), args)
        x_train, y_train = data.qx_train, data.y_train
        x_test, y_test = data.qx_test, data.y_test
    else:
        model_class = _import_class(f"qml_hep_lhc.models.{args.model_class}")
        model = model_class(data.config(), args)
        x_train, y_train = data.x_train, data.y_train
        x_test, y_test = data.x_test, data.y_test

    

    callbacks = []
    if args.wandb:
        wandb.init(project='mnist',
                   config={
                       'dataset': "mnist-dataset",
                       'lr': 0.001,
                   })
        callbacks.append(wandb.keras.WandbCallback())

    loss = args.loss
    loss_fn = getattr(tf.keras.losses, loss)
    optimizer = getattr(tf.keras.optimizers, args.optimizer)
    accuracy = args.accuracy
    batch_size = args.batch_size
    epochs = args.epochs
    validation_split = args.validation_split
    num_workers = args.num_workers
    INIT_LR = args.learning_rate
    MAX_LR = 1e-2

    steps_per_epoch = batch_size
    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,
                                              maximal_learning_rate=MAX_LR,
                                              scale_fn=lambda x: 1 /
                                              (2.**(x - 1)),
                                              step_size=2 * steps_per_epoch)

    if args.hinge_labels:
        accuracy = hinge_accuracy


    print(model.build_graph().summary())
    model.compile(loss=loss_fn(),
                  metrics=['accuracy'],
                  optimizer=optimizer(clr))

    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              validation_split=validation_split,
              shuffle=True,
              workers=num_workers)

    model.evaluate(x_test, y_test, callbacks=callbacks, workers=num_workers)


if __name__ == "__main__":
    main()
