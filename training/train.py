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
    parser.add_argument("--labels-to-categorical", action="store_true", default=False)
    
    # argument for using wandb
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--resnet-depth", type=int, default=20)
    parser.add_argument("--resize", nargs='+',type=int,default=[28,28])
    parser.add_argument("--quantum", action="store_true", default=False)
    parser.add_argument("--binary-encoding", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--binary-data", nargs='+', type=int, default=None)
    parser.add_argument("--hinge-labels", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-class", type=str, default="MNIST")
    parser.add_argument("--model-class", type=str, default="ResnetV1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--loss", type=str, default="CategoricalCrossentropy")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--accuracy", type=str, default="accuracy")
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--percent-samples", type=float, default=1.0)
    parser.add_argument("--learning-rate",type=float, default=0.0001)
    return parser

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"qml_hep_lhc.data.{args.data_class}")
    

    data = data_class(args)
    data.prepare_data()
    data.setup()

    if args.quantum:
        model_class = _import_class(f"qml_hep_lhc.models.quantum.{args.model_class}")
        model = model_class(data.q_data_config(), args)
        x_train, y_train = data.qx_train, data.y_train
        x_test, y_test = data.qx_test, data.y_test
    else:
        model_class = _import_class(f"qml_hep_lhc.models.{args.model_class}")
        model = model_class(data.config())
        x_train, y_train = data.x_train, data.y_train
        x_test, y_test = data.x_test, data.y_test

    print(repr(data))

    callbacks = []
    if args.wandb:
        wandb.init(project='mnist', config={
            'dataset':"mnist-dataset",
            'lr':0.001,
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
        maximal_learning_rate = MAX_LR,
        scale_fn=lambda x: 1/(2.**(x-1)),
        step_size=2 * steps_per_epoch
    )

    if args.hinge_labels:
        accuracy = hinge_accuracy

    model.compile(loss = loss_fn(),
              metrics = ['accuracy'],
              optimizer = optimizer(clr))
            
    model.fit(x_train,y_train,
        batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_split=validation_split, shuffle=True, workers=num_workers)

    model.evaluate(x_test, y_test, callbacks=callbacks, workers= num_workers)
    
if __name__ == "__main__":
    main()