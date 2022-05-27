from gc import callbacks
from xmlrpc.client import boolean
from qml_hep_lhc.data.mnist import Mnist
from qml_hep_lhc.models.resnet.v2 import ResnetV2
import tensorflow as tf
import wandb
import argparse


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_to_categorical", action="store_true", default=False)
    
    # argument for using wandb
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--resnet-depth", type=int, default=20)
    parser.add_argument("--resize", nargs='+',type=int,default=[28,28])
    return parser

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    print(args.resize)
    print(type(args.resize))
    data = Mnist(args)
    data.prepare_data()
    data.setup()
    data_config = data.config()
    model = ResnetV2(data_config, args)
    print(repr(data))

    callbacks = []
    if args.wandb:
        wandb.init(project='mnist', config={
            'dataset':"mnist-dataset",
            'lr':0.001,
        })
        callbacks.append(wandb.keras.WandbCallback())

    model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = tf.keras.metrics.CategoricalAccuracy(),
              optimizer = tf.keras.optimizers.Adam())
    model.fit(data.x_train, data.y_train,
         batch_size=128, epochs=3, callbacks=callbacks);

if __name__ == "__main__":
    main()