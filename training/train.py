from gc import callbacks
from qml_hep_lhc.data.mnist import Mnist
from qml_hep_lhc.models.mlp import Mlp
import tensorflow as tf
import wandb

def main():
    data = Mnist()
    model = Mlp(10)
    
    wandb.init(project='mnist', config={
        'dataset':"mnist-dataset",
        'lr':0.001,
    })
    model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = tf.keras.metrics.CategoricalAccuracy(),
              optimizer = tf.keras.optimizers.Adam())
    model.fit(data.x_train, data.y_train,
         batch_size=128, epochs=3, callbacks=[wandb.keras.WandbCallback()]);

if __name__ == "__main__":
    main()