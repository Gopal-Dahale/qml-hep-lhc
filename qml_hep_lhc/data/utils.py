import numpy as np
import tensorflow as tf


def extract_samples(x, y, mapping, percent):
    samples_per_class = int((len(y) / len(mapping)) * percent)
    keep = []
    for i in mapping:
        keep += list(np.where(y == i)[0][:samples_per_class])
    x, y = x[keep], y[keep]
    return x, y


def create_tf_ds(x, y, batch_size):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(100)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds
