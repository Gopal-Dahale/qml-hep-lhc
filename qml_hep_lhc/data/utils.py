import numpy as np


def extract_samples(x, y, mapping, percent):
    samples_per_class = int((len(y) / len(mapping)) * percent)
    keep = []
    for i in mapping:
        keep += list(np.where(y == i)[0][:samples_per_class])
    x, y = x[keep], y[keep]
    return x, y
