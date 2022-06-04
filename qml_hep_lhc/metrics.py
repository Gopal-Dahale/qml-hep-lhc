import tensorflow as tf


def hinge_accuracy(y_true, y_pred):
    """
    It returns the fraction of the time that the predicted label is the same as the true label
    
    Args:
      y_true: The true labels.
      y_pred: The predicted values.
    
    Returns:
      The mean of the result of the comparison of the two tensors.
    """
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)
