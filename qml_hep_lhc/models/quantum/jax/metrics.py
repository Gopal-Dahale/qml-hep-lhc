import jax.numpy as jnp


def custom_accuracy(y_true, y_pred):
    target_class = jnp.argmax(y_true, axis=1)
    predicted_class = jnp.argmax(y_pred, axis=1)
    return jnp.sum(predicted_class == target_class)


def pauli_z_accuracy(y_true, y_pred):
    y_true = jnp.squeeze(y_true)
    y_pred = (y_pred >= 0)
    y_pred = 2 * y_pred - 1
    return jnp.sum(y_true == y_pred)