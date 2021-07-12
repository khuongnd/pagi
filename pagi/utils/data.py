
import tensorflow as tf
from tensorflow.keras import datasets


@tf.function
def tf_normalize_features_and_labels(x, y):
    '''
    Normalize samples
    :param x: tensor of samples
    :param y: labels
    :return: (x:0...1, y: 0 or 1)
    '''
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


def tf_get_mnist_dataset(bath_size=100):
    '''
    Get tensorflow mnist dataset
    '''
    (x, y), _ = datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(tf_normalize_features_and_labels)
    ds = ds.batch(bath_size)
    return ds
