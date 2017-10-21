import tensorflow as tf


def summary_mean_and_std(name, mean, std):
    with tf.name_scope(name):
        return [tf.summary.scalar('mean', mean),
                tf.summary.scalar('std', std)]


class TransformedSampler:
    def __init__(self, base_distribution, transformation):
        self._base_distribution = base_distribution
        self._transformation = transformation

    def sample(self, *args, **kwargs):
        z = self._base_distribution.sample(*args, **kwargs)
        return self._transformation(z)
