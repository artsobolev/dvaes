import tensorflow as tf


def summary_mean_and_std(name, mean, std):
    with tf.name_scope(name):
        return [tf.summary.scalar('mean', mean),
                tf.summary.scalar('std', std)]


class TransformedSampler(tf.distributions.Distribution):
    def __init__(self, base_distribution, transformation, reparameterized=True):
        if reparameterized:
            reparameterization_type = tf.distributions.FULLY_REPARAMETERIZED
        else:
            reparameterization_type = tf.distributions.NOT_REPARAMETERIZED

        super(TransformedSampler, self).__init__(dtype=tf.float32, reparameterization_type=reparameterization_type,
                                                 validate_args=False, allow_nan_stats=False)

        self._base_distribution = base_distribution
        self._transformation = transformation
        self._reparameterized = reparameterized

    def sample(self, *args, **kwargs):
        z = self._base_distribution.sample(*args, **kwargs)
        return self._transformation(z)
