import tensorflow as tf

from abstract_dvae import AbstractDVAE


class AbstractBackwardRelaxedDVAE(AbstractDVAE):
    def __init__(self, *args, **kwargs):
        AbstractDVAE.__init__(self, *args, **kwargs)

    def _backward(self, logits, *args, **kwargs):
        raise NotImplementedError()
    
    def _build_relaxed_encoder(self, logits):
        backward_factory = self._backward
        with tf.name_scope('encoder'):
            distribution = tf.distributions.Bernoulli(logits=logits)

        class _Implementation:
            def sample(self, *args, **kwargs):
                # forward pass: sigmoid(logits)
                # backward pass: logits

                forward = tf.to_float(distribution.sample(*args, **kwargs))
                backward = backward_factory(logits, *args, **kwargs)
                return backward + tf.stop_gradient(forward - backward)

        return _Implementation()


class StraightThroughDVAE(AbstractBackwardRelaxedDVAE):
    def __init__(self, *args, **kwargs):
        AbstractBackwardRelaxedDVAE.__init__(self, *args, **kwargs)

    def _backward(self, logits, *args, **kwargs):
        return logits


class BackwardMeanRelaxedDVAE(AbstractBackwardRelaxedDVAE):
    def __init__(self, *args, **kwargs):
        AbstractBackwardRelaxedDVAE.__init__(self, *args, **kwargs)

    def _backward(self, logits, *args, **kwargs):
        return tf.sigmoid(logits)


class BackwardGumbelRelaxedDVAE(AbstractBackwardRelaxedDVAE):
    def __init__(self, *args, **kwargs):
        self.tau = kwargs.get('tau')
        AbstractBackwardRelaxedDVAE.__init__(self, *args, **kwargs)

    def _backward(self, logits, *args, **kwargs):
        logistic = tf.contrib.distributions.Logistic(loc=logits / self.tau, scale=1. / self.tau)
        transformation = tf.contrib.distributions.bijectors.Sigmoid()
        distribution = tf.contrib.distributions.TransformedDistribution(logistic, bijector=transformation)
        return distribution.sample(*args, **kwargs)