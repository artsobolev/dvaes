import tensorflow as tf

import model_utils
from abstract_dvae import AbstractReparametrizedDVAE


class AbstractBackwardRelaxedDVAE(AbstractReparametrizedDVAE):
    def _backward(self, u, logits):
        raise NotImplementedError()

    def _forward(self, u, logits):
        return tf.to_float(tf.less(u, tf.sigmoid(logits)))

    def _build_relaxed_encoder(self, logits):
        with tf.name_scope('encoder'):
            uniform = tf.distributions.Uniform(high=tf.ones_like(logits))

        def transform(u):
            # forward pass: samples
            # backward pass: logits
            forward = self._forward(u, logits)
            backward = self._backward(u, logits)
            return backward + tf.stop_gradient(forward - backward)

        return model_utils.TransformedSampler(uniform, transform)


class StraightThroughDVAE(AbstractBackwardRelaxedDVAE):
    def _backward(self, u, logits):
        return logits


class BackwardMeanRelaxedDVAE(AbstractBackwardRelaxedDVAE):
    def _backward(self, u, logits):
        return tf.sigmoid(logits)


class BackwardGumbelRelaxedDVAE(AbstractBackwardRelaxedDVAE):
    def __init__(self, *args, **kwargs):
        self.tau = kwargs.get('tau')
        AbstractBackwardRelaxedDVAE.__init__(self, *args, **kwargs)

    def _backward(self, u, logits):
        logistic = tf.log(u) - tf.log1p(-u)
        return tf.sigmoid((logistic + logits) / self.tau)
