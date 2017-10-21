import tensorflow as tf

from abstract_dvae import AbstractDVAE


class ConcretelyRelaxedDVAE(AbstractDVAE):
    def __init__(self, *args, **kwargs):
        self.tau_ = kwargs.get('tau', 1.0)

        AbstractDVAE.__init__(self, *args, **kwargs)
    
    def _build_relaxed_encoder(self, logits):
        with tf.name_scope('encoder'):
            logistic = tf.contrib.distributions.Logistic(loc=logits, scale=1.)
            transformation = tf.contrib.distributions.bijectors.Chain([
                tf.contrib.distributions.bijectors.Affine(scale_identity_multiplier=1.0 / self.tau_),
                tf.contrib.distributions.bijectors.Sigmoid()
            ])
            return tf.contrib.distributions.TransformedDistribution(logistic, bijector=transformation)


class GeneralizedRelaxedDVAE(AbstractDVAE):
    def __init__(self, distribution_factory, tau=1.0, *args, **kwargs):
        AbstractDVAE.__init__(self, *args, **kwargs)
        self.tau_ = tau
        self.distribution_factory_ = distribution_factory

    def _build_relaxed_encoder(self, logits):
        with tf.name_scope('encoder'):
            distribution = self.distribution_factory_(logits.get_shape())
            proba_c = tf.sigmoid(-logits)

            # This implements sigmoid(X - inv_cdf(1 - proba))
            transformation = tf.contrib.distributions.bijectors.Chain([
                tf.contrib.distributions.bijectors.Affine(shift=-distribution.quantile(proba_c)),
                tf.contrib.distributions.bijectors.Affine(scale_identity_multiplier=1.0 / self.tau_),
                tf.contrib.distributions.bijectors.Sigmoid()
            ])
            return tf.contrib.distributions.TransformedDistribution(distribution, bijector=transformation)
