import tensorflow as tf

from abstract_dvae import AbstractDVAE


class ConcretelyRelaxedDVAE(AbstractDVAE):
    def __init__(self, *args, **kwargs):
        self.tau = kwargs.get('tau', 1.0)

        AbstractDVAE.__init__(self, *args, **kwargs)
    
    def _build_relaxed_encoder(self, logits):
        with tf.name_scope('encoder'):
            logistic = tf.contrib.distributions.Logistic(loc=logits / self.tau, scale=1. / self.tau)
            transformation = tf.contrib.distributions.bijectors.Sigmoid()
            return tf.contrib.distributions.TransformedDistribution(logistic, bijector=transformation)


class GeneralizedRelaxedDVAE(AbstractDVAE):
    FACTORIES = {
        'Uniform': lambda shape: tf.distributions.Uniform(low=tf.zeros(shape), high=tf.ones(shape)),
        'Laplace': lambda shape: tf.distributions.Laplace(loc=tf.zeros(shape), scale=tf.ones(shape)),
        'Normal': lambda shape: tf.distributions.Normal(loc=tf.zeros(shape), scale=tf.ones(shape)),
    }

    def __init__(self, relaxation_distribution, *args, **kwargs):
        self.tau = kwargs.get('tau', 1.0)
        self.distribution_factory_ = self.FACTORIES[relaxation_distribution]

        AbstractDVAE.__init__(self, *args, **kwargs)

    def _build_relaxed_encoder(self, logits):
        with tf.name_scope('encoder'):
            distribution = self.distribution_factory_(tf.shape(logits))
            proba_c = tf.sigmoid(-logits)

            # This implements sigmoid(X - inv_cdf(1 - proba))
            transformation = tf.contrib.distributions.bijectors.Chain([
                tf.contrib.distributions.bijectors.Sigmoid(),
                tf.contrib.distributions.bijectors.Affine(scale_identity_multiplier=1.0 / self.tau),
                tf.contrib.distributions.bijectors.Affine(shift=-distribution.quantile(proba_c)),
            ])
            return tf.contrib.distributions.TransformedDistribution(distribution, bijector=transformation)
