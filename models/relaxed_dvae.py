import tensorflow as tf

import model_utils
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


class _UniformWithQuantile(tf.distributions.Uniform):
    def __init__(self, *args, **kwargs):
        tf.distributions.Uniform.__init__(self, *args, **kwargs)

    def _quantile(self, p):
        return p * self.range() + self.low


class GeneralizedRelaxedDVAE(AbstractDVAE):
    DISTRIBUTION_FACTORIES = {
        'Uniform': lambda shape: _UniformWithQuantile(low=tf.zeros(shape), high=tf.ones(shape)),
        'Normal': lambda shape: tf.distributions.Normal(loc=tf.zeros(shape), scale=tf.ones(shape)),
    }

    def __init__(self, relaxation_distribution, *args, **kwargs):
        self.tau = kwargs.get('tau', 1.0)
        self.distribution_factory_ = self.DISTRIBUTION_FACTORIES[relaxation_distribution]

        AbstractDVAE.__init__(self, *args, **kwargs)

    def _build_relaxed_encoder(self, logits):
        with tf.name_scope('encoder'):
            distribution = self.distribution_factory_(tf.shape(logits))
            proba_c = tf.sigmoid(-logits)

            # This implements sigmoid(X - inv_cdf(1 - proba))
            transform = tf.contrib.distributions.bijectors.Chain([
                tf.contrib.distributions.bijectors.Sigmoid(),
                tf.contrib.distributions.bijectors.Affine(scale_identity_multiplier=1.0 / self.tau),
                tf.contrib.distributions.bijectors.Affine(shift=-distribution.quantile(proba_c)),
            ])
            return tf.contrib.distributions.TransformedDistribution(distribution, bijector=transform)


class _TruncatedExponential:
    def __init__(self, beta):
        self.beta = beta

    def quantile(self, rho):
        return tf.log1p(rho * tf.expm1(self.beta)) / self.beta


class NoiseRelaxedDVAE(AbstractDVAE):
    NOISE_FACTORIES = {
        'Normal': lambda shape, tau: tf.distributions.Normal(loc=tf.ones(shape), scale=tau * tf.ones(shape)),
        'TruncatedExponential': lambda shape, tau: _TruncatedExponential(beta=tf.ones(shape) / tau),
    }

    def __init__(self, noise_distribution, *args, **kwargs):
        self.tau = kwargs.get('tau', 1.0)
        self.noise_factory_ = self.NOISE_FACTORIES[noise_distribution]

        AbstractDVAE.__init__(self, *args, **kwargs)

    def _build_relaxed_encoder(self, logits):
        with tf.name_scope('encoder'):
            logits = tf.clip_by_value(logits, -5, 5)
            noise_distribution = self.noise_factory_(tf.shape(logits), self.tau)
            uniform = tf.distributions.Uniform(low=tf.zeros_like(logits), high=tf.ones_like(logits))
            proba_inv = 1. + tf.exp(-logits)
            proba_c = tf.sigmoid(-logits)

        def transform(rho):
            discrete_case = tf.zeros_like(rho)
            continuous_case = noise_distribution.quantile(tf.clip_by_value((rho - proba_c) * proba_inv, 1e-5, 1-1e-5))

            return tf.where(rho < proba_c, discrete_case, continuous_case)

        return model_utils.TransformedSampler(uniform, transform)
