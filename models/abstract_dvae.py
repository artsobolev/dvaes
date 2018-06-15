import abc
import itertools

import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.special

import model_utils


def log_sigmoid(x):
    return -tf.nn.softplus(-x)


# Implements log(1 - exp(-x))
def log1mexp(x):
    return tf.where(x < tf.log(2.), tf.log(-tf.expm1(-x)), tf.log1p(-tf.exp(-x)))


class AbstractDVAE:
    __metaclass__ = abc.ABCMeta

    def __init__(self, code_size, input_size, prior_p, lam, output_bias, jackknife_depths=(), jackknife_samples=(),
                 *args, **kwargs):
        self.code_size = code_size
        self.input_size = input_size
        self.prior_p = prior_p
        self.lam = lam
        self.output_bias = output_bias

        self._layer_params = dict(kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.input_ = tf.placeholder(tf.float32, shape=(None, self.input_size), name='Input')

        self.discrete_code_, self.discrete_reconstruction_, \
        self.discrete_elbo_ = self._build(self.input_, tf.contrib.distributions.Bernoulli,
                                          reuse=tf.AUTO_REUSE, scope_name='discrete')

        self.loss_ = self._build_loss(self.input_)

        self.multisample_elbos_ = {(k, d): self._build_jackknife_elbo(self.input_, k, d, reuse=True)
                                   for k, d in zip(jackknife_samples, jackknife_depths)}

        if (1, 0) in self.multisample_elbos_ and len(self.multisample_elbos_) > 1:
            max_k = max(self.multisample_elbos_.keys(), key=lambda x: x[0])
            self.posterior_approximation_kl_ = self.multisample_elbos_[max_k] - self.multisample_elbos_[1, 0]

        self.summaries_op_ = tf.summary.merge_all()

    def _build_encoder_logits(self, x, reuse):
        net = self._to_signed_binary(x)
        with tf.variable_scope('encoder', reuse=reuse):
            net = tf.layers.dense(net, 200, activation=tf.nn.tanh, **self._layer_params)
            net = tf.layers.dense(net, 200, activation=tf.nn.tanh, **self._layer_params)
            net = tf.layers.dense(net, self.code_size, activation=None, **self._layer_params)

        return net

    def _build_decoder_logits(self, code, reuse):
        net = self._to_signed_binary(code)
        with tf.variable_scope('decoder', reuse=reuse):
            net = tf.layers.dense(net, 200, activation=tf.nn.tanh, **self._layer_params)
            net = tf.layers.dense(net, 200, activation=tf.nn.tanh, **self._layer_params)
            net = tf.layers.dense(net, self.input_size, activation=None,
                                  bias_initializer=tf.constant_initializer(self.output_bias),
                                  **self._layer_params)
        
        return net

    @abc.abstractmethod
    def _build_target(self, x):
        pass
    
    def _build_prior(self):
        with tf.name_scope('prior'):
            return tf.contrib.distributions.Bernoulli(probs=self.prior_p)

    def _build_elbo(self, encoder_logits, reconstruction, scope_name):
        with tf.name_scope('elbo'):
            q = tf.sigmoid(encoder_logits)
            q_neg = tf.sigmoid(-encoder_logits)

            kl = q * (log_sigmoid(encoder_logits) - tf.log(self.prior_p)) \
                 + q_neg * (log_sigmoid(-encoder_logits) - tf.log(1 - self.prior_p))
            kl_per_object = tf.reduce_sum(kl, axis=1)
            kl_mean, kl_var = tf.nn.moments(kl_per_object, axes=[0])

            reconstruction_per_object = tf.reduce_sum(reconstruction, axis=1)
            elbo_per_object = reconstruction_per_object - kl_per_object

            elbo_mean, elbo_var = tf.nn.moments(elbo_per_object, axes=[0])

        with tf.name_scope(scope_name):
            model_utils.summary_mean_and_std('elbo', elbo_mean, elbo_var ** 0.5)
            model_utils.summary_mean_and_std('kl', kl_mean, kl_var ** 0.5)

        return elbo_per_object

    def _build_loss(self, x):
        surrogate_loss = self._build_target(x)

        regularization = self._get_regularization()
        loss = -tf.reduce_mean(surrogate_loss, axis=0) + regularization

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('regularization', regularization)

        return loss

    def _get_regularization(self):
        regularizables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        regularizer = tf.contrib.layers.l2_regularizer(self.lam)
        return tf.contrib.layers.apply_regularization(regularizer, regularizables)

    def _encoder_loss(self, stochastic_tensor, value, loss):
        return tf.contrib.bayesflow.stochastic_gradient_estimators.score_function(stochastic_tensor, value, loss)

    def _build(self, x, encoder_builder, reuse, scope_name):
        encoder_logits = self._build_encoder_logits(x, reuse=reuse)
        encoder = encoder_builder(encoder_logits)
        code = tf.contrib.bayesflow.stochastic_tensor.StochasticTensor(encoder, loss_fn=self._encoder_loss)

        code = tf.to_float(code)

        decoder_logits = self._build_decoder_logits(code, reuse=reuse)
        decoder = tf.contrib.distributions.Bernoulli(logits=decoder_logits)
        reconstruction = tf.contrib.bayesflow.stochastic_tensor.StochasticTensor(decoder)

        elbo_per_object = self._build_elbo(encoder_logits, decoder.log_prob(x), scope_name)
        return code, reconstruction, elbo_per_object

    def _build_jackknife_elbo(self, x, k_samples, max_depth, reuse):
        assert k_samples > 0
        assert max_depth < k_samples

        encoder_logits = self._build_encoder_logits(x, reuse=reuse)
        encoder = tf.contrib.distributions.Bernoulli(encoder_logits)  # N x C
        code = tf.to_float(encoder.sample(k_samples))  # K x N x C

        decoder_logits = self._build_decoder_logits(code, reuse=reuse)
        decoder = tf.contrib.distributions.Bernoulli(logits=decoder_logits)  # K x N x M

        prior = self._build_prior()

        with tf.name_scope("jackknife-elbo-depth_{}-samples_{}".format(max_depth, k_samples)):
            reconstruction = tf.reduce_sum(decoder.log_prob(x), axis=2)
            kl = tf.reduce_sum(encoder.log_prob(code) - prior.log_prob(code), axis=2)
            elbos = reconstruction - kl  # K x batch_size

            jackknife_elbo = self._jackknife(elbos, k_samples, max_depth)

        return tf.reduce_sum(jackknife_elbo, axis=0)

    @staticmethod
    def _jackknife(w, k_samples, max_depth):
        jackknife_elbo = 0
        for d in range(max_depth + 1):
            coef = (-1) ** d * np.exp(np.log(k_samples - d) * max_depth
                                      - sp.special.gammaln(d + 1) - sp.special.gammaln(max_depth - d + 1))

            subsample_size = k_samples - d
            subsample_estimators = [tf.reduce_logsumexp(tf.gather(w, subsample_indices), axis=0)
                                    for subsample_indices in itertools.combinations(range(k_samples), subsample_size)]

            jackknife_elbo += coef * (tf.reduce_mean(subsample_estimators, axis=0) - np.log(subsample_size))

        return jackknife_elbo

    def decode(self, z, deterministic=False):
        # TODO: refactor me
        decoder = self.discrete_reconstruction_.distribution
        reconstruction = decoder.mean() if deterministic else decoder.sample()
        return reconstruction.eval(feed_dict={self.discrete_code_: z})

    def encode(self, x, deterministic=False):
        code = self.discrete_code_.mean() if deterministic else self.discrete_code_.value()
        return code.eval(feed_dict={self.input_: x})

    @staticmethod
    def _to_signed_binary(x):
        return tf.subtract(tf.scalar_mul(2, x), 1)


class AbstractReparametrizedDVAE(AbstractDVAE):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _build_relaxed_encoder(self, logits):
        pass

    def _build_target(self, x):
        self.relaxed_code_, self.relaxed_reconstruction_, \
        self.relaxed_elbo_ = self._build(x, self._build_relaxed_encoder, reuse=tf.AUTO_REUSE, scope_name='relaxed')

        return self.relaxed_elbo_


class AbstractScoreFunctionDVAE(AbstractDVAE):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _baseline(self, x, code, reconstruction):
        pass

    @abc.abstractmethod
    def _baseline_bias(self, x, code, reconstruction):
        pass

    def _build_target(self, x):
        code = tf.stop_gradient(self.discrete_code_)
        code_distribution = code.distribution
        baseline = self._baseline(x, self.discrete_code_, self.discrete_reconstruction_)
        baseline_bias = self._baseline_bias(x, self.discrete_code_, self.discrete_reconstruction_)

        return tf.stop_gradient(self.discrete_elbo_ - baseline) * code_distribution.log_prob(code) + baseline_bias
