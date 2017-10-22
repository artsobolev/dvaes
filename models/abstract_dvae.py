import tensorflow as tf
import numpy as np

import model_utils


def log_sigmoid(x):
    return -tf.nn.softplus(-x)


class AbstractDVAE:
    def __init__(self, code_size, input_size, prior_p, lam, output_bias, multisample_ks=(), *args, **kwargs):
        self.code_size = code_size
        self.input_size = input_size
        self.prior_p = prior_p
        self.lam = lam
        self.output_bias = output_bias
        self.batch_size = kwargs.get('batch_size')

        self._layer_params = dict(kernel_initializer=tf.contrib.layers.xavier_initializer())

        # Can't use dynamic-sized batching since some Normal distribution's
        # quantile function can't work with it
        self.input_ = tf.placeholder(tf.float32, shape=(self.batch_size, self.input_size), name='Input')

        self.relaxed_encoder_, self.relaxed_code_,\
        self.relaxed_decoder_, self.relaxed_loss_, self.relaxed_summaries_ = self._build(self.input_,
                                                                                         reuse=False, relaxed=True)

        self.discrete_encoder_, self.discrete_code_, \
        self.discrete_decoder_, self.discrete_loss_, self.discrete_summaries_ = self._build(self.input_,
                                                                                            reuse=True, relaxed=False)

        self.summaries_op_ = tf.summary.merge(self.relaxed_summaries_ + self.discrete_summaries_)

        self.multisample_elbos_ = {k: self._build_multisample_elbo(self.input_, k, reuse=True) for k in multisample_ks}

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
    
    def _build_decoder(self, logits):
        with tf.name_scope('decoder'):
            return tf.contrib.distributions.Bernoulli(logits=logits)
    
    def _build_encoder(self, logits):
        with tf.name_scope('encoder'):
            return tf.contrib.distributions.Bernoulli(logits=logits)

    def _build_relaxed_encoder(self, logits):
        raise NotImplementedError()
    
    def _build_prior(self):
        with tf.name_scope('prior'):
            return tf.contrib.distributions.Bernoulli(probs=self.prior_p)

    def _build_loss(self, encoder_logits, reconstruction, scope_name):
        summaries = []
        with tf.name_scope('loss'):
            q = tf.sigmoid(encoder_logits)
            q_neg = tf.sigmoid(-encoder_logits)

            kl = q * (log_sigmoid(encoder_logits) - tf.log(self.prior_p)) \
                 + q_neg * (log_sigmoid(-encoder_logits) - tf.log(1 - self.prior_p))
            kl_per_object = tf.reduce_sum(kl, axis=1)
            kl_mean, kl_var = tf.nn.moments(kl_per_object, axes=[0])

            reconstruction_per_object = tf.reduce_sum(reconstruction, axis=1)
            elbo_per_object = reconstruction_per_object - kl_per_object
            elbo_mean, elbo_var = tf.nn.moments(elbo_per_object, axes=[0])

            regularization = self._get_regularization()
            loss = -elbo_mean + regularization

        with tf.name_scope(scope_name):
            summaries.append(tf.summary.scalar('loss', loss))
            summaries.append(tf.summary.scalar('regularization', regularization))
            summaries += model_utils.summary_mean_and_std('elbo', elbo_mean, elbo_var ** 0.5)
            summaries += model_utils.summary_mean_and_std('kl', kl_mean, kl_var ** 0.5)

        return loss, summaries

    def _get_regularization(self):
        regularizables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        regularizer = tf.contrib.layers.l2_regularizer(self.lam)
        return tf.contrib.layers.apply_regularization(regularizer, regularizables)

    def _build(self, x, reuse, relaxed=False):
        encoder_logits = self._build_encoder_logits(x, reuse=reuse)
        encoder = (self._build_relaxed_encoder if relaxed else self._build_encoder)(encoder_logits)
        code = tf.to_float(encoder.sample())

        decoder_logits = self._build_decoder_logits(code, reuse=reuse)
        decoder = self._build_decoder(decoder_logits)

        loss, summaries = self._build_loss(encoder_logits, decoder.log_prob(x), 'relaxed' if relaxed else 'discrete')
        return encoder, code, decoder, loss, summaries

    def _build_multisample_elbo(self, x, k_samples, reuse):
        encoder_logits = self._build_encoder_logits(x, reuse=reuse)
        encoder = self._build_encoder(encoder_logits)  # N x C
        code = tf.to_float(encoder.sample(k_samples))  # K x N x C

        decoder_logits = self._build_decoder_logits(code, reuse=reuse)
        decoder = self._build_decoder(decoder_logits) # K x N x M

        prior = self._build_prior()

        with tf.name_scope("{}-sample_elbo".format(k_samples)):
            reconstruction = tf.reduce_sum(decoder.log_prob(x), axis=2)
            kl = tf.reduce_sum(encoder.log_prob(code) - prior.log_prob(code), axis=2)
            elbos = reconstruction - kl
            multisample_elbo = tf.reduce_logsumexp(elbos, axis=0) - np.log(k_samples)

        return tf.reduce_sum(multisample_elbo, axis=0)
    
    def decode(self, z, deterministic=False):
        decoder = self.discrete_decoder_
        reconstruction = decoder.mean() if deterministic else decoder.sample()
        return reconstruction.eval(feed_dict={self.discrete_code_: z})

    def encode(self, x, deterministic=False):
        encoder = self.discrete_encoder_
        code = encoder.mean() if deterministic else encoder.sample()
        return code.eval(feed_dict={self.input_: x})

    @staticmethod
    def _to_signed_binary(x):
        return 2 * x - 1
