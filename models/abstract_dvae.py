import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers

import model_utils

log_sigmoid = lambda x: -tf.nn.softplus(-x)

class AbstractDVAE:
    def __init__(self, code_size, p, lam, multisample_ks=(100, 1000, 10000)):
        self.code_size = code_size
        self.p = p
        self.lam = lam
        
        self.input_ = tf.placeholder(tf.float32, shape=(None, 28 * 28))

        self.relaxed_encoder_, self.relaxed_code_, \
        self.relaxed_decoder_, self.relaxed_loss_, self.relaxed_summaries_ = self._build(self.input_, relaxed=True)

        self.discrete_encoder_, self.discrete_code_, \
        self.discrete_decoder_, self.discrete_loss_, self.discrete_summaries_ = self._build(self.input_, relaxed=False)

        self.summaries_op_ = tf.summary.merge(self.relaxed_summaries_ + self.discrete_summaries_)

        multisample_elbos, multisample_summaries = zip(*[self._build_multisample_elbo(self.input_, k)
                                                         for k in multisample_ks])
        self.multisample_elbos_ = dict(zip(multisample_ks, multisample_elbos))
        self.multisample_summaries_op_ = tf.summary.merge(multisample_summaries)
    
    def _build_encoder_logits(self, eh0):
        with tf.name_scope('encoder'):        
            eh1 = tf.layers.dense(eh0, 200, activation=tf.nn.tanh)
            eh2 = tf.layers.dense(eh1, self.code_size, activation=None)

        return eh2 

    def _build_decoder_logits(self, dh0):
        with tf.name_scope('decoder'):
            dh1 = tf.layers.dense(dh0, 200, activation=tf.nn.tanh)
            dh2 = tf.layers.dense(dh1, 28*28, activation=None)
        
        return dh2 
    
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
            return tf.contrib.distributions.Bernoulli(probs=self.p)

    def _build_loss(self, encoder_logits, reconstruction, scope_name):
        summaries = []
        with tf.name_scope('loss'):
            q = tf.sigmoid(encoder_logits)
            q_neg = tf.sigmoid(encoder_logits)

            kl = q * (log_sigmoid(encoder_logits) - tf.log(self.p)) \
                 + q_neg * (log_sigmoid(-encoder_logits) - tf.log(1 - self.p))
            kl_per_object = tf.reduce_sum(kl, axis=1)
            kl_mean, kl_var = tf.nn.moments(kl_per_object, axes=[0])

            reconstruction_per_object = tf.reduce_sum(reconstruction, axis=1)
            elbo_per_object = reconstruction_per_object - kl_per_object
            elbo_mean, elbo_var = tf.nn.moments(elbo_per_object, axes=[0])

            reg_losses = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            regularizer = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.lam), reg_losses)
            loss = -elbo_mean + regularizer

        with tf.name_scope(scope_name):
            summaries.append(tf.summary.scalar('loss', loss))
            summaries += model_utils.summary_mean_and_std('elbo', elbo_mean, elbo_var ** 0.5)
            summaries += model_utils.summary_mean_and_std('kl', kl_mean, kl_var ** 0.5)

        return loss, summaries

    def _build(self, x, relaxed=False):
        encoder_logits = self._build_encoder_logits(x)
        encoder = (self._build_relaxed_encoder if relaxed else self._build_encoder)(encoder_logits)
        code = tf.to_float(encoder.sample())

        decoder_logits = self._build_decoder_logits(code)
        decoder = self._build_decoder(decoder_logits)

        loss, summaries = self._build_loss(encoder_logits, decoder.log_prob(x), 'relaxed' if relaxed else 'discrete')
        return encoder, code, decoder, loss, summaries

    def _build_multisample_elbo(self, x, K):
        encoder_logits = self._build_encoder_logits(x)
        encoder = self._build_encoder(encoder_logits) # N x C
        code = tf.to_float(encoder.sample(K)) # N x K x C

        decoder_logits = self._build_decoder_logits(code)
        decoder = self._build_decoder(decoder_logits) # N x K x M

        prior = self._build_prior()

        with tf.name_scope("{}-sample_elbo".format(K)):
            reconstruction = tf.reduce_sum(decoder.log_prob(x), axis=2)
            kl = tf.reduce_sum(encoder.log_prob(code) - prior.log_prob(code), axis=2)
            elbos = reconstruction - kl
            multisample_elbo = tf.reduce_logsumexp(elbos, axis=1) - np.log(K)

            multisample_elbo_mean, multisample_elbo_var = tf.nn.moments(multisample_elbo, axes=[0])
            summaries = model_utils.summary_mean_and_std('multisample_elbo',
                                                         multisample_elbo_mean, multisample_elbo_var ** 0.5)

        return multisample_elbo_mean, summaries
    
    def decode(self, z, deterministic=False):
        decoder = self.discrete_decoder_
        ret = decoder.mean() if deterministic else decoder.sample()
        return ret.eval(feed_dict={self.discrete_code_: z})

    def encode(self, x, deterministic=False):
        encoder = self.discrete_encoder_
        ret = encoder.mean() if deterministic else encoder.sample()
        return ret.eval(feed_dict={self.input_: x})