import tensorflow as tf

from abstract_dvae import AbstractScoreFunctionDVAE


class MuPropDVAE(AbstractScoreFunctionDVAE):
    @staticmethod
    def _mean_encoder(logits):
        return tf.contrib.distributions.Deterministic(tf.sigmoid(logits))

    def _baseline(self, x, code, reconstruction):
        relaxed_code, relaxed_reconstruction, relaxed_elbo = self._build(x, self._mean_encoder,
                                                                         reuse=True, scope_name='relaxed')

        tf.gradients()
        code - relaxed_code
        return