import tensorflow as tf

from abstract_dvae import AbstractDVAE

class AsymptoticallyReparametrisedDVAE(AbstractDVAE):
    def __init__(self, *args, **kwargs):
        AbstractDVAE.__init__(self, *args, **kwargs)
    
    def _build_relaxed_encoder(self, logits):
        with tf.name_scope('encoder'):        
            q = tf.sigmoid(logits)
            q_neg = tf.sigmoid(-logits)
            return tf.contrib.distributions.Normal(2 * q - 1, (4 * q * q_neg) ** 0.5)
