import tensorflow as tf

def summary_mean_and_std(name, mean, std):
    with tf.name_scope(name):
        return [tf.summary.scalar('mean', mean),
                tf.summary.scalar('std', std)]
