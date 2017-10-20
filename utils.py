import numpy as np
import tensorflow as tf

import time
import sys


def get_mnist_dataset():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets('MNIST_data', one_hot=True)


def train(dvae, X_train, X_val, learning_rate=1.0, epochs=10, batch_size=100,
          evaluate_every=10, shuffle=True, summaries_path='./experiment/', sess=None):
    if sess is None:
        sess = tf.get_default_session()

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(dvae.relaxed_loss_, global_step=global_step)

    train_writer = tf.summary.FileWriter(summaries_path, sess.graph)

    train_size = len(X_train)
    indices = np.arange(train_size)
    batches = (train_size + batch_size - 1) / batch_size

    avg_run_time = 0
    mu = 1

    tf.global_variables_initializer().run()
    for epoch in range(epochs):
        if shuffle:
            np.random.shuffle(indices)

        if epoch % evaluate_every == 0:
            start = time.time()
            multisample_elbos, summary = dvae.evaluate_multisample(X_val)
            eval_time = time.time() - start
            train_writer.add_summary(summary, tf.train.global_step(sess, global_step))

            ks = sorted(multisample_elbos.keys())
            elbos_str = ["k={}: {:.3f}".format(k, multisample_elbos[k]) for k in ks]
            print "\rEpoch {}: ELBOs: {} (eval. time = {:.2f})".format(epoch, ", ".join(elbos_str), eval_time)
            
        for batch_id in range(batches):
            batch_begin = batch_id * batch_size
            batch_end = batch_begin + batch_size
            batch_indices = indices[batch_begin:batch_end]

            X_batch = X_train[batch_indices]
            X_samples = np.random.binomial(1, X_batch)

            start = time.time()
            _, summary = sess.run([train_op, dvae.summaries_op_], feed_dict={dvae.input_: X_samples})
            execution_time = time.time() - start

            avg_run_time = mu * execution_time + (1 - mu) * avg_run_time
            mu = 0.9
            sys.stdout.write("\rEpoch {}.{}: Time per batch: {:.2f}s".format(epoch, batch_id, avg_run_time) + " " * 30)
            sys.stdout.flush()

            train_writer.add_summary(summary, tf.train.global_step(sess, global_step))