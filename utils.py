import numpy as np
import tensorflow as tf

import time
import sys


def get_mnist_dataset():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets('MNIST_data', one_hot=True)


def batch_evaluate(target_node, input_node, data, batch_size, progress_line=None):
    n = len(data)
    batches = (n + batch_size - 1) / batch_size

    res = 0
    for batch_id in range(batches):
        if progress_line is not None:
            sys.stdout.write(progress_line.format(percent=batch_id * 1. / batches))
            sys.stdout.flush()

        begin = batch_id * batch_size
        end = begin + batch_size
        data_batch = data[begin:end]

        res += target_node.eval(feed_dict={input_node: data_batch}) / n

    return res


def to_summary(tag_values):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_values.iteritems()])


def train(dvae, X_train, X_val, learning_rate=1.0, epochs=10, batch_size=100, evaluate_every=None, shuffle=True,
          summaries_path='./experiment/', subset_validation=1000*1000*1000, sess=None):

    sess = sess or tf.get_default_session()
    evaluate_every = evaluate_every or {}

    global_step = tf.train.create_global_step()
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

        for k_samples, k_sample_elbo_evaluate_every in sorted(evaluate_every.items()):
            if epoch % k_sample_elbo_evaluate_every != 0:
                continue

            progress_line = "\rEpoch {}: computing {}-ELBO... {}".format(epoch, k_samples,
                                                                         "{percent:.2%}" + " " * 30)
            start = time.time()
            elbo = batch_evaluate(dvae.multisample_elbos_[k_samples], dvae.input_, X_val[:subset_validation],
                                  batch_size=dvae.batch_size, progress_line=progress_line)

            eval_time = time.time() - start

            train_writer.add_summary(to_summary({"{}-sample ELBO".format(k_samples): elbo}),
                                     tf.train.global_step(sess, global_step))

            print "\rEpoch {}: {}-ELBO: {:.3f} (eval. time = {:.2f})".format(epoch, k_samples, elbo, eval_time)
            
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
            sys.stdout.write("\rEpoch {}.{}: Time per batch: {:.4f}s".format(epoch, batch_id, avg_run_time) + " " * 30)
            sys.stdout.flush()

            train_writer.add_summary(summary, tf.train.global_step(sess, global_step))

        train_writer.flush()
