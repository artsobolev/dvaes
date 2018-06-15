import numpy as np
import tensorflow as tf

import time
import sys


class MeanVarianceEstimator:
    def __init__(self):
        self._mu = np.nan
        self._tau = 0.
        self._n = 0

    def update(self, x):
        mu = self._mu if not np.isnan(self._mu) else 0.

        self._mu = mu + (x - mu) / (self._n + 1)
        self._tau += (x - mu) * (x - self._mu)

        assert self._mu == self._mu
        assert self._tau == self._tau

        self._n += 1

    @property
    def mean(self):
        return self._mu

    @property
    def variance(self):
        return self._tau / (self._n - 1) if self._n > 1 else 0

    @property
    def std(self):
        return self.variance ** 0.5


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


def get_time_left(epochs_total, epochs_passed, timers, k=2):
    mu = 0
    variance = 0

    for period, timer in timers:
        fires_left = epochs_total / period - epochs_passed / period
        mu += timer.mean * fires_left
        variance += timer.variance * fires_left

    return mu + k * variance ** 0.5


def to_time_string(n):
    if n != n:
        return "Unknown"

    time_data = zip([np.inf, 24, 60, 60], '{}d {:2}h {:2}m {:2}s'.split(' '))
    res = []
    for size, fmt in reversed(time_data):
        n, k = divmod(n, size)
        if k:
            res.append(fmt.format(int(k)))

    return " ".join(reversed(res))


def train(dvae, X_train, X_val, learning_rate, epochs_total, batch_size, eval_batch_size, evaluate_every=None,
          shuffle=True, experiment_path='./experiment/', subset_validation=1000 * 1000 * 1000, sess=None):

    sess = sess or tf.get_default_session()
    evaluate_every = evaluate_every or {}

    global_step = tf.train.create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(dvae.loss_, global_step=global_step)

    train_writer = tf.summary.FileWriter(experiment_path + "/logs", sess.graph)

    train_size = len(X_train)
    indices = np.arange(train_size)
    batches = (train_size + batch_size - 1) / batch_size

    avg_batch_time = MeanVarianceEstimator()
    avg_k_elbo_time = {kd: MeanVarianceEstimator() for kd in evaluate_every.iterkeys()}

    timers = [(period, avg_k_elbo_time[k]) for k, period in evaluate_every.iteritems()]
    timers.append((1. / batches, avg_batch_time))

    most_accurate_eval = max(evaluate_every.keys(), key=lambda kd: kd[0])

    tf.global_variables_initializer().run()
    try:
        for epoch in range(epochs_total):
            if shuffle:
                np.random.shuffle(indices)

            elbos = {}
            for kd, kd_jackknife_elbo_eval_every in sorted(evaluate_every.items()):
                if epoch % kd_jackknife_elbo_eval_every != 0:
                    continue

                progress_line = "\rEpoch {}: computing {}-ELBO... {}".format(epoch, kd, "{percent:.2%}" + " " * 30)
                start = time.time()
                elbo = batch_evaluate(dvae.multisample_elbos_[kd], dvae.input_, X_val[:subset_validation],
                                      batch_size=eval_batch_size, progress_line=progress_line)
                elbos[kd] = elbo

                eval_time = time.time() - start
                avg_k_elbo_time[kd].update(eval_time)

                train_writer.add_summary(to_summary({"{}-sample {}-deep JK ELBO".format(*kd): elbo}),
                                         tf.train.global_step(sess, global_step))

                time_left = to_time_string(get_time_left(epochs_total, epoch, timers))
                print "\rEpoch {}: ETA: {}, {}-ELBO: {:.3f} " \
                      "(eval. time = {:.2f}, avg. = {:.2f})".format(epoch, time_left, kd, elbo,
                                                                    eval_time, avg_k_elbo_time[kd].mean)

            if (1, 0) in elbos and most_accurate_eval in elbos:
                kl = elbos[most_accurate_eval] - elbos[1, 0]
                train_writer.add_summary(to_summary({"{}-sample posterior KL".format(most_accurate_eval): kl}),
                                         tf.train.global_step(sess, global_step))

            for batch_id in range(batches):
                batch_begin = batch_id * batch_size
                batch_end = batch_begin + batch_size
                batch_indices = indices[batch_begin:batch_end]

                X_batch = X_train[batch_indices]
                X_samples = np.random.binomial(1, X_batch)

                start = time.time()
                _, summary = sess.run([train_op, dvae.summaries_op_], feed_dict={dvae.input_: X_samples})
                run_time = time.time() - start
                avg_batch_time.update(run_time)

                sys.stdout.write("\rEpoch {}.{}: Time per batch: {:.4f}s "
                                 "(avg. = {:.4f}s)".format(epoch, batch_id, run_time, avg_batch_time.mean) + " " * 30)
                sys.stdout.flush()

                train_writer.add_summary(summary, tf.train.global_step(sess, global_step))

            train_writer.flush()
    except (KeyboardInterrupt, SystemExit):
        print '\r' + '=' * 30
        print 'Training was stopped manually'
        print '=' * 30

    tf.train.Saver().save(sess, experiment_path + "/model")
