import numpy as np
import tensorflow as tf

import models
import utils

import argparse

np.random.seed(0)

if __name__ == "__main__":

    available_models = [x for x in dir(models) if 'VAE' in x]

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        'model',
        choices=available_models,
        help='Model class.')

    argparser.add_argument(
        '--code_size', type=int, default=200,
        help='Dimension of latent code')

    argparser.add_argument(
        '--prior_proba', type=float, default=0.5,
        help='Prior probability on code')

    argparser.add_argument(
        '--learning_rate', type=float, default=1e-3,
        help='Learning rate')

    argparser.add_argument(
        '--lam', type=float, default=1e-3,
        help='Regularisation coefficient')

    argparser.add_argument(
        '--tau', type=float, default=1.0,
        help='Relaxation temperature')

    argparser.add_argument(
        '--relaxation_distribution', type=str, default='Uniform',
        choices=models.GeneralizedRelaxedDVAE.DISTRIBUTION_FACTORIES.keys(),
        help='Underlying distribution for Generalized Sigmoid relaxation')

    argparser.add_argument(
        '--noise_distribution', type=str, default='Normal',
        choices=models.NoiseRelaxedDVAE.NOISE_FACTORIES.keys(),
        help='Noise distribution for noise relaxation')

    argparser.add_argument(
        '--batch_size', type=int, default=50,
        help='Batch size')

    argparser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of epochs')

    argparser.add_argument(
        '--evaluate_every', type=int, default=5,
        help='Evaluate model every X batches')

    argparser.add_argument(
        '--experiment_path', type=str, default='experiments/tmp/',
        help='Path to save experiment\'s data')

    argparser.add_argument(
        '--subset_validation', type=int, default=1000*1000*1000,
        help='Number of validation samples to compute marginal '
             'log-likelihood on.')

    args = argparser.parse_args()

    if args.model not in available_models:
        raise ValueError("Unknown model name: {}".format(args.model))
    model_class = getattr(models, args.model)

    dataset = utils.get_mnist_dataset()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        dvae = model_class(code_size=args.code_size, input_size=28*28, prior_p=args.prior_proba, lam=args.lam,
                           tau=args.tau, relaxation_distribution=args.relaxation_distribution,
                           batch_size=args.batch_size, noise_distribution=args.noise_distribution)

        utils.train(dvae, dataset.train.images, dataset.validation.images, learning_rate=args.learning_rate,
                    epochs=args.epochs, batch_size=args.batch_size, evaluate_every=args.evaluate_every,
                    summaries_path=args.experiment_path, sess=sess, subset_validation=args.subset_validation)

        save_path = tf.train.Saver().save(sess, args.experiment_path + "/model")
