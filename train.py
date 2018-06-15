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
        '--learning_rate', type=float, default=1e-4,
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
        '--eval_batch_size', type=int, default=50,
        help='Batch size for evaluation')

    argparser.add_argument(
        '--epochs', type=int, default=10000,
        help='Number of epochs')

    argparser.add_argument(
        '--jackknife_samples', type=int, nargs='+', default=[1, 10, 10, 10, 100, 100, 100, 1000, 1000],
        help='List of sample sizes for jackknife ELBOs')

    argparser.add_argument(
        '--jackknife_depths', type=int, nargs='+', default=[0, 0, 1, 2, 0, 1, 2, 0, 1],
        help='List of depths for jackknife ELBOs')

    argparser.add_argument(
        '--evaluate_every', type=int, nargs='+', default=[1, 3, 3, 3, 100, 100, 100, 200, 200],
        help='Evaluate model on ELBO every X epochs '
             '(for number for each multisample ELBO)')

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

    evaluate_every = dict(zip(zip(args.jackknife_samples, args.jackknife_depths), args.evaluate_every))

    model_class = getattr(models, args.model)
    dataset = utils.get_mnist_dataset()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        train_mean = dataset.train.images.mean(axis=0)
        output_bias = -np.log(1. / np.clip(train_mean, 0.001, 0.999) - 1.)

        dvae = model_class(code_size=args.code_size, input_size=28*28, prior_p=args.prior_proba,
                           lam=args.lam, tau=args.tau, relaxation_distribution=args.relaxation_distribution,
                           output_bias=output_bias, jackknife_depths=args.jackknife_depths,
                           jackknife_samples=args.jackknife_samples, noise_distribution=args.noise_distribution)

        utils.train(dvae, dataset.train.images, dataset.validation.images, learning_rate=args.learning_rate,
                    epochs_total=args.epochs, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                    evaluate_every=evaluate_every, experiment_path=args.experiment_path,
                    subset_validation=args.subset_validation, sess=sess)
