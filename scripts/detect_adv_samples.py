from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
from sklearn.neighbors import KernelDensity
from keras.models import load_model

from src.util import (get_data, get_noisy_samples, get_mc_predictions,
                      get_deep_representations, score_samples)

BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00}


def detect(model, X_train, Y_train, X_test, X_test_adv, X_test_noisy,
           dataset, batch_size):
    ### Get Bayesian uncertainty scores ###
    print('Getting Monte Carlo dropout variance predictions...')
    uncert_normal = get_mc_predictions(model, X_test, batch_size=batch_size)\
        .var(axis=0).mean(axis=1)
    uncert_adv = get_mc_predictions(model, X_test_adv, batch_size=batch_size)\
        .var(axis=0).mean(axis=1)
    uncert_noisy = get_mc_predictions(model, X_test_noisy,
                                      batch_size=batch_size)\
        .var(axis=0).mean(axis=1)

    ### Get KDE scores ###
    # Get deep feature representations
    print('Getting deep feature representations...')
    X_train_features = get_deep_representations(model, X_train,
                                                batch_size=batch_size)
    X_test_features = get_deep_representations(model, X_test,
                                               batch_size=batch_size)
    X_test_adv_features = get_deep_representations(model, X_test_adv,
                                                   batch_size=batch_size)
    X_test_noisy_features = get_deep_representations(model, X_test_noisy,
                                                     batch_size=batch_size)
    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(Y_train.shape[1]):
        class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]
    kdes = {}
    for i in range(Y_train.shape[1]):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=BANDWIDTHS[dataset])\
            .fit(X_train_features[class_inds[i]])
    # Get model predictions
    print('Computing model predictions...')
    preds_test = model.predict_classes(X_test, verbose=0,
                                       batch_size=batch_size)
    preds_test_adv = model.predict_classes(X_test_adv, verbose=0,
                                           batch_size=batch_size)
    preds_test_noisy = model.predict_classes(X_test_noisy, verbose=0,
                                             batch_size=batch_size)
    # Get density estimates
    print('computing densities...')
    densities_normal = score_samples(
        kdes,
        X_test_features,
        preds_test
    )
    densities_adv = score_samples(
        kdes,
        X_test_adv_features,
        preds_test_adv
    )
    densities_noisy = score_samples(
        kdes,
        X_test_noisy_features,
        preds_test_noisy
    )

    ### Build detector ###


def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    assert os.path.isfile('../data/model_%s.h5' % args.dataset), \
        'model file not found... must first train model using train_model.py.'
    assert os.path.isfile('../data/Adv_%s_%s.npy' %
                          (args.dataset, args.attack)), \
        'adversarial sample file not found... must first craft adversarial ' \
        'samples using craft_adv_samples.py'
    model = load_model('../data/model_%s.h5' % args.dataset)
    X_train, Y_train, X_test, Y_test = get_data(args.dataset)
    if args.attack == 'all':
        # TODO
        raise NotImplementedError('All types detector not yet implemented.')
    else:
        # load adversarial samples
        X_test_adv = np.load('../data/Adv_%s_%s.npy' % (args.dataset,
                                                        args.attack))
        # craft an equal number of noisy samples
        X_test_noisy = get_noisy_samples(X_test, X_test_adv, args.attack)
        detect(model, X_train, Y_train, X_test, X_test_adv, X_test_noisy,
               args.dataset, args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', "
             "'jsma' 'cw' or 'all'",
        required=False, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(attack='all')
    parser.set_defaults(batch_size=256)
    args = parser.parse_args()
    main(args)
