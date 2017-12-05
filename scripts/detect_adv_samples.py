from __future__ import division, absolute_import, print_function

import os
import argparse
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
from keras.models import load_model

from detect.util import (get_data, get_noisy_samples, get_mc_predictions,
                         get_deep_representations, score_samples, normalize,
                         train_lr, compute_roc)

# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00}


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
    print('Loading the data and model...')
    # Load the model
    model = load_model('../data/model_%s.h5' % args.dataset)
    # Load the dataset
    X_train, Y_train, X_test, Y_test = get_data(args.dataset)
    # Check attack type, select adversarial and noisy samples accordingly
    print('Loading noisy and adversarial samples...')
    if args.attack == 'all':
        # TODO: implement 'all' option
        #X_test_adv = ...
        #X_test_noisy = ...
        raise NotImplementedError("'All' types detector not yet implemented.")
    else:
        # Load adversarial samples
        X_test_adv = np.load('../data/Adv_%s_%s.npy' % (args.dataset,
                                                        args.attack))
        # Craft an equal number of noisy samples
        X_test_noisy = get_noisy_samples(X_test, X_test_adv, args.dataset,
                                         args.attack)
    # Check model accuracies on each sample type
    for s_type, dataset in zip(['normal', 'noisy', 'adversarial'],
                               [X_test, X_test_noisy, X_test_adv]):
        _, acc = model.evaluate(dataset, Y_test, batch_size=args.batch_size,
                                verbose=0)
        print("Model accuracy on the %s test set: %0.2f%%" %
              (s_type, 100 * acc))
        # Compute and display average perturbation sizes
        if not s_type == 'normal':
            l2_diff = np.linalg.norm(
                dataset.reshape((len(X_test), -1)) -
                X_test.reshape((len(X_test), -1)),
                axis=1
            ).mean()
            print("Average L-2 perturbation size of the %s test set: %0.2f" %
                  (s_type, l2_diff))
    # Refine the normal, noisy and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    preds_test = model.predict_classes(X_test, verbose=0,
                                       batch_size=args.batch_size)
    inds_correct = np.where(preds_test == Y_test.argmax(axis=1))[0]
    X_test = X_test[inds_correct]
    X_test_noisy = X_test_noisy[inds_correct]
    X_test_adv = X_test_adv[inds_correct]

    ## Get Bayesian uncertainty scores
    print('Getting Monte Carlo dropout variance predictions...')
    uncerts_normal = get_mc_predictions(model, X_test,
                                        batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)
    uncerts_noisy = get_mc_predictions(model, X_test_noisy,
                                       batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)
    uncerts_adv = get_mc_predictions(model, X_test_adv,
                                     batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)

    ## Get KDE scores
    # Get deep feature representations
    print('Getting deep feature representations...')
    X_train_features = get_deep_representations(model, X_train,
                                                batch_size=args.batch_size)
    X_test_normal_features = get_deep_representations(model, X_test,
                                                      batch_size=args.batch_size)
    X_test_noisy_features = get_deep_representations(model, X_test_noisy,
                                                     batch_size=args.batch_size)
    X_test_adv_features = get_deep_representations(model, X_test_adv,
                                                   batch_size=args.batch_size)
    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(Y_train.shape[1]):
        class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]
    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined "
                  "optimal for the specific CNN models of the paper. If you've "
                  "changed your model, you'll need to re-optimize the "
                  "bandwidth.")
    for i in range(Y_train.shape[1]):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=BANDWIDTHS[args.dataset]) \
            .fit(X_train_features[class_inds[i]])
    # Get model predictions
    print('Computing model predictions...')
    preds_test_normal = model.predict_classes(X_test, verbose=0,
                                              batch_size=args.batch_size)
    preds_test_noisy = model.predict_classes(X_test_noisy, verbose=0,
                                             batch_size=args.batch_size)
    preds_test_adv = model.predict_classes(X_test_adv, verbose=0,
                                           batch_size=args.batch_size)
    # Get density estimates
    print('computing densities...')
    densities_normal = score_samples(
        kdes,
        X_test_normal_features,
        preds_test_normal
    )
    densities_noisy = score_samples(
        kdes,
        X_test_noisy_features,
        preds_test_noisy
    )
    densities_adv = score_samples(
        kdes,
        X_test_adv_features,
        preds_test_adv
    )

    ## Z-score the uncertainty and density values
    uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z = normalize(
        uncerts_normal,
        uncerts_adv,
        uncerts_noisy
    )
    densities_normal_z, densities_adv_z, densities_noisy_z = normalize(
        densities_normal,
        densities_adv,
        densities_noisy
    )

    ## Build detector
    values, labels, lr = train_lr(
        densities_pos=densities_adv_z,
        densities_neg=np.concatenate((densities_normal_z, densities_noisy_z)),
        uncerts_pos=uncerts_adv_z,
        uncerts_neg=np.concatenate((uncerts_normal_z, uncerts_noisy_z))
    )

    ## Evaluate detector
    # Compute logistic regression model predictions
    probs = lr.predict_proba(values)[:, 1]
    # Compute AUC
    n_samples = len(X_test)
    # The first 2/3 of 'probs' is the negative class (normal and noisy samples),
    # and the last 1/3 is the positive class (adversarial samples).
    _, _, auc_score = compute_roc(
        probs_neg=probs[:2 * n_samples],
        probs_pos=probs[2 * n_samples:]
    )
    print('Detector ROC-AUC score: %0.4f' % auc_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma' 'cw' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=256)
    args = parser.parse_args()
    main(args)
