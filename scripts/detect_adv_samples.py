from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
from keras.models import load_model

from src.util import get_data, get_noisy_samples, get_mc_predictions


def detect_one_type(model, X_test, attack):
    X_test_adv = np.load('../data/X_adv_%s.npy' % attack)
    # craft an equal number of noisy samples
    X_test_noisy = get_noisy_samples(X_test, X_test_adv, attack)
    # Get dropout variance values
    print('Getting Monte Carlo dropout variance predictions...')
    uncert_normal = get_mc_predictions(model, X_test).var(axis=0).mean(axis=1)
    uncert_adv = get_mc_predictions(model, X_test_adv).var(axis=0).mean(axis=1)
    uncert_noisy = get_mc_predictions(model, X_test_noisy)\
        .var(axis=0).mean(axis=1)
    # Get KDE estimates
    print('Getting KDE estimates...')


def detect_all_types(model, X_test):
    # TODO
    raise NotImplementedError("'All' option not yet implemented.")


def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    assert os.path.isfile('../data/model_%s.h5' % args.dataset), \
        'model file not found... must first train model using train_model.py.'
    assert os.path.isfile('../data/X_adv_%s.npy' % args.attack), \
        'adversarial sample file not found... must first craft adversarial ' \
        'samples using craft_adv_samples.py'
    model = load_model('../data/model_%s.h5' % args.dataset)
    X_train, Y_train, X_test, Y_test = get_data(args.dataset)
    if args.attack == 'all':
        detect_all_types(model, X_test)
    else:
        detect_one_type(model, X_test, args.attack)


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
    parser.set_defaults(attack='all')
    args = parser.parse_args()
    main(args)
