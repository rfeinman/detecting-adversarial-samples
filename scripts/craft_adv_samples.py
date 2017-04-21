from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model

from src.util import get_data
from src.attacks import fast_gradient_sign_method, basic_iterative_method


def craft_one_type(sess, model, X_test, Y_test, attack):
    """

    :param sess:
    :param model:
    :param X_test:
    :param Y_test:
    :param attack:
    :return:
    """
    if attack == 'fgsm':
        X_test_adv = fast_gradient_sign_method(
            sess, model, X_test, Y_test, eps=0.35, clip_min=0., clip_max=1.
        )
    elif attack == 'bim-a':
        its, results = basic_iterative_method(
            sess, model, X_test, Y_test, eps=0.35,
            eps_iter=0.02, clip_min=0., clip_max=1.
        )
        #TODO
        raise NotImplementedError('JSMA attack not yet implemented.')
    elif attack == 'bim-b':
        its, results = basic_iterative_method(
            sess, model, X_test, Y_test, eps=0.35,
            eps_iter=0.02, clip_min=0., clip_max=1.
        )
        X_test_adv = results[-1]
    elif attack == 'jsma':
        #TODO
        raise NotImplementedError('JSMA attack not yet implemented.')
    else:
        #TODO
        raise NotImplementedError('CW attack not yet implemented.')
    _, acc = model.evaluate(X_test_adv, Y_test, batch_size=512, verbose=0)
    print("Accuracy on the adversarial test set: %0.2f%%" % (100 * acc))
    np.save('../data/X_adv_%s.npy' % args.attack, X_test_adv)


def craft_all_types(sess, model, X_test, Y_test):
    """

    :param sess:
    :param model:
    :param X_test:
    :param Y_test:
    :return:
    """
    #TODO
    raise NotImplementedError("'All' option not yet implemented.")
    return


def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    assert os.path.isfile('../data/model_%s.h5' % args.dataset), \
        'model file not found... must first train model using train_model.py.'
    print('Dataset: %s. Attack: %s' % (args.dataset, args.attack))
    # create TF session, set it as Keras backend
    sess = tf.Session()
    K.set_session(sess)
    model = load_model('../data/model_%s.h5' % args.dataset)
    _, _, X_test, Y_test = get_data(args.dataset)
    _, acc = model.evaluate(X_test, Y_test, batch_size=512, verbose=0)
    print("Accuracy on the test set: %0.2f%%" % (100*acc))
    if args.attack == 'all':
        craft_all_types(sess, model, X_test, Y_test)
    else:
        craft_one_type(sess, model, X_test, Y_test, args.attack)
    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma', 'cw'"
             " or 'all'",
        required=False, type=str
    )
    parser.set_defaults(attack='all')
    args = parser.parse_args()
    main(args)
