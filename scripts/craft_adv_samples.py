import os
import argparse
import tensorflow as tf
import keras.backend as K
from keras.models import load_model

from src.util import get_data
from src.attacks import fast_gradient_sign_method

def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    print('Dataset: %s' % args.dataset)
    assert os.path.isfile('../data/model_%s.h5' % args.dataset), \
        'model file not found... must first train model using train_model.py.'
    # create TF session, set it as Keras backend
    sess = tf.Session()
    K.set_session(sess)
    model = load_model('../data/model_%s.h5' % args.dataset)
    _, _, X_test, Y_test = get_data(args.dataset)
    _, acc = model.evaluate(X_test, Y_test, batch_size=512, verbose=0)
    print("Accuracy on the test set: %0.2f%%" % (100*acc))
    X_test_adv = fast_gradient_sign_method(
        sess, model, X_test, Y_test, eps=0.35, clip_min=0., clip_max=1.
    )
    _, acc = model.evaluate(X_test_adv, Y_test, batch_size=512, verbose=0)
    print("Accuracy on the adversarial test set: %0.2f%%" % (100 * acc))
    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    args = parser.parse_args()
    main(args)
