from __future__ import absolute_import
from __future__ import print_function

import os
import multiprocessing as mp
from subprocess import call
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import keras.backend as K
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2


def get_data(dataset='mnist'):
    """
    TODO
    :param dataset: TODO
    :return: TODO
    """
    assert dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    if dataset == 'mnist':
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # reshape to (n_samples, 28, 28, 1)
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
    elif dataset == 'cifar':
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    else:
        if not os.path.isfile("../data/svhn_train.mat"):
            print('Downloading SVHN train set...')
            call(
                "curl -o ../data/svhn_train.mat "
                "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                shell=True
            )
        if not os.path.isfile("../data/svhn_test.mat"):
            print('Downloading SVHN test set...')
            call(
                "curl -o ../data/svhn_test.mat "
                "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                shell=True
            )
        train = sio.loadmat('../data/svhn_train.mat')
        test = sio.loadmat('../data/svhn_test.mat')
        X_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
        X_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
        # reshape (n_samples, 1) to (n_samples,) and change 1-index
        # to 0-index
        y_train = np.reshape(train['y'], (-1,)) - 1
        y_test = np.reshape(test['y'], (-1,)) - 1

    # cast pixels to floats, normalize to [0, 1] range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # one-hot-encode the labels
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    return X_train, Y_train, X_test, Y_test


def get_model(dataset='mnist'):
    """
    Takes in a parameter indicating which model type to use ('mnist',
    'cifar' or 'svhn') and returns the appropriate Keras model.
    :param dataset: TODO
    :return: TODO
    """
    assert dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    if dataset == 'mnist':
        # MNIST model
        layers = [
            Conv2D(
                64, (3, 3),
                padding='valid',
                input_shape=(28, 28, 1)
            ),
            Activation('relu'),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(128),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')
        ]
    elif dataset == 'cifar':
        # CIFAR-10 model
        layers = [
            Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same'),
            Activation('relu'),
            Conv2D(32, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), padding='same'),
            Activation('relu'),
            Conv2D(64, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), padding='same'),
            Activation('relu'),
            Conv2D(128, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation('relu'),
            Dropout(0.5),
            Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')
        ]
    else:
        # SVHN model
        layers = [
            Conv2D(
                64, (3, 3),
                padding='valid',
                input_shape=(32, 32, 3)
            ),
            Activation('relu'),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(512),
            Activation('relu'),
            Dropout(0.5),
            Dense(128),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')
        ]

    model = Sequential()
    for layer in layers:
        model.add(layer)
    return model


def get_noisy_samples(X_test, X_test_adv, attack):
    """
    TODO
    :param X_test: TODO
    :param X_test_adv: TODO
    :param attack: TODO
    :return:
    """
    # TODO: Update this; should compute average perturbation size of \
    # TODO: adversarial samples and select noise size accordingly; should also \
    # TODO: use 'attack' parameter to determine the type of noise
    return np.minimum(
        np.maximum(
            X_test + np.random.normal(loc=0, scale=0.25, size=X_test.shape),
            0
        ),
        1
    )


def get_mc_predictions(model, X, nb_iter=50, batch_size=256):
    """

    :param model:
    :param X:
    :param nb_iter:
    :param batch_size:
    :return:
    """
    output_dim = model.layers[-1].output.shape[-1].value

    get_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-1].output]
    )

    def predict():
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        output = np.zeros(shape=(len(X), output_dim))
        for i in range(n_batches):
            output[i * batch_size:(i + 1) * batch_size] = \
                get_output([X[i * batch_size:(i + 1) * batch_size], 1])[0]
        return output

    preds_mc = []
    for i in tqdm(range(nb_iter)):
        preds_mc.append(predict())
    return np.asarray(preds_mc)


def get_deep_representations(model, X, batch_size=256):
    #last_hidden_layer = f(model)
    # last hidden layer is always at index -4
    output_dim = model.layers[-4].output.shape[-1].value
    get_encoding = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-4].output]
    )

    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    for i in range(n_batches):
        output[i * batch_size:(i + 1) * batch_size] = \
            get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[0]

    #return np.reshape(out, (X.shape[0], -1))
    return output


def score_point(tup):
    """

    :param tup:
    :return:
    """
    x, kde = tup
    return kde.score_samples(np.reshape(x, (1, -1)))[0]


def score_samples(kdes, samples, preds):
    """

    :param kdes:
    :param samples:
    :param preds:
    :return:
    """
    p = mp.Pool()
    results = np.asarray(
        p.map(
            score_point,
            [(x, kdes[i]) for x, i in zip(samples, preds)]
        )
    )
    p.close()
    p.join()
    return results
