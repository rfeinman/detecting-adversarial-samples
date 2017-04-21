from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from cleverhans.utils_tf import batch_eval


def fgsm(x, predictions, eps, clip_min=None, clip_max=None, y=None):
    """
    Computes symbolic TF tensor for the adversarial samples. This must
    be evaluate with a session.run call.
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :param eps: the epsilon (input variation parameter)
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :param y: the output placeholder. Use None (the default) to avoid the
            label leaking effect.
    :return: a tensor for the adversarial example
    """

    # Compute loss
    if y is None:
        # In this case, use model predictions as ground truth
        y = tf.to_float(
            tf.equal(predictions,
                     tf.reduce_max(predictions, 1, keep_dims=True)))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    logits, = predictions.op.inputs
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    )

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


def fast_gradient_sign_method(sess, model, X, Y, eps,
                              clip_min=None, clip_max=None):
    # Define TF placeholders for the input and output
    x = tf.placeholder(tf.float32, shape=(None,) + X.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,) + Y.shape[1:])
    adv_x = fgsm(
        x, model(x), eps=eps,
        clip_min=clip_min,
        clip_max=clip_max, y=y
    )
    X_adv, = batch_eval(
        sess, [x, y], [adv_x],
        [X, Y], args={'batch_size': 512}
    )

    return X_adv


def basic_iterative_method(sess, model, X, Y, eps, eps_iter, n_iter=50,
                           clip_min=None, clip_max=None):
    # Define TF placeholders for the input and output
    x = tf.placeholder(tf.float32, shape=(None,)+X.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,)+Y.shape[1:])
    # results will hold the adversarial inputs at each iteration of BIM;
    # thus it will have shape (n_iter, n_samples, n_rows, n_cols, n_channels)
    results = np.zeros((n_iter, X.shape[0],) + X.shape[1:])
    # Initialize adversarial samples as the original samples, set upper and
    # lower bounds
    X_adv = X
    X_min = X_adv - eps
    X_max = X_adv + eps
    # TODO: explain its, results
    print('Running BIM iterations...')
    its = {}
    out = set()
    for i in tqdm(range(n_iter)):
        adv_x = fgsm(
            x, model(x), eps=eps_iter,
            clip_min=clip_min, clip_max=clip_max, y=y
        )
        X_adv, = batch_eval(
            sess, [x, y], [adv_x],
            [X_adv, Y], args={'batch_size': 512}
        )
        X_adv = np.maximum(np.minimum(X_adv, X_max), X_min)
        results[i] = X_adv
        # check misclassifieds
        predictions = model.predict_classes(X_adv, batch_size=512, verbose=0)
        misclassifieds = np.where(predictions != Y.argmax(axis=1))[0]
        for elt in misclassifieds:
            if not elt in out:
                its[elt] = i
                out.add(elt)

    return its, results
