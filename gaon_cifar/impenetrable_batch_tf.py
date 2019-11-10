"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from PIL import Image

import tensorflow as tf
import numpy as np
from pgd_attack import LinfPGDAttack
# import time

# import os


def PGD(x, y, model_fn, attack_params, first_order=True):
    eps = attack_params['eps']
    step_size = attack_params['step_size']
    num_steps = attack_params['num_steps']
    bounds = attack_params['bounds']
    random_start = attack_params['random_start']

    lower_bound = tf.maximum(x-eps, bounds[0])
    upper_bound = tf.minimum(x+eps, bounds[1])

    if random_start:
        x += tf.random_uniform(tf.shape(x), -eps, eps)
        x = tf.clip_by_value(x, lower_bound, upper_bound)

    for i in range(num_steps):
        logits = model_fn(x)
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=y)
        grad = tf.gradients(y_xent, x)[0]
        if first_order:
            x += step_size * tf.stop_gradient(tf.sign(grad))
        else:
            x += step_size * tf.sign(grad)
        x = tf.clip_by_value(x, lower_bound, upper_bound)

    return x

class Impenetrable(object):
    def __init__(self, model, args):
        self.model = model
        self.loss_func = args.loss_func
        self.imp_num_steps = args.imp_num_steps
        self.imp_delta = args.imp_delta
        self.pgd_eps = args.pgd_eps
        self.pgd_num_steps = args.pgd_num_steps
        self.pgd_step_size = args.pgd_step_size
        self.pgd_restarts = args.pgd_restarts
        self.pgd_random_start = args.pgd_random_start or (self.pgd_restarts > 1)
        self.imp_step_size = args.imp_step_size

        self.bounds = (0, 255)

        self.attack_params = {
            'eps': self.pgd_eps,
            'step_size': self.pgd_step_size,
            'num_steps': self.pgd_num_steps,
            'random_start': self.pgd_random_start,
            'bounds': self.bounds,
        }

        self._build_model()

    def _build_model(self):
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self.x_input = tf.placeholder(
                tf.float32,
                shape=[None, 32, 32, 3])
            self.y_input = tf.placeholder(tf.int64, shape=None)

            self.rms_v = tf.placeholder(
                tf.float32,
                shape=[None, 32, 32, 3])
            self.grads = tf.placeholder(
                tf.float32,
                shape=[None, 32, 32, 3])

            # pgd loss calculation
            self.x_pgd = PGD(self.x_input, self.y_input, self.model.fprop, self.attack_params)
            logits = self.model.fprop(self.x_pgd)
            y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.y_input
            )
            predictions = tf.argmax(logits, 1)
            correct_prediction = tf.equal(self.y_input, predictions)

            self.xent = tf.reduce_mean(y_xent)
            self.corrects = correct_prediction
            self.grad = tf.gradients(y_xent, self.x_input)[0]

            # rmsprop step caculation
            rho = 0.9
            self.new_rms_v = rho * self.rms_v + (1-rho) * tf.pow(self.grads, 2)
            self.new_x = self.x_input - self.imp_step_size * self.grads / (tf.sqrt(self.new_rms_v + 1e-7))
            self.new_x = tf.clip_by_value(self.new_x)
