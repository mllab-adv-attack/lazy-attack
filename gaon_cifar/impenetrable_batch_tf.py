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
        self.adam = args.imp_adam
        self.rms = args.imp_rms

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
        self.x_input = tf.placeholder(
            tf.float32,
            shape=[None, 32, 32, 3])
        self.y_input = tf.placeholder(tf.int64, shape=None)

        self.x = tf.get_variable(name='safe_spot', shape=[None, 32, 32, 3], dtype=tf.float32)
        self.correct_mask = tf.get_variable(name='correct mask', shape=[None], dtype=tf.bool)

        self.init_op = tf.assign(self.x, self.x_input)

        lower_bound = tf.maximum(self.x_input-self.imp_delta, self.bounds[0])
        upper_bound = tf.minimum(self.x_input+self.imp_delta, self.bounds[1])

        x_clipped = tf.clip_by_value(self.x, lower_bound, upper_bound)
        self.clip_op = tf.assign(self.x, x_clipped)

        xents = 0

        corrects = tf.ones_like(self.y_input, dtype=bool)

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            for _ in range(self.pgd_random_start):
                self.x_pgd = PGD(self.x, self.y_input, self.model.fprop, self.attack_params)
                logits = self.model.fprop(self.x_pgd)
                y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=self.y_input
                )
                predictions = tf.argmax(logits, 1)
                correct_prediction = tf.equal(self.y_input, predictions)
                corrects *= correct_prediction

                xents += y_xent

            self.mean_xent = xents / self.pgd_random_start
            self.corrects = corrects
            self.mean_corrects = tf.reduce_mean(tf.cast(self.corrects, dtype=tf.float32))

