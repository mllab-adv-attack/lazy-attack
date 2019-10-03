# Implementation of ResNet modified from
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

import tensorflow as tf
from discriminator import Discriminator
from generator import generator


def PGD(x, y, model_fn, attack_params):
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
        x += step_size * tf.sign(grad)
        x = tf.clip_by_value(x, lower_bound, upper_bound)

    return x
        

class Model(object):

    def __init__(self, mode, model, args):

        self.mode = mode
        self.model = model
        self.delta = args.delta
        self.bounds = (0, 255)
        self.attack_params = {
            'eps': args.eps,
            'step_size': args.step_size,
            'num_steps': args.num_steps,
            'random_start': args.random_start,
            'bounds': self.bounds,
        }

        self.use_d = args.use_d
        self.patch = args.patch

        self._build_model()

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        is_train = True if self.mode == 'train' else False

        with tf.variable_scope('infer_input', reuse=tf.AUTO_REUSE):
            self.x_input = tf.placeholder(
                tf.float32,
                shape=[None, 32, 32, 3])
            self.y_input = tf.placeholder(tf.int64, shape=None)

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            safe_generator = tf.make_template('generator', generator, f_dim=64, output_size=32, c_dim=3, is_training=is_train)
            self.x_safe = self.x_input + self.delta * safe_generator(self.x_input)
            self.x_safe = tf.clip_by_value(self.x_safe, self.bounds[0], self.bounds[1])

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self.x_safe_pgd = PGD(self.x_safe, self.y_input, self.model.fprop, self.attack_params)
            diff = self.x_safe_pgd - self.x_safe
            diff = tf.stop_gradient(diff)
            x_safe_pgd_fo = self.x_safe + diff

            # eval original image
            orig_pre_softmax = self.model.fprop(self.x_input)

            orig_predictions = tf.argmax(orig_pre_softmax, 1)
            orig_correct_prediction = tf.equal(orig_predictions, self.y_input)
            self.orig_accuracy = tf.reduce_mean(
                tf.cast(orig_correct_prediction, tf.float32))

            orig_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=orig_pre_softmax, labels=self.y_input)
            self.orig_mean_xent = tf.reduce_mean(orig_y_xent)
 
            # eval safe image
            safe_pre_softmax = self.model.fprop(self.x_safe)

            safe_predictions = tf.argmax(safe_pre_softmax, 1)
            safe_correct_prediction = tf.equal(safe_predictions, self.y_input)
            self.safe_accuracy = tf.reduce_mean(
                tf.cast(safe_correct_prediction, tf.float32))

            safe_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=safe_pre_softmax, labels=self.y_input)
            self.safe_mean_xent = tf.reduce_mean(safe_y_xent)

            # eval attacked safe image
            safe_pgd_pre_softmax = self.model.fprop(x_safe_pgd_fo)

            safe_pgd_predictions = tf.argmax(safe_pgd_pre_softmax, 1)
            safe_pgd_correct_prediction = tf.equal(safe_pgd_predictions, self.y_input)
            self.safe_pgd_accuracy = tf.reduce_mean(
                tf.cast(safe_pgd_correct_prediction, tf.float32))

            safe_pgd_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=safe_pgd_pre_softmax, labels=self.y_input)
            self.safe_pgd_mean_xent = tf.reduce_mean(safe_pgd_y_xent)

        if self.use_d:
            self.discriminator = Discriminator(self.patch, is_train)

            self.x_input_alg = tf.placeholder(
                tf.float32,
                shape=[None, 32, 32, 3]
            )

            d_alg_out = self.discriminator(self.x_input_alg)
            d_safe_out = self.discriminator(self.x_safe)

            self.d_loss = tf.reduce_mean(tf.nn.l2_loss(d_alg_out-1) + tf.nn.l2_loss(d_safe_out))
            self.g_loss = tf.reduce_mean(tf.nn.l2_loss(d_safe_out-1))
