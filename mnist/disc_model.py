# Implementation of ResNet modified from
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

import tensorflow as tf
import numpy as np
from discriminator import pix2pixDiscriminator as Discriminator

NUM_CLASSES = 10


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
        self.bounds = (0, 1)
        self.attack_params = {
            'eps': args.eps,
            'step_size': args.step_size,
            'num_steps': args.num_steps,
            'random_start': args.random_start,
            'bounds': self.bounds,
        }

        self.f_dim = args.f_dim
        self.drop = 0 if not args.dropout else args.dropout_rate
        self.patch = args.patch

        self._build_model()

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        is_train = True if self.mode == 'train' else False

        self.discriminator = Discriminator(self.patch, is_train)

        self.x_input = tf.placeholder(
            tf.float32,
            shape=[None, 32, 32, 3])

        self.x_input_alg = tf.placeholder(
            tf.float32,
            shape=[None, 32, 32, 3]
        )
        self.x_input_alg_li = [tf.placeholder(
            tf.float32,
            shape=[None, 32, 32, 3]
        ) for _ in range(NUM_CLASSES)]

        self.y_input = tf.placeholder(tf.int64, shape=None)
        self.y_fake_input = tf.placeholder(tf.int64, shape=None)
        self.mask_input = tf.placeholder(tf.int64, shape=None)
        
        real = tf.ones_like(self.y_input)
        fake = tf.zeros_like(self.y_input)
        half = (real + fake) / 2

        # basic inference
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self.alg_noise = (self.x_input_alg-self.x_input)/self.delta

            self.d_out_single = self.discriminator(self.x_input, self.alg_noise)

            self.d_mean_out_single = tf.reduce_mean(tf.layers.flatten(self.d_out_single), axis=1)


            self.d_decisions_single = tf.where(self.d_mean_out_single >= 0.5, real, fake)

            self.d_loss_single = tf.reduce_mean(tf.losses.mean_squared_error(self.d_out_single, self.y_input))

        # inference & train procedure
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self.alg_noise_li = [(x_input_alg-self.x_input)/self.delta for x_input_alg in self.x_input_alg_li]

            self.d_out_li = [self.discriminator(self.x_input, alg_noise) for alg_noise in self.alg_noise_li]

            self.d_mean_out_li = [tf.reduce_mean(tf.layers.flatten(d_out), axis=1, keepdims=True) for d_out in self.d_out_li]

            self.d_out_full = tf.concat(self.d_mean_out_li, axis=1)

            # inference
            self.predictions = tf.argmax(self.d_out_full, axis=1)

            self.correct_predictions = tf.equal(self.predictions, self.y_input)

            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.int32))
            self.num_correct = tf.reduce_sum(tf.cast(self.correct_predictions, tf.int32))

            # train
            self.y_filtered = tf.where(tf.cast(self.mask_input, tf.float64) >= half,
                                       self.y_input, self.y_fake_input)

            self.alg_noise_stack = tf.stack(self.alg_noise_li)
            self.alg_noise_filtered = tf.gather_nd(self.alg_noise_li, self.y_filtered)

            self.d_out_train = self.discriminator(self.x_input, self.alg_noise_filtered)

            self.d_loss_train = tf.reduce_mean(tf.losses.mean_squared_error(self.d_out_train, self.mask_input))

            self.d_mean_out_train = tf.reduce_mean(tf.layers.flatten(self.d_out_train), axis=1)

            self.d_decisions_train = tf.where(tf.cast(self.d_mean_out_single, tf.float64) >= half,
                                              real, fake)

            self.num_correct_train_real = tf.reduce_sum(self.d_decisions_train * self.mask_input)
            self.num_correct_train_fake = tf.reduce_sum((1-self.d_decisions_train) * (1-self.mask_input))

            self.accuracy_train_real = self.num_correct_train_real / tf.reduce_sum(self.mask_input)
            self.accuracy_train_fake = self.num_correct_train_fake / tf.reduce_sum(1-self.mask_input)

            self.num_correct_train = self.num_correct_train_real + self.num_correct_train_fake
            self.accuracy_train = tf.reduce_mean(self.d_decisions_train * self.mask_input)


    def generate_fakes(self, y):
        # generate always not-equal random labels
        fake = np.copy(y)

        while np.sum(fake == y) > 0:
            same_mask = np.where(fake == y)
            new_fake = np.random.randint(NUM_CLASSES, size=np.size(y))

            fake = np.where(same_mask, new_fake, fake)

        mask = np.random.randint(2, size=np.size(y))

        return fake, mask

