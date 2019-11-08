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
        self.c_loss = args.c_loss

        self._build_model()

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        is_train = True if self.mode == 'train' else False

        self.discriminator = Discriminator(self.patch, is_train)

        self.x_input = tf.placeholder(
            tf.float32,
            shape=[None, 28, 28, 1])

        self.x_input_alg = tf.placeholder(
            tf.float32,
            shape=[None, 28, 28, 1]
        )

        self.mask_input = tf.placeholder(tf.float32, shape=None)

        real = tf.ones_like(self.mask_input)
        fake = tf.zeros_like(self.mask_input)

        # basic inference
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self.alg_noise = (self.x_input_alg-self.x_input)/self.delta

            self.d_out = self.discriminator(self.x_input, self.alg_noise)

            self.d_mean_out = tf.reduce_mean(tf.layers.flatten(self.d_out), axis=1)
            
            self.d_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.d_out, self.mask_input))
            self.c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.mask_input, logits=self.d_mean_out))

            self.d_decisions = tf.where(self.d_mean_out >= 0.5, real, fake)
            self.c_predictions = tf.nn.sigmoid(self.d_mean_out)
            self.c_decisions = tf.where(self.c_predictions >= 0.5, real, fake)

            # d accuracy
            self.d_num_correct_real = tf.reduce_sum(self.d_decisions * self.mask_input)
            self.d_num_correct_fake = tf.reduce_sum((1-self.d_decisions) * (1-self.mask_input))

            self.d_accuracy_real = self.d_num_correct_real / tf.reduce_sum(self.mask_input)
            self.d_accuracy_fake = self.d_num_correct_fake / tf.reduce_sum(1-self.mask_input)

            self.d_num_correct = self.d_num_correct_real + self.d_num_correct_fake
            self.d_accuracy = tf.reduce_mean(self.d_decisions * self.mask_input +
                                           (1-self.d_decisions) * (1-self.mask_input))
            
            # c accuracy
            self.c_num_correct_real = tf.reduce_sum(self.c_decisions * self.mask_input)
            self.c_num_correct_fake = tf.reduce_sum((1-self.c_decisions) * (1-self.mask_input))

            self.c_accuracy_real = self.c_num_correct_real / tf.reduce_sum(self.mask_input)
            self.c_accuracy_fake = self.c_num_correct_fake / tf.reduce_sum(1-self.mask_input)

            self.c_num_correct = self.c_num_correct_real + self.c_num_correct_fake
            self.c_accuracy = tf.reduce_mean(self.c_decisions * self.mask_input +
                                           (1-self.c_decisions) * (1-self.mask_input))

    def generate_fakes(self, y, x_input_alg_li):
        # generate always not-equal random labels
        fake = np.copy(y)

        while np.sum(fake == y) > 0:
            same_mask = fake == y
            new_fake = np.random.randint(NUM_CLASSES, size=np.size(y))

            fake = np.where(same_mask, new_fake, fake)

        mask = np.random.randint(2, size=np.size(y))

        half_fake = np.where(mask > 0, y, fake)

        x_input_alg_half_fake = np.copy(x_input_alg_li[0])
        for i in range(len(y)):
            x_input_alg_half_fake[i] = x_input_alg_li[half_fake[i]][i]

        return half_fake, mask, x_input_alg_half_fake

    def infer(self, sess, x_input, x_input_alg_li):
        d_outs = []
        for i in range(NUM_CLASSES):
            feed_dict = {self.x_input: x_input,
                         self.x_input_alg: x_input_alg_li[i]}
            d_out_batch = sess.run(self.d_mean_out,
                                   feed_dict=feed_dict)
            d_outs.append(d_out_batch)

        d_outs = np.stack(d_outs, axis=-1)
        d_preds = np.argmax(d_outs, axis=1)

        return d_preds
