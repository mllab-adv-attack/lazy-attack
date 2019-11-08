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

        self.discriminator = Discriminator(self.patch, is_train, self.f_dim)

        self.x_input = tf.placeholder(
            tf.float32,
            shape=[None, 28, 28, 1])

        self.x_input_alg = tf.placeholder(
            tf.float32,
            shape=[None, 28, 28, 1]
        )

        self.mask_input = tf.placeholder(tf.float32, shape=None)
        self.y_input = tf.placeholder(tf.int64, shape=None)

        real = tf.ones_like(self.mask_input)
        fake = tf.zeros_like(self.mask_input)

        # basic inference
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self.alg_noise = (self.x_input_alg-self.x_input)/self.delta

            self.d_out = self.discriminator(self.x_input, self.alg_noise)
            #print(self.d_out.get_shape().as_list())

            self.d_mean_out = tf.reduce_mean(tf.layers.flatten(self.d_out), axis=1)
            #print(self.d_mean_out.get_shape().as_list())
            
            self.d_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.d_mean_out, self.mask_input))
            self.c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.mask_input, logits=self.d_mean_out))

            self.d_decisions = tf.where(self.d_mean_out >= 0.5, real, fake)
            #print(self.d_decisions.get_shape().as_list())
            self.c_predictions = tf.nn.sigmoid(self.d_mean_out)
            #print(self.c_predictions.get_shape().as_list())
            self.c_decisions = tf.where(self.c_predictions >= 0.5, real, fake)
            #print(self.c_decisions.get_shape().as_list())

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

        # sanity check
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            # eval original image (clean)
            orig_pre_softmax = self.model(self.x_input)

            orig_predictions = tf.argmax(orig_pre_softmax, 1)
            self.orig_correct_prediction = tf.equal(orig_predictions, self.y_input)
            self.orig_accuracy = tf.reduce_mean(
                tf.cast(self.orig_correct_prediction, tf.float32))

            orig_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=orig_pre_softmax, labels=self.y_input)
            self.orig_mean_xent = tf.reduce_mean(orig_y_xent)

            # eval original image (PGD)
            x_input_pgd = PGD(self.x_input, self.y_input, self.model, self.attack_params)
            
            orig_pgd_pre_softmax = self.model(x_input_pgd)

            orig_pgd_predictions = tf.argmax(orig_pgd_pre_softmax, 1)
            self.orig_pgd_correct_prediction = tf.equal(orig_pgd_predictions, self.y_input)
            self.orig_pgd_accuracy = tf.reduce_mean(
                tf.cast(self.orig_pgd_correct_prediction, tf.float32))

            orig_pgd_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=orig_pgd_pre_softmax, labels=self.y_input)
            self.orig_pgd_mean_xent = tf.reduce_mean(orig_pgd_y_xent)
            
            # eval alg image (clean)
            alg_pre_softmax = self.model(self.x_input_alg)

            alg_predictions = tf.argmax(alg_pre_softmax, 1)
            self.alg_correct_prediction = tf.equal(alg_predictions, self.y_input)
            self.alg_accuracy = tf.reduce_mean(
                tf.cast(self.alg_correct_prediction, tf.float32))

            alg_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=alg_pre_softmax, labels=self.y_input)
            self.alg_mean_xent = tf.reduce_mean(alg_y_xent)

            # eval alg image (PGD)
            x_input_pgd = PGD(self.x_input_alg, self.y_input, self.model, self.attack_params)
            
            alg_pgd_pre_softmax = self.model(x_input_pgd)

            alg_pgd_predictions = tf.argmax(alg_pgd_pre_softmax, 1)
            self.alg_pgd_correct_prediction = tf.equal(alg_pgd_predictions, self.y_input)
            self.alg_pgd_accuracy = tf.reduce_mean(
                tf.cast(self.alg_pgd_correct_prediction, tf.float32))

            alg_pgd_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=alg_pgd_pre_softmax, labels=self.y_input)
            self.alg_pgd_mean_xent = tf.reduce_mean(alg_pgd_y_xent)

    def generate_fakes(self, y, x_input_alg_li):
        # generate always not-equal random labels
        fake = np.copy(y)

        while np.sum(fake == y) > 0:
            same_mask = fake == y
            new_fake = np.random.randint(NUM_CLASSES, size=np.size(y))

            fake = np.where(same_mask, new_fake, fake)

        assert np.sum(fake == y) < 1e-6

        mask_index = np.arange(np.size(y))
        np.random.shuffle(mask_index)
        mask_index_cut = np.split(mask_index, 2)[0]
        mask = np.array([(1 if i in mask_index_cut else 0) for i in range(np.size(y))])

        half_fake = np.where(mask > 0.5, y, fake)

        x_input_alg_half_fake = np.copy(x_input_alg_li[0])
        for i in range(np.size(y)):
            x_input_alg_half_fake[i] = x_input_alg_li[half_fake[i]][i]

        return half_fake, mask, x_input_alg_half_fake

    def infer(self, sess, x_input, x_input_alg_li, return_images=False):
        d_outs = []
        for i in range(NUM_CLASSES):
            feed_dict = {self.x_input: x_input,
                         self.x_input_alg: x_input_alg_li[i]}
            d_out_batch = sess.run(self.d_mean_out,
                                   feed_dict=feed_dict)
            d_outs.append(d_out_batch)

        d_outs = np.stack(d_outs, axis=-1)
        d_preds = np.argmax(d_outs, axis=1)

        if return_images:
            infer_alg = np.copy(x_input_alg_li[0])
            for i in range(np.size(d_preds)):
                infer_alg[i] = x_input_alg_li[d_preds[i]][i]

            return d_preds, infer_alg

        return d_preds
