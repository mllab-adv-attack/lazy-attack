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

        self.mask_input = tf.placeholder(tf.float32, shape=[None])
        self.y_input = tf.placeholder(tf.int64, shape=None)

        real = tf.ones_like(self.mask_input)
        fake = tf.zeros_like(self.mask_input)

        # basic inference
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):

            # scale-up input noise
            self.alg_noise = (self.x_input_alg-self.x_input)/self.delta

            # Discriminator loss
            self.d_out = self.discriminator(self.x_input, self.alg_noise)
            self.d_mean_out = tf.reduce_mean(tf.layers.flatten(self.d_out), axis=1)
            self.d_decisions = tf.where(self.d_mean_out >= 0.5, real, fake)

            self.d_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.d_mean_out, self.mask_input))

            # Classification loss
            self.c_predictions = tf.nn.sigmoid(self.d_mean_out)
            self.c_decisions = tf.where(self.c_predictions >= 0.5, real, fake)

            self.c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.mask_input,
                                                                                 logits=self.d_mean_out))

            # Discriminator accuracy
            self.d_num_correct_real, self.d_num_correct_fake, \
                self.d_accuracy_real, self.d_accuracy_fake, \
                self.d_num_correct, self.d_accuracy = \
                self._evaluate(self.d_decisions, self.mask_input)
            
            # Classification accuracy
            self.c_num_correct_real, self.c_num_correct_fake, \
                self.c_accuracy_real, self.c_accuracy_fake, \
                self.c_num_correct, self.c_accuracy = \
                self._evaluate(self.c_decisions, self.mask_input)

        # sanity check
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            # eval original image (clean)
            self.orig_correct_prediction, self.orig_accuracy, self.orig_mean_xent = \
                self._sanity_check(self.x_input, self.y_input)

            # eval original image (PGD)
            x_input_pgd = PGD(self.x_input, self.y_input, self.model, self.attack_params)

            self.orig_pgd_correct_prediction, self.orig_pgd_accuracy, self.orig_pgd_mean_xent = \
                self._sanity_check(x_input_pgd, self.y_input)
            
            # eval alg image (clean)
            self.alg_correct_prediction, self.alg_accuracy, self.alg_mean_xent = \
                self._sanity_check(self.x_input_alg, self.y_input)

            # eval alg image (PGD)
            x_alg_pgd = PGD(self.x_input_alg, self.y_input, self.model, self.attack_params)
            
            self.alg_pgd_correct_prediction, self.alg_pgd_accuracy, self.alg_pgd_mean_xent = \
                self._sanity_check(x_alg_pgd, self.y_input)

    @staticmethod
    def _evaluate(decisions, mask):
        num_correct_real = tf.reduce_sum(decisions * mask)
        num_correct_fake = tf.reduce_sum((1-decisions) * (1-mask))

        accuracy_real = num_correct_real / tf.reduce_sum(mask)
        accuracy_fake = num_correct_fake / tf.reduce_sum(1-mask)

        num_correct = num_correct_real + num_correct_fake
        accuracy = tf.reduce_mean(decisions * mask + (1-decisions) * (1-mask))

        return num_correct_real, num_correct_fake, accuracy_real, accuracy_fake, num_correct, accuracy

    def _sanity_check(self, x, y):
        pre_softmax = self.model(x)

        predictions = tf.argmax(pre_softmax, 1)
        correct_prediction = tf.equal(predictions, y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pre_softmax, labels=y
        )
        mean_xent = tf.reduce_mean(y_xent)

        return correct_prediction, accuracy, mean_xent

    @staticmethod
    def generate_fakes(y, x_input_alg_li):
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


class ModelMultiClass(object):

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
        self.multi_pass = args.multi_pass

        self._build_model()

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        is_train = True if self.mode == 'train' else False

        if self.multi_pass:
            print('building multi pass model!')
            self.discriminator = Discriminator(self.patch, is_train, self.f_dim, multi_class=False)
        else:
            print('building single pass model!')
            self.discriminator = Discriminator(self.patch, is_train, self.f_dim, multi_class=True)

        self.x_input = tf.placeholder(
            tf.float32,
            shape=[None, 28, 28, 1])

        self.x_input_alg = tf.placeholder(
            tf.float32,
            shape=[None, 28, 28, 1 * NUM_CLASSES]
        )

        self.y_input = tf.placeholder(tf.int64, shape=None)

        # basic inference
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            # scale-up input noise
            self.alg_noise = (self.x_input_alg-self.x_input)/self.delta

            if self.multi_pass:
                splits = tf.split(self.alg_noise, NUM_CLASSES, axis=-1)

                self.d_outs = [self.discriminator(self.x_input, noise_split) for noise_split in splits]
                self.d_out = tf.concat(self.d_outs, axis=-1)

            else:

                self.d_out = self.discriminator(self.x_input, self.alg_noise)
            #print(self.d_out.get_shape().as_list())

            self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_input, logits=self.d_out)
            self.xent = tf.reduce_mean(self.y_xent)

            self.predictions = tf.argmax(self.d_out, 1)
            self.correct_prediction = tf.equal(self.predictions, self.y_input)
            self.num_correct = tf.reduce_sum(
                tf.cast(self.correct_prediction, tf.int32))
            self.accuracy = tf.reduce_mean(
                tf.cast(self.correct_prediction, tf.float32))

        # sanity check
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            # eval original image (clean)
            self.orig_correct_prediction, self.orig_accuracy, self.orig_mean_xent = \
                self._sanity_check(self.x_input, self.y_input)

            # eval original image (PGD)
            x_input_pgd = PGD(self.x_input, self.y_input, self.model, self.attack_params)

            self.orig_pgd_correct_prediction, self.orig_pgd_accuracy, self.orig_pgd_mean_xent = \
                self._sanity_check(x_input_pgd, self.y_input)

    def _sanity_check(self, x, y):
        pre_softmax = self.model(x)

        predictions = tf.argmax(pre_softmax, 1)
        correct_prediction = tf.equal(predictions, y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pre_softmax, labels=y
        )
        mean_xent = tf.reduce_mean(y_xent)

        return correct_prediction, accuracy, mean_xent

    def infer(self, sess, x_input, x_input_alg_li, return_images=False):

        x_input_alg_full = np.concatenate(x_input_alg_li, axis=-1)
        feed_dict = {self.x_input: x_input,
                     self.x_input_alg: x_input_alg_full}
        preds = sess.run(self.predictions,
                         feed_dict=feed_dict)

        if return_images:
            infer_alg = np.copy(x_input_alg_li[0])
            for i in range(np.size(preds)):
                infer_alg[i] = x_input_alg_li[preds[i]][i]

            return preds, infer_alg

        return preds
