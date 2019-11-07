# Implementation of ResNet modified from
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

import tensorflow as tf
from discriminator import Discriminator
from generator import generator_v2 as generator


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

        self.noise_only = args.noise_only
        self.use_d = args.use_d
        self.unet = args.unet
        self.use_advG = args.use_advG
        self.f_dim = args.f_dim
        self.drop = 0 if not args.dropout else args.dropout_rate
        self.lp_loss = args.lp_loss

        self._build_model()

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        is_train = True if self.mode == 'train' else False

        with tf.variable_scope('infer_input', reuse=tf.AUTO_REUSE):
            self.x_input = tf.placeholder(
                tf.float32,
                shape=[None, 28, 28, 1])
            self.y_input = tf.placeholder(tf.int64, shape=None)
            
            self.x_input_alg = tf.placeholder(
                tf.float32,
                shape=[None, 28, 28, 1]
            )

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self.def_generator = tf.make_template('generator', generator, f_dim=self.f_dim, c_dim=1, drop=self.drop,
                                                  unet=self.unet, is_training=is_train)
            self.x_safe = self.x_input + self.delta * self.def_generator(self.x_input)
            self.x_safe = tf.clip_by_value(self.x_safe, self.bounds[0], self.bounds[1])

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            if self.use_advG:
                # use adv generator as attacker (PGD only when evaluation)
                self.adv_generator = tf.make_template('adv_generator', generator, f_dim=self.f_dim, c_dim=1,
                                                      unet=self.unet,
                                                      drop=self.drop, is_training=is_train)
                self.x_safe_adv = self.x_safe + self.delta * self.adv_generator(self.x_safe)
                self.x_safe_adv = tf.clip_by_value(self.x_safe_adv, self.bounds[0], self.bounds[1])
                self.x_safe_pgd = PGD(self.x_safe, self.y_input, self.model, self.attack_params)
            else:
                # use PGD as attacker
                self.x_safe_adv = PGD(self.x_safe, self.y_input, self.model, self.attack_params)
                self.x_safe_pgd = self.x_safe_adv

            diff = self.x_safe_adv - self.x_safe
            diff = tf.stop_gradient(diff)
            x_safe_adv_fo = self.x_safe + diff

            # eval original image
            orig_pre_softmax = self.model(self.x_input)

            orig_predictions = tf.argmax(orig_pre_softmax, 1)
            orig_correct_prediction = tf.equal(orig_predictions, self.y_input)
            self.orig_accuracy = tf.reduce_mean(
                tf.cast(orig_correct_prediction, tf.float32))

            orig_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=orig_pre_softmax, labels=self.y_input)
            self.orig_mean_xent = tf.reduce_mean(orig_y_xent)
 
            # eval safe image
            self.safe_pre_softmax = self.model(self.x_safe)

            safe_predictions = tf.argmax(self.safe_pre_softmax, 1)
            safe_correct_prediction = tf.equal(safe_predictions, self.y_input)
            self.safe_accuracy = tf.reduce_mean(
                tf.cast(safe_correct_prediction, tf.float32))

            safe_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.safe_pre_softmax, labels=self.y_input)
            self.safe_mean_xent = tf.reduce_mean(safe_y_xent)

            # eval attacked safe image
            if self.use_advG:
                safe_adv_pre_softmax = self.model(self.x_safe_adv)
            else:
                # use first order for PGD attack
                safe_adv_pre_softmax = self.model(x_safe_adv_fo)

            safe_adv_predictions = tf.argmax(safe_adv_pre_softmax, 1)
            safe_adv_correct_prediction = tf.equal(safe_adv_predictions, self.y_input)
            self.safe_adv_accuracy = tf.reduce_mean(
                tf.cast(safe_adv_correct_prediction, tf.float32))

            safe_adv_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=safe_adv_pre_softmax, labels=self.y_input)
            self.safe_adv_mean_xent = tf.reduce_mean(safe_adv_y_xent)

            # eval PGD attacked safe image
            if self.use_advG:
                safe_pgd_pre_softmax = self.model(self.x_safe_pgd)

                safe_pgd_predictions = tf.argmax(safe_pgd_pre_softmax, 1)
                safe_pgd_correct_prediction = tf.equal(safe_pgd_predictions, self.y_input)
                self.safe_pgd_accuracy = tf.reduce_mean(
                    tf.cast(safe_pgd_correct_prediction, tf.float32))

                safe_pgd_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=safe_pgd_pre_softmax, labels=self.y_input)
                self.safe_pgd_mean_xent = tf.reduce_mean(safe_pgd_y_xent)
            else:
                self.safe_pgd_accuracy = self.safe_adv_accuracy
                self.safe_pgd_mean_xent = self.safe_adv_mean_xent

            # eval alg image
            self.alg_pre_softmax = self.model(self.x_input_alg)

            alg_predictions = tf.argmax(self.alg_pre_softmax, 1)
            alg_correct_prediction = tf.equal(alg_predictions, self.y_input)
            self.alg_accuracy = tf.reduce_mean(
                tf.cast(alg_correct_prediction, tf.float32))

            alg_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.alg_pre_softmax, labels=self.y_input)
            self.alg_mean_xent = tf.reduce_mean(alg_y_xent)

        if self.use_d:
            self.discriminator = Discriminator()

            if self.noise_only:
                self.d_alg_out = self.discriminator((self.x_input_alg-self.x_input)/self.delta)
                self.d_safe_out = self.discriminator((self.x_safe-self.x_input)/self.delta)
            else:
                self.d_alg_out = self.discriminator(self.x_input_alg)
                self.d_safe_out = self.discriminator(self.x_safe)

            real = tf.ones_like(self.d_alg_out)
            fake = tf.zeros_like(self.d_alg_out)

            self.d_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.d_alg_out, real) +
                                         tf.losses.mean_squared_error(self.d_safe_out, fake))
            self.g_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.d_safe_out, real))

