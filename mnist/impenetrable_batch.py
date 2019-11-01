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
#import time

# import os


class Impenetrable(object):
    def __init__(self, model, args):
        self.model = model
        self.loss_func = args.loss_func
        self.imp_random_start = args.imp_random_start
        self.imp_gray_start = args.imp_gray_start
        self.imp_num_steps = args.imp_num_steps
        self.imp_delta = args.imp_delta
        self.pgd_eps = args.pgd_eps
        self.pgd_num_steps = args.pgd_num_steps
        self.pgd_step_size = args.pgd_step_size
        # self.pgd_step_size = self.pgd_eps/4.0
        self.pgd_restarts = args.pgd_restarts
        self.pgd_random_start = args.pgd_random_start or (self.pgd_restarts > 1)
        self.imp_step_size = args.imp_step_size
        self.val_step_per = args.val_step_per
        self.val_restarts = args.val_restarts
        self.val_num_steps = args.val_num_steps
        self.val_step_size = args.val_num_steps
        self.val_eps = args.val_eps
        self.adam = args.imp_adam
        self.rms = args.imp_rms
        self.adagrad = args.imp_adagrad
        self.rep = args.imp_rep
        self.pp = args.imp_pp
        assert self.pp <= 0
        self.imp_no_sign = args.imp_no_sign

        self.pgd = LinfPGDAttack(self.model,
                                 self.pgd_eps,
                                 self.pgd_num_steps,
                                 self.pgd_step_size,
                                 self.pgd_random_start,
                                 args.loss_func)

        if self.loss_func == 'xent':
            self.loss = self.model.xent
            self.loss_full = self.model.y_xent
        elif self.loss_func == 'cw':
            label_mask = tf.one_hot(self.model.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * self.model.pre_softmax, axis=1)
            wrong_logit = tf.reduce_max((1-label_mask) * self.model.pre_softmax - 1e4*label_mask, axis=1)
            self.loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss = self.model.xent

        self.grad = tf.gradients(self.loss, self.model.x_input)[0]
        self.softmax = self.model.softmax

    def fortify(self, x_orig, y, sess):

        suc_flag = False

        num_images = len(x_orig)

        # initialize image
        if self.imp_random_start > 0:
            x = x_orig + np.random.uniform(-self.imp_random_start, self.imp_random_start, x_orig.shape)
            x = np.clip(x, 0, 1)
        elif self.imp_gray_start:
            x = np.ones_like(x_orig).astype('float32') * 0.5
        else:
            x = np.copy(x_orig)

        orig_loss, orig_corr, orig_soft = sess.run([self.loss, self.model.num_correct, self.softmax],
                                                   feed_dict={self.model.x_input: x,
                                                              self.model.y_input: y})

        print("original accuracy: {:.2f}%".format(orig_corr/num_images*100))
        print("original loss: {:.20f}".format(orig_loss/num_images))

        # adam & rms parameters
        beta1 = 0.9
        beta2 = 0.999
        adam_m = np.zeros_like(x)
        adam_v = np.zeros_like(x)

        rho = 0.9
        rms_v = np.zeros_like(x)

        accum = 0.1

        # validation (original image)
        '''
        if self.val_step_per > 0 and orig_corr > 0:

            suc_flag = self.validation(x, y_hard)

        if suc_flag:
            return x
        '''

        step = 1
        while self.imp_num_steps <= 0 or step <= self.imp_num_steps:
            print("step:", step)

            # attack image
            grad_li = []
            adv_loss_li = []
            mask = np.array([True for _ in range(len(x))])

            for _ in range(self.pgd_restarts):
                x_adv_batch, x_adv_batch_full = self.pgd.perturb(x, y, sess)

                adv_loss, adv_loss_full, \
                    correct_predictions, grad = sess.run([self.loss, self.loss_full,
                                                          self.model.correct_prediction, self.grad],
                                                         feed_dict={self.model.x_input: x_adv_batch,
                                                                    self.model.y_input: y})

                mask *= correct_predictions
                if self.rep:
                    # use reptile (x_adv - x as gd direction)
                    grad = x_adv_batch - x

                grad_li.append(np.copy(grad))
                adv_loss_li.append(adv_loss)

            grad = np.mean(grad_li, axis=0)
            assert grad.shape == x.shape
            adv_loss = np.mean(adv_loss_li)

            # l1_dist = np.linalg.norm((x_adv - x).flatten(), 1) / x.size
            print("attack accuracy: {:.2f}%".format(np.mean(mask)*100))
            # print("l1 distance: {:.2f}".format(l1_dist))
            print("adv loss: {:.20f}".format(adv_loss/num_images))

            # restore image
            if self.adam:
                adam_lr = self.imp_step_size * np.sqrt(1-beta2**step)/(1-beta1**step)
                adam_m = beta1 * adam_m + (1-beta1) * grad
                adam_v = beta2 * adam_v + (1-beta2) * grad * grad
                x_res = x - adam_lr * adam_m / (np.sqrt(adam_v) + 1e-8)
            elif self.rms:
                rms_v = rho * rms_v + (1-rho) * (grad**2)
                x_res = x - self.imp_step_size * grad / (np.sqrt(rms_v + 1e-7))
            elif self.adagrad:
                accum += grad**2
                x_res = x - self.imp_step_size * grad / (np.sqrt(accum) + 1e-7)
            else:
                if self.imp_no_sign:
                    x_res = x - self.imp_step_size * grad
                else:
                    x_res = x - self.imp_step_size * np.sign(grad)

            x_res = np.clip(x_res, 0, 1)

            if self.imp_delta > 0:
                x_res = np.clip(x_res, x_orig-self.imp_delta, x_orig+self.imp_delta)

            res_loss, res_corr, res_soft = sess.run([self.loss, self.model.num_correct, self.softmax],
                                                    feed_dict={self.model.x_input: x_res,
                                                               self.model.y_input: y})

            l2_dist = np.linalg.norm((x_res - x_orig).flatten())
            print("restored accuracy: {:.2f}%".format(res_corr/num_images*100))
            print("l2 distance: {:.2f}".format(l2_dist))
            print("res loss: {:.20f}".format(res_loss/num_images))

            x = x_res

            # validation
            if self.val_step_per > 0 and step % self.val_step_per == 0:

                if np.sum(mask) == len(x):

                    suc_flag = self.validation(x, y, sess)

            print()

            # early stop
            if suc_flag:
                break

            step += 1

        return x, step

    def reset_pgd(self):
        self.pgd.epsilon = self.pgd_eps
        self.pgd.step_size = self.pgd_step_size
        self.pgd.num_steps = self.pgd_num_steps
        self.pgd.random_start = self.pgd_random_start

    def set_pgd(self, eps, step_size, num_steps, random_start):
        self.pgd.epsilon = eps
        self.pgd.step_size = step_size
        self.pgd.num_steps = num_steps
        self.pgd.random_start = random_start

    def set_pgd_val(self):
        self.set_pgd(self.val_eps, self.val_step_size, self.val_num_steps, True)

    def validation(self, x, y, sess):
        
        print()

        num_images = len(x)

        suc_flag = False

        self.set_pgd_val()

        val_iter = self.val_restarts
        val_total_corr = 0

        if val_iter >= 100:
            assert val_iter % 100 == 0

            x_val_batch = np.tile(x, (100, 1, 1, 1))
            # y_val_batch = np.tile(y, (100, 1))
            y_hard_val_batch = np.tile(y, 100)

            for i in range(val_iter//100):

                x_val, _ = self.pgd.perturb(x_val_batch, y_hard_val_batch, sess, rand=True)

                cur_corr = sess.run(self.model.num_correct,
                                    feed_dict={self.model.x_input: x_val,
                                               self.model.y_input: y})

                val_total_corr += cur_corr
        else:
            x_val_batch = np.tile(x, (val_iter, 1, 1, 1))
            # y_val_batch = np.tile(y, (100, 1))
            y_hard_val_batch = np.tile(y, val_iter)

            x_val, _ = self.pgd.perturb(x_val_batch, y_hard_val_batch, sess, rand=True)

            cur_corr = sess.run(self.model.num_correct,
                                feed_dict={self.model.x_input: x_val,
                                           self.model.y_input: y})

            val_total_corr += cur_corr

        print("{} validation accuracy: {:.2f}%".format(self.val_eps, val_total_corr / (num_images * val_iter) * 100))

        # goal achievement check
        if val_total_corr == (num_images * val_iter):
            print("reached performance goal for", self.val_eps)

            print("reached final objective!")
            suc_flag = True
        else:
            print("failed for", self.val_eps)

        self.reset_pgd()

        return suc_flag
