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
        # self.pgd_step_size = args.pgd_step_size
        self.pgd_step_size = self.pgd_eps/4.0
        self.pgd_restarts = args.pgd_restarts
        self.pgd_random_start = args.pgd_random_start or (self.pgd_restarts > 1)
        self.imp_step_size = args.imp_step_size
        self.val_step_per = args.val_step_per
        self.val_restarts = args.val_restarts
        self.val_num_steps = args.val_num_steps
        self.val_eps = args.val_eps
        self.adam = args.imp_adam
        self.rep = args.imp_rep
        self.pp = args.imp_pp
        self.soft_label = args.soft_label
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
            self.loss2 = self.model.xent2
            self.loss2_full = self.model.y_xent2
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
        self.grad2 = tf.gradients(self.loss2, self.model.x_input)[0]
        self.softmax = self.model.softmax

    def fortify(self, x_orig, y, sess, y_gt=None):
        num_classes = 10

        soft_label_tmp = self.soft_label

        # y_hard: soft label to hard label (batch, 10)
        y_hard = np.argmax(y, axis=1)
        y_hard = np.eye(10)[y_hard.reshape(-1)]

        print('gt class:', np.nonzero(y_gt == 1)[1][0])

        pred_class = np.nonzero(y_hard == 1)[1][0]
        print('pred class:', pred_class)

        suc_flag = False

        num_images = len(x_orig)

        # initialize image
        if self.imp_random_start > 0:
            x = x_orig + np.random.uniform(-self.imp_random_start, self.imp_random_start, x_orig.shape)
            x = np.clip(x, 0, 255)
        elif self.imp_gray_start:
            x = np.ones_like(x_orig).astype('float32') * 128
        else:
            x = np.copy(x_orig)

        orig_loss, orig_corr, orig_soft = sess.run([self.loss2, self.model.num_correct2, self.softmax],
                                                   feed_dict={self.model.x_input: x,
                                                              self.model.y_input2: y_gt if self.soft_label > 0
                                                              else y_hard})

        print("original accuracy: {:.2f}%".format(orig_corr/num_images*100))
        print("original loss: {:.20f}".format(orig_loss/num_images))
        if self.soft_label >= 1:
            print("original softmax: ", orig_soft)
        print()

        # adam parameters
        beta1 = 0.9
        beta2 = 0.999
        m = np.zeros_like(x)
        v = np.zeros_like(x)

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
            if self.soft_label < 1:
                x_adv_batch = np.tile(x, (self.pgd_restarts, 1, 1, 1))
                y_hard_val_batch = np.tile(y_hard, (self.pgd_restarts, 1))
            else:
                x_adv_batch = np.tile(x, (num_classes*self.pgd_restarts, 1, 1, 1))
                y_hard_val_batch = np.array([i//self.pgd_restarts for i in range(num_classes*self.pgd_restarts)]).flatten()
                y_hard_val_batch = np.eye(10)[y_hard_val_batch.reshape(-1)]

            x_adv_batch, x_adv_batch_full = self.pgd.perturb(x_adv_batch, y_hard_val_batch, sess,
                                           proj=True, reverse=False, pp=self.pp)

            #print(self.pgd_restarts, self.pgd_num_steps, self.pp)
            #print(x_adv_batch_full.shape)

            if self.pp > 0:
                x_adv_batch = x_adv_batch_full
                #print(x_adv_batch.shape)
                y_hard_val_batch = np.tile(y_hard, (len(x_adv_batch), 1))
            
            adv_loss, adv_loss_full, adv_corr, grad = sess.run([self.loss2, self.loss2_full,
                                                                self.model.num_correct2, self.grad2],
                                                               feed_dict={self.model.x_input: x_adv_batch,
                                                                          self.model.y_input2: y_hard_val_batch})
            
            # soft label test
            if 1 <= self.soft_label < 5:
                grad_full = np.linalg.norm(grad.reshape(len(x_adv_batch), -1), axis=1)
                #print('grad l2 norm(full):', grad_full.reshape(num_classes, -1))
                #print('adv loss(full):', adv_loss_full.reshape(num_classes, -1))
                grad_full2 = []
                adv_loss_full2 = []
                for idx in range(num_classes):
                    grad_full2.append(np.mean(grad_full[idx * self.pgd_restarts: (idx+1) * self.pgd_restarts]))
                    adv_loss_full2.append(np.mean(adv_loss_full[idx * self.pgd_restarts: (idx+1) * self.pgd_restarts]))
                grad_full = np.array(grad_full2)
                adv_loss_full = np.array(adv_loss_full2)
                print('grad l2 norm:', grad_full)
                print('max grad class:', np.argmax(grad_full))
                print('min grad class:', np.argmin(grad_full))
                print('loss_full:', adv_loss_full)
                print('max loss class:', np.argmax(adv_loss_full))
                print('min loss class:', np.argmin(adv_loss_full))
                
                if self.soft_label == 1 and np.argmin(grad_full) == np.argmin(adv_loss_full):
                    print('setting gt class to:', np.argmin(adv_loss_full))
                    pred_class = np.argmin(adv_loss_full)
                
                if self.soft_label == 2:
                    print('setting gt class to:', np.argmax(adv_loss_full))
                    pred_class = np.argmax(adv_loss_full)
                
                if self.soft_label == 3:
                    print('setting gt class to:', np.argmax(grad_full))
                    pred_class = np.argmax(grad_full)

                if self.soft_label in [1, 2, 3]:

                    x_adv_batch = x_adv_batch[pred_class * self.pgd_restarts: (pred_class+1) * self.pgd_restarts]
                    
                    y_hard_val_batch = np.array([pred_class for _ in range(len(x_adv_batch))]).flatten()
                    y_hard_val_batch = np.eye(10)[y_hard_val_batch.reshape(-1)]
            
                    adv_loss, adv_loss_full, adv_corr, grad = sess.run([self.loss2, self.loss2_full,
                                                                        self.model.num_correct2, self.grad2],
                                                                       feed_dict={self.model.x_input: x_adv_batch,
                                                                                  self.model.y_input2: y_hard_val_batch})

            if self.rep:
                # use reptile (x_adv - x as gd direction)
                grad = np.mean(x_adv_batch - x, axis=0)
            else:
                # use MAML
                grad = np.mean(grad, axis=0)


            # l1_dist = np.linalg.norm((x_adv - x).flatten(), 1) / x.size
            print("attack accuracy: {:.2f}%".format(adv_corr/len(x_adv_batch)*100))
            # print("l1 distance: {:.2f}".format(l1_dist))
            print("adv loss: {:.20f}".format(adv_loss/len(x_adv_batch)))

            # restore image
            if self.adam:
                adam_lr = self.imp_step_size * np.sqrt(1-beta2**step)/(1-beta1**step)
                m = beta1 * m + (1-beta1) * grad
                v = beta2 * v + (1-beta2) * grad * grad
                x_res = x - adam_lr * m / (np.sqrt(v) + 1e-8)
            else:
                if self.imp_no_sign:
                    x_res = x - self.imp_step_size * grad
                else:
                    x_res = x - self.imp_step_size * np.sign(grad)

            x_res = np.clip(x_res, 0, 255)

            if self.imp_delta > 0:
                x_res = np.clip(x_res, x_orig-self.imp_delta, x_orig+self.imp_delta)

            res_loss, res_corr, res_soft = sess.run([self.loss2, self.model.num_correct2, self.softmax],
                                                    feed_dict={self.model.x_input: x_res,
                                                               self.model.y_input2: y_gt if self.soft_label > 0
                                                               else y_hard})

            l2_dist = np.linalg.norm((x_res - x_orig).flatten()) / 255.0
            print("restored accuracy: {:.2f}%".format(res_corr/num_images*100))
            print("l2 distance: {:.2f}".format(l2_dist))
            print("res loss: {:.20f}".format(res_loss/num_images))
            if self.soft_label >= 1:
                res_pred = np.argmax(res_soft)
                print("res softmax:", res_soft)
                print("res pred:", np.argmax(res_soft))

                if self.soft_label >= 5:
                    if step <= 8 and res_pred != pred_class:
                        print('setting gt class to:', res_pred)
                        y_hard = np.argmax(res_soft, axis=1)
                        y_hard = np.eye(10)[y_hard.reshape(-1)]
                        self.soft_label = 0
                    elif step >= 8:
                        print('no change to class')
                        self.soft_label = 0

            x = x_res

            # validation
            if self.val_step_per > 0 and step % self.val_step_per == 0:

                if adv_corr == len(x_adv_batch):

                    suc_flag = self.validation(x, y_hard, sess)
                    
                    # special early stop for soft_label 4
                    if self.soft_label == 4:
                        suc_flag = True

            print()

            # early stop
            if suc_flag:
                break

            step += 1

        self.soft_label = soft_label_tmp

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
        self.set_pgd(self.val_eps, self.val_eps/4.0, self.val_num_steps, True)

    def validation(self, x, y_hard, sess):
        
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
            y_hard_val_batch = np.tile(y_hard, (100, 1))

            for i in range(val_iter//100):

                x_val, _ = self.pgd.perturb(x_val_batch, y_hard_val_batch, sess,
                                         proj=True, reverse=False, rand=True)

                cur_corr = sess.run(self.model.num_correct2,
                                    feed_dict={self.model.x_input: x_val,
                                               self.model.y_input2: y_hard})

                val_total_corr += cur_corr
        else:
            x_val_batch = np.tile(x, (val_iter, 1, 1, 1))
            # y_val_batch = np.tile(y, (100, 1))
            y_hard_val_batch = np.tile(y_hard, (val_iter, 1))

            x_val, _ = self.pgd.perturb(x_val_batch, y_hard_val_batch, sess,
                                     proj=True, reverse=False, rand=True)

            cur_corr = sess.run(self.model.num_correct2,
                                feed_dict={self.model.x_input: x_val,
                                           self.model.y_input2: y_hard})

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
