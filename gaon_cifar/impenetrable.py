"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from PIL import Image

import math
import tensorflow as tf
import numpy as np
import cifar10_input
from pgd_attack import LinfPGDAttack

# import os


class Impenetrable(object):
    def __init__(self, model, args):
        self.model = model
        self.loss_func = args.loss_func
        self.imp_random_start = args.imp_random_start
        self.imp_gray_start = args.imp_gray_start
        self.imp_num_steps = args.imp_num_steps
        self.imp_eps = args.imp_eps
        self.pgd_eps = args.pgd_eps
        self.pgd_num_steps = args.pgd_num_steps
        # self.pgd_step_size = args.pgd_step_size
        self.pgd_step_size = self.pgd_eps/4.0
        self.pgd_restarts = args.pgd_restarts
        self.pgd_random_start = args.pgd_random_start or (self.pgd_restarts > 1)
        self.imp_step_size = args.imp_step_size
        self.save_dir_num = args.save_dir_num
        self.val_step_per = args.val_step_per
        self.val_restarts = args.val_restarts
        self.val_num_steps = args.val_num_steps
        self.val_eps = args.val_eps
        self.adam = args.imp_adam
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

        np.set_printoptions(precision=6, suppress=True)
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
        if self.imp_random_start:
            x = np.random.randint(0, 256, x_orig.shape, dtype='int32').astype('float32')
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

            x_adv_batch = self.pgd.perturb(x_adv_batch, y_hard_val_batch, sess,
                                           proj=True, reverse=False)

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

            if self.imp_eps > 0:
                x_res = np.clip(x_res, x_orig-self.imp_eps, x_orig+self.imp_eps)

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

                    suc_flag = self.validation(x, y_hard)

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

    def validation(self, x, y_hard):
        
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

                x_val = self.pgd.perturb(x_val_batch, y_hard_val_batch, sess,
                                         proj=True, reverse=False, rand=True)

                cur_corr = sess.run(self.model.num_correct2,
                                    feed_dict={self.model.x_input: x_val,
                                               self.model.y_input2: y_hard})

                val_total_corr += cur_corr
        else:
            x_val_batch = np.tile(x, (val_iter, 1, 1, 1))
            # y_val_batch = np.tile(y, (100, 1))
            y_hard_val_batch = np.tile(y_hard, (val_iter, 1))

            x_val = self.pgd.perturb(x_val_batch, y_hard_val_batch, sess,
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


def result(x_imp, model, sess, x_full_batch, y_full_batch):
    num_eval_examples = x_imp.shape[0]
    eval_batch_size = min(num_eval_examples, 100)
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    # if one-hot, decode one-hot to labels
    if y_full_batch.size > y_full_batch.shape[0]:
        y_full_batch = np.argmax(y_full_batch, axis=1).reshape(-1, 1)

    total_corr = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_full_batch[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_batch}
        cur_corr, y_pred_batch = sess.run([model.num_correct, model.predictions],
                                          feed_dict=dict_adv)
        total_corr += cur_corr
    accuracy = total_corr / num_eval_examples

    print('nat Accuracy: {:.2f}%'.format(100.0 * accuracy))

    total_corr = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_imp[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_batch}
        cur_corr, y_pred_batch = sess.run([model.num_correct, model.predictions],
                                          feed_dict=dict_adv)
        total_corr += cur_corr
    accuracy = total_corr / num_eval_examples

    print('imp Accuracy: {:.2f}%'.format(100.0 * accuracy))


if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', default=1000, help='sample size', type=int)
    parser.add_argument('--bstart', default=0, type=int)
    parser.add_argument('--model_dir', default='naturally_trained', type=str)
    parser.add_argument('--corr_only', action='store_true')
    parser.add_argument('--fail_only', action='store_false')
    parser.add_argument('--save_dir_num', default=10, type=int)
    parser.add_argument('--loss_func', default='xent', type=str)
    # PGD (training)
    parser.add_argument('--pgd_eps', default=8, help='Attack eps', type=int)
    parser.add_argument('--pgd_num_steps', default=20, type=int)
    parser.add_argument('--pgd_step_size', default=2, type=float)
    parser.add_argument('--pgd_random_start', action='store_true')
    parser.add_argument('--pgd_restarts', default=20, help="training PGD restart numbers per eps", type=int)
    # impenetrable
    parser.add_argument('--imp_eps', default=0, help='<= 0 for no imp eps', type=float)
    parser.add_argument('--imp_random_start', action='store_true')
    parser.add_argument('--imp_gray_start', action='store_true')
    parser.add_argument('--imp_num_steps', default=1000, help='0 for until convergence', type=int)
    parser.add_argument('--imp_step_size', default=1, type=float)
    parser.add_argument('--imp_adam', action='store_true')
    parser.add_argument('--imp_no_sign', action='store_true')
    parser.add_argument('--soft_label', default=0, help='0: hard gt, +:soft inferred, -: -(target label+1)', type=int)
    # PGD (evaluation)
    parser.add_argument('--val_step_per', default=10, help="validation per val_step. =< 0 means no eval", type=int)
    parser.add_argument('--val_eps', default=8, help='Evaluation eps', type=int)
    parser.add_argument('--val_num_steps', default=20, help="validation PGD number of steps per PGD", type=int)
    parser.add_argument('--val_restarts', default=20, help="validation PGD restart numbers per eps", type=int)
    params = parser.parse_args()
    for key, val in vars(params).items():
        print('{}={}'.format(key, val))

    assert not (params.imp_random_start and params.imp_gray_start)
    assert not (params.corr_only and params.fail_only)

    meta_name = 'nat' if params.model_dir == 'naturally_trained' else 'adv'
    meta_name += '_pgd' + '_' + str(params.pgd_eps) \
                 + '_' + str(params.pgd_num_steps) \
                 + '_' + str(params.pgd_step_size) \
                 + ('_rand' if params.pgd_random_start else '') \
                 + '_' + str(params.pgd_restarts)
    meta_name += '_imp' + ('_' + str(params.imp_eps)) \
                 + ('_' + str(params.imp_num_steps)) \
                 + ('_' + str(params.imp_step_size))
    meta_name += ('_adam' if params.imp_adam else '') + ('_nosign' if params.imp_no_sign else '')
    meta_name += ('_corr' if params.corr_only else '')
    meta_name += '_val' + ('_' + str(params.val_step_per)) \
                 + ('_' + str(params.val_eps)) \
                 + ('_' + str(params.val_num_steps)) \
                 + ('_' + str(params.val_restarts))
    meta_name += '_' + str(params.soft_label)

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint('models/' + params.model_dir)
    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model(mode='eval')
    impenet = Impenetrable(model,
                           params)
    saver = tf.train.Saver()

    data_path = config['data_path']
    cifar = cifar10_input.CIFAR10Data(data_path)

    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    with tf.Session(config=configs) as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        # Set number of examples to evaluate
        num_eval_examples = params.sample_size

        if params.corr_only:
            if params.model_dir == 'naturally_trained':
                indices = np.load('/data/home/gaon/lazy-attack/cifar10_data/nat_indices_untargeted.npy')
            else:
                indices = np.load('/data/home/gaon/lazy-attack/cifar10_data/indices_untargeted.npy')
        elif params.fail_only:
            if params.model_dir == 'naturally_trained':
                indices = np.load('/data/home/gaon/lazy-attack/cifar10_data/fail_nat_indices_untargeted.npy')
            else:
                indices = np.load('/data/home/gaon/lazy-attack/cifar10_data/fail_indices_untargeted.npy')
        else:
            indices = [i for i in range(params.sample_size)]

        # load data
        bstart = params.bstart
        while True:
            '''
            x_candid = cifar.eval_data.xs[indices[bstart:bstart + 100]]
            y_candid = cifar.eval_data.ys[indices[bstart:bstart + 100]]
            mask, logits = sess.run([model.correct_prediction, model.pre_softmax],
                                    feed_dict={model.x_input: x_candid,
                                               model.y_input: y_candid})
            x_masked = x_candid[mask]
            y_masked = y_candid[mask]
            logit_masked = logits[mask]
            print(len(x_masked))
            if bstart == params.bstart:
                x_full_batch = x_masked[:min(num_eval_examples, len(x_masked))]
                y_full_batch = y_masked[:min(num_eval_examples, len(y_masked))]
                logit_full_batch = logit_masked[:min(num_eval_examples, len(logit_masked))]
            else:
                index = min(num_eval_examples - len(x_full_batch), len(x_masked))
                x_full_batch = np.concatenate((x_full_batch, x_masked[:index]))
                y_full_batch = np.concatenate((y_full_batch, y_masked[:index]))
                logit_full_batch = np.concatenate((logit_full_batch, logit_masked[:index]))
            bstart += 100
            if (len(x_full_batch) >= num_eval_examples) or bstart >= 10000:
                break
            '''
            x_candid = cifar.eval_data.xs[indices[bstart:bstart + 100]]
            y_candid = cifar.eval_data.ys[indices[bstart:bstart + 100]]
            mask, logits = sess.run([model.correct_prediction, model.pre_softmax],
                                    feed_dict={model.x_input: x_candid,
                                               model.y_input: y_candid})
            print(sum(mask))
            if params.corr_only and (np.mean(mask) < 1.0 - 1E-6):
                raise Exception
            if params.fail_only and (np.mean(mask) > 0.0 + 1E-6):
                raise Exception
            if bstart == params.bstart:
                x_full_batch = x_candid[:min(num_eval_examples, len(x_candid))]
                y_full_batch = y_candid[:min(num_eval_examples, len(y_candid))]
                logit_full_batch = logits[:min(num_eval_examples, len(logits))]
            else:
                index = min(num_eval_examples - len(x_full_batch), len(x_candid))
                x_full_batch = np.concatenate((x_full_batch, x_candid[:index]))
                y_full_batch = np.concatenate((y_full_batch, y_candid[:index]))
                logit_full_batch = np.concatenate((logit_full_batch, logits[:index]))
            bstart += 100
            if (len(x_full_batch) >= num_eval_examples) or bstart >= len(indices):
                break

        # Adjust num_eval_examples. Iterate over the samples batch-by-batch
        num_eval_examples = len(x_full_batch)

        if params.val_step_per > 0:
            eval_batch_size = 1
        else:
            eval_batch_size = min(config['eval_batch_size'], num_eval_examples)

        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        print('Iterating over {} batches'.format(num_batches))

        x_full_batch = x_full_batch.astype(np.float32)

        x_imp = []  # imp accumulator
        step_imp = []  # num steps accumulator

        # y to one-hot
        # y_full_batch: (batch, 1)
        if params.soft_label == 0:
            # y_full_batch_oh: ground truth label (batch, 10)
            y_full_batch_oh = np.eye(10)[y_full_batch.reshape(-1)]
        elif params.soft_label > 0:
            # y_full_batch_oh: softmax layer output (batch, 10)
            y_full_batch_oh = logit_full_batch
            y_full_batch_oh = np.exp(y_full_batch_oh) / np.sum(np.exp(y_full_batch_oh), axis=1).reshape(-1, 1)
        else:
            # y_full_batch_oh: target label (batch, 10)
            print('target label:', -params.soft_label-1)
            target_batch = np.array([(-params.soft_label - 1) for _ in range(len(x_full_batch))])
            y_full_batch_oh = np.eye(10)[target_batch.reshape(-1)]

        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}'.format(bend - bstart))

            x_batch = x_full_batch[bstart:bend, :]
            y_batch = y_full_batch_oh[bstart:bend, :]
            y_batch_gt = y_full_batch[bstart:bend]
            # y_batch_gt: ground truth label in one-hot (batch, 10)
            y_batch_gt = np.eye(10)[y_batch_gt.reshape(-1)]

            # run our algorithm
            print('fortifying image ', bstart)
            x_batch_imp, step_batch_imp = impenet.fortify(x_batch, y_batch, sess, y_batch_gt)
            print()

            # evaluation
            # y_batch_hard = np.argmax(y_batch, axis=1)
            # y_batch_hard = np.eye(10)[y_batch_hard.reshape(-1)]

            # suc_flag = impenet.validation(x_batch_imp, y_batch, y_batch_hard)

            x_imp.append(x_batch_imp)
            step_imp.append(step_batch_imp)

        x_imp = np.concatenate(x_imp)
        step_imp = np.concatenate(step_imp)

        # save image
        folder_name = './arr' + '_main' + '/'
        batch_name = '_' + str(params.bstart) + '_' + str(params.sample_size)
        common_name = folder_name + meta_name + batch_name
        
        np.save(common_name + '_x_org', x_full_batch)
        np.save(common_name + '_x_imp', x_imp)
        np.save(common_name + '_step_imp', step_imp)
        np.save(common_name + '_y', y_full_batch)

        # sanity check
        if np.amax(x_imp) > 255.0001 or \
            np.amin(x_imp) < -0.0001 or \
            np.isnan(np.amax(x_imp)):
            print('Invalid pixel range in x_imp. Expected [0,255], fount[{},{}]'.format(np.amin(x_imp),
                                                                                        np.amax(x_imp)))
        else:
            result(x_imp, model, sess, x_full_batch, y_full_batch)


