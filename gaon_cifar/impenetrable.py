"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import math
import tensorflow as tf
import numpy as np
import cifar10_input
from pgd_attack import LinfPGDAttack

import os


class Impenetrable(object):
    def __init__(self, model, args):
        self.model = model
        self.eps = args.eps
        self.loss_func = args.loss_func
        self.imp_random_start = args.imp_random_start
        self.imp_gray_start = args.imp_gray_start
        self.imp_num_steps = args.imp_num_steps
        self.pgd_num_steps = args.pgd_num_steps
        self.pgd_step_size = args.pgd_step_size
        self.res_num_steps = args.res_num_steps
        self.res_step_size = args.res_step_size
        self.save_dir_num = args.save_dir_num
        self.val_step = args.val_step
        self.val_num = args.val_num

        self.pgd = LinfPGDAttack(self.model,
                                 self.eps,
                                 args.pgd_num_steps,
                                 args.pgd_step_size,
                                 args.pgd_random_start,
                                 args.loss_func)

        if self.loss_func == 'xent':
            self.loss = self.model.xent
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

    def fortify(self, x_orig, y, ibatch, meta_name, sess):

        img_dir = './img'+str(self.save_dir_num) + '/'
        arr_dir = './arr'+str(self.save_dir_num) + '/'
        
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if not os.path.exists(arr_dir):
            os.makedirs(arr_dir)

        suc_flag = False
        obj_eps = 8

        num_images = len(x_orig)

        if self.imp_random_start:
            x = np.random.randint(0, 256, x_orig.shape, dtype='int32').astype('float32')
        elif self.imp_gray_start:
            x = np.ones_like(x_orig).astype('float32') * 128
        else:
            x = np.copy(x_orig)

        orig_corr = sess.run(self.model.num_correct,
                            feed_dict={self.model.x_input: x,
                                       self.model.y_input: y})

        print("original accuracy: {:.2f}%".format(orig_corr/num_images*100))
        print()
        
        step = 0

        if self.val_step > 0:

            filename = "{}_img{}_step{}".format(meta_name, ibatch, step)

            val_eps = 8
            self.pgd.num_steps = 20

            while val_eps <= obj_eps:
                val_iter = self.val_num
                val_total_corr = 0

                assert val_iter % 100 == 0

                x_val_batch = np.tile(x, (100, 1, 1, 1))
                y_val_batch = np.tile(y, 100)

                self.pgd.epsilon = val_eps
                self.pgd.step_size = self.pgd.epsilon//4

                for i in range(val_iter//100):

                    x_val = self.pgd.perturb(x_val_batch, y_val_batch, sess,
                                             proj=True, reverse=False, rand=True)

                    cur_corr = sess.run(self.model.num_correct,
                                        feed_dict={self.model.x_input: x_val,
                                                   self.model.y_input: y})

                    val_total_corr += cur_corr

                print("{} validation accuracy: {:.2f}%".format(val_eps, val_total_corr / (num_images * val_iter) * 100))

                # goal achievement check
                if val_total_corr == (num_images * val_iter):
                    print("reached performance goal for", val_eps)
                    filename += '_{}'.format(val_eps)

                    if val_eps == obj_eps:
                        print("reached final objective!")
                        suc_flag = True
                        break
                else:
                    print("failed for", val_eps)
                    break

                val_eps += 1

            self.pgd.epsilon = self.eps
            self.pgd.step_size = self.pgd_step_size
            self.pgd.num_steps = self.pgd_num_steps

            np.save(arr_dir + filename, x)
            im = Image.fromarray(np.uint8(x).reshape((32, 32, 3)))
            im.save(img_dir + filename + '.png')

        step = 1
        while self.imp_num_steps <= 0 or step <= self.imp_num_steps:
            print("step:", step)

            # attack image
            x_adv = self.pgd.perturb(x, y, sess,
                                     proj=True, reverse=False)

            adv_loss, adv_corr, grad = sess.run([self.loss, self.model.num_correct, self.grad],
                                      feed_dict={self.model.x_input: x_adv,
                                                 self.model.y_input: y})

            l1_dist = np.linalg.norm((x_adv - x).flatten(), 1) / x.size
            print("attack accuracy: {:.2f}%".format(adv_corr/num_images*100))
            print("l1 distance: {:.2f}".format(l1_dist))
            print("adv loss: {:.20f}".format(adv_loss/num_images*100))

            # restore image
            x_res = x - self.res_step_size * np.sign(grad)
            x_res = np.clip(x_res, 0, 255)

            res_loss, res_corr = sess.run([self.loss, self.model.num_correct],
                                feed_dict={self.model.x_input: x_res,
                                           self.model.y_input: y})

            l1_dist = np.linalg.norm((x_res - x).flatten(), 1) / x.size
            print("restored accuracy: {:.2f}%".format(res_corr/num_images*100))
            print("l1 distance: {:.2f}".format(l1_dist))
            print("res loss: {:.20f}".format(res_loss/num_images*100))

            x = x_res

            # validation
            if self.val_step > 0 and step % self.val_step == 0:

                filename = "{}_img{}_step{}".format(meta_name, ibatch, step)

                if adv_corr == num_images:

                    val_eps = 8
                    self.pgd.num_steps = 20

                    while val_eps <= obj_eps:
                        val_iter = self.val_num
                        val_total_corr = 0

                        assert val_iter%100 == 0

                        x_val_batch = np.tile(x, (100, 1, 1, 1))
                        y_val_batch = np.tile(y, 100)

                        self.pgd.epsilon = val_eps
                        self.pgd.step_size = self.pgd.epsilon//4

                        for i in range(val_iter//100):

                            x_val = self.pgd.perturb(x_val_batch, y_val_batch, sess,
                                                     proj=True, reverse=False, rand=True)

                            cur_corr = sess.run(self.model.num_correct,
                                                feed_dict={self.model.x_input: x_val,
                                                           self.model.y_input: y})

                            val_total_corr += cur_corr

                        print()

                        print("{} validation accuracy: {:.2f}%".format(val_eps, val_total_corr / (num_images * val_iter) * 100))

                        # goal achievement check
                        if val_total_corr == (num_images * val_iter):
                            print("reached performance goal for", val_eps)
                            filename += '_{}'.format(val_eps)

                            if val_eps == obj_eps:
                                print("reached final objective!")
                                suc_flag = True
                                break
                        else:
                            print("failed for", val_eps)
                            break

                        val_eps += 1

                    self.pgd.epsilon = self.eps
                    self.pgd.step_size = self.pgd_step_size
                    self.pgd.num_steps = self.pgd_num_steps

                np.save(arr_dir + filename, x)
                im = Image.fromarray(np.uint8(x).reshape((32, 32, 3)))
                im.save(img_dir + filename +'.png')

            print()

            step += 1

            if suc_flag:
                break

        return x


def result(x_imp, x_adv, model, sess, x_full_batch, y_full_batch):
    num_eval_examples = x_imp.shape[0]
    eval_batch_size = min(num_eval_examples, 100)
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

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
        cur_corr, y_pred_batch, correct_prediction, losses = \
            sess.run([model.num_correct, model.predictions, model.correct_prediction, model.y_xent],
                     feed_dict=dict_adv)
        total_corr += cur_corr
        accuracy = total_corr / num_eval_examples

    print('imp Accuracy: {:.2f}%'.format(100.0 * accuracy))

    total_corr = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_adv[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_batch}
        cur_corr, y_pred_batch, correct_prediction, losses = \
            sess.run([model.num_correct, model.predictions, model.correct_prediction, model.y_xent],
                     feed_dict=dict_adv)
        total_corr += cur_corr
        accuracy = total_corr / num_eval_examples

    print('adv Accuracy: {:.2f}%'.format(100.0 * accuracy))


if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', default=1000, help='sample size', type=int)
    parser.add_argument('--bstart', default=0, type=int)
    parser.add_argument('--model_dir', default='adv_trained', type=str)
    parser.add_argument('--save_dir_num', default=10, type=int)
    parser.add_argument('--loss_func', default='xent', type=str)
    # PGD
    parser.add_argument('--eps', default=8, help='Attack eps', type=int)
    parser.add_argument('--pgd_random_start', action='store_true')
    parser.add_argument('--pgd_num_steps', default=20, type=int)
    parser.add_argument('--pgd_step_size', default=2, type=float)
    # impenetrable
    parser.add_argument('--imp_random_start', action='store_true')
    parser.add_argument('--imp_gray_start', action='store_true')
    parser.add_argument('--imp_num_steps', default=1000, help='0 for until convergence', type=int)
    parser.add_argument('--res_num_steps', default=0, type=int)
    parser.add_argument('--res_step_size', default=1, type=int)
    # evaluation
    parser.add_argument('--val_step', default=100, help="validation per val_step iterations. =< 0 means no evaluation", type=int)
    parser.add_argument('--val_num', default=100, help="validation PGD numbers per eps", type=int)
    params = parser.parse_args()
    for key, val in vars(params).items():
        print('{}={}'.format(key, val))

    assert not (params.imp_random_start and params.imp_gray_start)

    meta_name = 'nat' if params.model_dir=='naturally_trained' else 'adv'
    meta_name += '_pgd' + '_' + str(params.eps) + '_' + str(params.pgd_num_steps) + '_' + str(params.pgd_step_size) + ('_rand' if params.pgd_random_start else '')
    meta_name += '_imp' + '_' + str(params.imp_num_steps) + ('_rand' if params.imp_random_start else '') + ('_gray' if params.imp_gray_start else '')
    meta_name += '_res' + '_' + str(params.res_num_steps) + '_' + str(params.res_step_size)

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

        # Iterate over the samples batch-by-batch
        num_eval_examples = params.sample_size
        if params.val_step > 0:
            eval_batch_size = 1
        else:
            eval_batch_size = min(config['eval_batch_size'], num_eval_examples)

        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        x_imp = []  # imp accumulator
        x_adv = []  # adv accumulator

        if params.model_dir == 'naturally_trained':
            indices = np.load('/data/home/gaon/lazy-attack/cifar10_data/nat_indices_untargeted.npy')
        else:
            indices = np.load('/data/home/gaon/lazy-attack/cifar10_data/indices_untargeted.npy')

        bstart = params.bstart
        while (True):
            x_candid = cifar.eval_data.xs[indices[bstart:bstart + 100]]
            y_candid = cifar.eval_data.ys[indices[bstart:bstart + 100]]
            mask = sess.run(model.correct_prediction, feed_dict={model.x_input: x_candid,
                                                                 model.y_input: y_candid})
            x_masked = x_candid[mask]
            y_masked = y_candid[mask]
            print(len(x_masked))
            if bstart == 0:
                x_full_batch = x_masked[:min(num_eval_examples, len(x_masked))]
                y_full_batch = y_masked[:min(num_eval_examples, len(y_masked))]
            else:
                index = min(num_eval_examples - len(x_full_batch), len(x_masked))
                x_full_batch = np.concatenate((x_full_batch, x_masked[:index]))
                y_full_batch = np.concatenate((y_full_batch, y_masked[:index]))
            bstart += 100
            if len(x_full_batch) >= (num_eval_examples) or bstart >= 10000:
                break

        print('Iterating over {} batches'.format(num_batches))

        x_full_batch = x_full_batch.astype(np.float32)

        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}'.format(bend - bstart))

            x_batch = x_full_batch[bstart:bend, :]
            y_batch = y_full_batch[bstart:bend]

            print('fortifying image ', bstart)
            x_batch_imp = impenet.fortify(x_batch, y_batch, ibatch, meta_name, sess)

            # evaluation
            x_batch_adv = np.copy(x_batch_imp)

            val_eps = 8
            obj_eps = 8
            num_images = len(x_batch_imp)

            impenet.pgd.num_steps = 20

            success_mask = [True for _ in range(num_images)]

            while val_eps <= obj_eps:
                val_iter = impenet.val_num
                val_total_corr = 0

                impenet.pgd.epsilon = val_eps
                impenet.pgd.step_size = impenet.pgd.epsilon//4

                for i in range(val_iter):

                    x_batch_adv = impenet.pgd.perturb(x_batch_imp, y_batch, sess,
                                                      proj=True, reverse=False, rand=True)

                    corr_mask = sess.run(impenet.model.correct_prediction,
                                                   feed_dict={impenet.model.x_input: x_batch_adv,
                                                              impenet.model.y_input: y_batch})

                    success_mask *= corr_mask
                    num_survived = np.sum(success_mask) / len(success_mask)

                    if sum(success_mask) == 0:
                        break

                print("{} validation accuracy: {:.2f}%".format(val_eps, num_survived * 100))

                # goal achievement check
                if num_survived >= 1:
                    print("reached performance goal for", val_eps)

                    if val_eps == obj_eps:
                        print("reached final objective!")
                        break
                else:
                    print("failed for", val_eps)

                val_eps += 1

            impenet.pgd.epsilon = impenet.eps
            impenet.pgd.step_size = impenet.pgd_step_size
            impenet.pgd.num_steps = impenet.pgd_num_steps

            x_imp.append(x_batch_imp)
            x_adv.append(x_batch_adv)

        x_imp = np.concatenate(x_imp)
        x_adv = np.concatenate(x_adv)

        folder_name = './arr' + '_main' + '/'
        batch_name = '_' + str(params.bstart) + '_' + str(params.sample_size)
        common_name = folder_name + meta_name + batch_name
        
        np.save(common_name + '_x_org', x_full_batch)
        np.save(common_name + '_x_imp', x_imp)
        np.save(common_name + '_y', y_full_batch)

        if np.amax(x_imp) > 255.0001 or \
            np.amin(x_imp) < -0.0001 or \
            np.isnan(np.amax(x_imp)):
            print('Invalid pixel range in x_imp. Expected [0,255], fount[{},{}]'.format(np.amin(x_imp),
                                                                                        np.amax(x_imp)))
        elif np.amax(x_adv) > 255.0001 or \
            np.amin(x_adv) < -0.0001 or \
            np.isnan(np.amax(x_adv)):
            print('Invalid pixel range in x_adv. Expected [0,255], fount[{},{}]'.format(np.amin(x_adv),
                                                                                        np.amax(x_adv)))
        else:
            result(x_imp, x_adv, model, sess, x_full_batch, y_full_batch)


