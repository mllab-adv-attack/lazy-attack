"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np
import cifar10_input
from pgd_attack import LinfPGDAttack


class Impenetrable(object):
    def __init__(self, model, args):
        self.model = model
        self.eps = args.eps
        self.loss_func = args.loss_func
        self.imp_random_start = args.imp_random_start
        self.imp_num_steps = args.imp_num_steps
        self.res_num_steps = args.res_num_steps
        self.res_step_size = args.res_step_size

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

    def fortify(self, x_orig, y, sess):

        num_images = len(x_orig)

        if self.random_start:
            x = np.random.randint(0, 256, x_orig.shape, dtype='int32')
        else:
            x = np.copy(x_orig)

        cur_corr = sess.run(self.model.num_correct,
                            feed_dict={self.model.x_input: x,
                                       self.model.y_input: y})

        print("original accuracy: {:.2f}%".format(cur_corr/num_images*100))
        print()

        step = 1
        while self.imp_num_steps <= 0 or step <= self.imp_num_steps:
            print("step:", step)

            # attack image
            x_adv = self.pgd.perturb(x, y, sess,
                                     proj=True, reverse=False)

            cur_corr = sess.run(self.model.num_correct,
                                      feed_dict={self.model.x_input: x_adv,
                                                 self.model.y_input: y})

            change_ratio = np.count_nonzero(x_adv - x) / x.size
            print("attack accuracy: {:.2f}%".format(cur_corr/num_images*100))
            print("change ratio: {:.2f}%".format(change_ratio*100))

            # restore image
            x_res = self.pgd.perturb(x, y, sess,
                                     proj=False, reverse=True,
                                     step_size=self.res_step_size, num_steps=self.res_num_steps)

            cur_corr = sess.run(self.model.num_correct,
                                feed_dict={self.model.x_input: x_res,
                                           self.model.y_input: y})

            change_ratio = np.count_nonzero(x_res - x_adv) / x.size
            print("restored accuracy: {:.2f}%".format(cur_corr/num_images*100))
            print("change ratio: {:.2f}%".format(change_ratio*100))

            # validation
            val_iter = 20
            val_total_corr = 0
            for i in range(val_iter):
                x_val = self.pgd.perturb(x_res, y, sess,
                                         proj=True, reverse=False, rand=True)

                cur_corr = sess.run(self.model.num_correct,
                                    feed_dict={self.model.x_input: x_val,
                                               self.model.y_input: y})

                val_total_corr += cur_corr

            print("validation accuracy: {:.2f}%".format(val_total_corr/(num_images*val_iter)*100))

            print()

            x = x_res

            step += 1

        return x


def result(x_imp, x_adv, model, sess, x_full_batch, y_full_batch):
    num_eval_samples = x_imp.shape[0]
    eval_batch_size = min(num_eval_samples, 100)
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    l_inf = np.amax(np.abs(x_full_batch - x_imp))
    if l_inf > params.eps + 0.001:
        print('breached maximum perturbation')
        print(l_inf)
        return

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
    parser.add_argument('--sample_size', default=100, help='sample size', type=int)
    parser.add_argument('--model_dir', default='adv_trained', type=str)
    parser.add_argument('--loss_func', default='xent', type=str)
    # PGD
    parser.add_argument('--eps', default=8, help='Attack eps', type=float)
    parser.add_argument('--pgd_random_start', action='store_true')
    parser.add_argument('--pgd_num_steps', default=20, type=int)
    parser.add_argument('--pgd_step_size', default=2, type=float)
    # impenetrable
    parser.add_argument('--imp_random_start', action='store_true')
    parser.add_argument('--imp_num_steps', default=0, help='0 for until convergence', type=int)
    parser.add_argument('--res_num_steps', default=8, type=int)
    parser.add_argument('--res_step_size', default=1, type=float)
    params = parser.parse_args()
    for key, val in vars(params).items():
        print('{}={}'.format(key, val))

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
        eval_batch_size = min(config['eval_batch_size'], num_eval_examples)
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        x_imp = []  # imp accumulator
        x_adv = []  # adv accumulator

        bstart = 0
        while (True):
            x_candid = cifar.eval_data.xs[bstart:bstart + 100]
            y_candid = cifar.eval_data.ys[bstart:bstart + 100]
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
            if len(x_full_batch) >= num_eval_examples or bstart >= 10000:
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
            x_batch_imp = impenet.fortify(x_batch, y_batch, sess)
            x_batch_adv = impenet.pgd.perturb(x_batch_imp, y_batch, sess)

            x_imp.append(x_batch_imp)
            x_adv.append(x_batch_adv)

        x_imp = np.concatenate(x_imp)
        x_adv = np.concatenate(x_adv)

        if np.amax(x_imp) > 255.0001 or \
            np.amin(x_imp) < -0.0001 or \
            np.isnan(np.amax(x_imp)):
            print('Invalid pixel range. Expected [0,255], fount[{},{}]'.format(np.amin(x_imp),
                                                                               np.amax(x_imp)))
        else:
            result(x_imp, x_adv, model, sess, x_full_batch, y_full_batch)


