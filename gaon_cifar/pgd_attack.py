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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', default=8, help='Attack eps', type=float)
    parser.add_argument('--sample_size', default=10000, help='sample size', type=int)
    parser.add_argument('--model_dir', default='naturally_trained', type=str)
    parser.add_argument('--loss_func', default='xent', type=str)
    parser.add_argument('--random_start', default='n', type=str)
    parser.add_argument('--num_steps', default=20, type=int)
    parser.add_argument('--step_size', default=2, type=float)
    parser.add_argument('--test', default='y', help='run attack', type=str)
    params = parser.parse_args()

    for key, val in vars(params).items():
        print('{}={}'.format(key,val))


class LinfPGDAttack:
    def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.step_size = step_size
        self.rand = random_start
        self.queries = []

        if loss_func == 'xent':
            self.loss = model.xent
        elif loss_func == 'cw':
            label_mask = tf.one_hot(model.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
            self.loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss = model.xent

        self.grad = tf.gradients(self.loss, model.x_input)[0]

    def perturb(self, x_nat, y, sess, proj=True, reverse=False, rand=False, step_size=None, num_steps=None):

        """Given a set of examples (x_nat, y), returns a set of adversarial
        examples within epsilon of x_nat in l_infinity norm."""

        if not step_size:
            step_size = self.step_size
        if not num_steps:
            num_steps = self.num_steps

        if rand or ((not reverse) and self.rand):
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, 0, 255)
        else:
            x = np.copy(x_nat)

        for i in range(num_steps):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                                  self.model.y_input: y})

            if not reverse:
                x += step_size * np.sign(grad)
            else:
                x -= step_size * np.sign(grad)

            if proj:
                x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)

            x = np.clip(x, 0, 255) # ensure valid pixel range

        return x


def run_attack(x_adv, model, sess, x_full_batch, y_full_batch):

    num_eval_samples = x_adv.shape[0]
    eval_batch_size = min(num_eval_samples, 100)
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_corr=0

    l_inf = np.amax(np.abs(x_full_batch-x_adv))
    if l_inf > params.eps+0.001:
        print('breached maximum perturbation')
        print(l_inf)
        return
    y_pred = []
    success = []
    succ_li = []
    loss_li = []
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
        loss_li.append(losses)
        succ_li.append(correct_prediction)
        total_corr += cur_corr
        y_pred.append(y_pred_batch)
        accuracy = total_corr / num_eval_examples
        success.append(np.array(np.nonzero(np.invert(correct_prediction)))+ibatch * eval_batch_size)

    print('on eps:{}, sample_size:{}'.format(params.eps, params.sample_size))
    print('adv Accuracy: {:.2f}%'.format(100.0 * accuracy))

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


if __name__ == '__main__':
    import json
    import sys

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint('models/'+params.model_dir)
    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model(mode='eval')
    attack = LinfPGDAttack(model,
                           params.eps,
                           params.num_steps,
                           params.step_size,
                           params.random_start == 'y',
                           params.loss_func)
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

        x_adv = [] # adv accumulator
        masks = []

        bstart = 0
        while(True):
            x_candid = cifar.eval_data.xs[bstart:bstart+100]
            y_candid = cifar.eval_data.ys[bstart:bstart+100]
            mask = sess.run(model.correct_prediction, feed_dict = {model.x_input: x_candid,
                                                                   model.y_input: y_candid})
            x_masked = x_candid[mask]
            y_masked = y_candid[mask]
            print(len(x_masked))
            if bstart == 0:
                x_full_batch = x_masked[:min(num_eval_examples, len(x_masked))]
                y_full_batch = y_masked[:min(num_eval_examples, len(y_masked))]
            else:
                index = min(num_eval_examples-len(x_full_batch), len(x_masked))
                x_full_batch = np.concatenate((x_full_batch, x_masked[:index]))
                y_full_batch = np.concatenate((y_full_batch, y_masked[:index]))
            bstart += 100
            masks.append(mask)
            if len(x_full_batch) >= num_eval_examples or bstart >= 10000:
                break

        masks = np.concatenate(masks)
        indices = np.nonzero(masks)
        np.save('nat_indices_untargeted.npy', indices)

        sys.exit()

        print('Iterating over {} batches'.format(num_batches))

        x_full_batch = x_full_batch.astype(np.float32)

        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}'.format(bend - bstart))

            x_batch = x_full_batch[bstart:bend, :]
            y_batch = y_full_batch[bstart:bend]

            print('Attacking image ', bstart)
            x_batch_adv = attack.perturb(x_batch, y_batch, sess)

            x_adv.append(x_batch_adv)

        x_adv = np.concatenate(x_adv)

        if params.test == 'y':
            if np.amax(x_adv) > 255.0001 or \
                np.amin(x_adv) < -0.0001 or \
                    np.isnan(np.amax(x_adv)):
                print('Invalid pixel range. Expected [0,255], fount[{},{}]'.format(np.amin(x_adv),
                                                                                   np.amax(x_adv)))
            else:
                run_attack(x_adv, model, sess, x_full_batch, y_full_batch)
