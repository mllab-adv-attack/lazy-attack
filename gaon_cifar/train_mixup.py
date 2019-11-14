"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil

import tensorflow as tf
import numpy as np
import math
import argparse

from model_mixup import Model
import cifar10_input

CIFAR10_NUM_TRAIN_IMAGES = 50000

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=0.1, type=float, help='initial lr')
    parser.add_argument('--schedule', default=[200, 300], type=int, nargs='+', help='learning rate decay epoch')
    parser.add_argument('--data_path', default='/data/home/gaon/lazy-attack/cifar10_data', type=str)
    parser.add_argument('--model_dir', default='./models/mixup', type=str)
    parser.add_argument('--epoch', default=400, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--mixup_alpha', default=2.0, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    for key, val in vars(args).items():
        print('{} = {}'.format(key, val))

    with open('config.json') as config_file:
        config = json.load(config_file)

# seeding randomness
    tf.set_random_seed(config['tf_random_seed'])
    np.random.seed(config['np_random_seed'])

# Setting up training parameters
    max_num_training_steps = int(args.epoch * CIFAR10_NUM_TRAIN_IMAGES / args.batch_size)
    weight_decay = args.weight_decay
    data_path = args.data_path
    momentum = args.momentum
    batch_size = args.batch_size
    eval_batch_size = args.batch_size

# Setting up the data and the model
    raw_cifar = cifar10_input.CIFAR10Data(data_path)
    global_step = tf.contrib.framework.get_or_create_global_step()

    scope = tf.get_variable_scope()
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        model = Model(mode='train', mixup_alpha=args.mixup_alpha)
        eval_model = Model(mode='eval', mixup_alpha=args.mixup_alpha)

# Setting up the optimizer
    boundaries = [int(schedule * CIFAR10_NUM_TRAIN_IMAGES / args.batch_size) for schedule in args.schedule]
    values = [args.lr, args.lr*0.1, args.lr*0.01]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values)
    xent_loss = model.mean_xent2
    weight_decay_loss = model.weight_decay_loss
    total_loss = xent_loss + weight_decay * weight_decay_loss
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(
        total_loss,
        global_step=global_step)

# Setting up the Tensorboard and checkpoint outputs
    model_dir = (args.model_dir + str(args.lr))
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

    saver = tf.train.Saver(max_to_keep=3)
    tf.summary.scalar('accuracy train', model.accuracy)
    tf.summary.scalar('accuracy', model.accuracy)
    tf.summary.scalar('xent train', xent_loss)
    tf.summary.scalar('xent', xent_loss)
    tf.summary.image('images train', model.x_input)
    merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
    shutil.copy('config.json', model_dir)

    with tf.Session() as sess:

      # initialize data augmentation
      cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

      sess.run(tf.global_variables_initializer())

      for ii in range(max_num_training_steps+1):
        x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                           multiple_passes=True)

        x, w, l, _ = sess.run([xent_loss, weight_decay_loss, total_loss, train_step],
                              feed_dict={model.x_input: x_batch,
                                         model.y_input: y_batch})
        print('loss: {:.4f}, xent: {:.4f}, weight decay loss: {:.4f}'.format(l, x, w))

        if ii != 0 and ii % 10000 == 0:
            num_eval = raw_cifar.eval_data.n
            num_batches = int(math.ceil(num_eval / eval_batch_size))

            num_correct = 0

            for batch in range(num_batches):
                bstart = batch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval)

                x_batch = raw_cifar.eval_data.xs[bstart:bend]
                y_batch  =raw_cifar.eval_data.ys[bstart:bend]

                corr = sess.run(eval_model.num_correct,
                                feed_dict={eval_model.x_input: x_batch,
                                           eval_model.y_input: y_batch})

                num_correct += corr

            accuracy = num_correct / num_eval * 100

            print('step: {}, accuracy: {:.2f}%'.format(ii, accuracy))

            print('saving...')
            saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
        
