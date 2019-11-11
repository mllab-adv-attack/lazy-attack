"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
import math

from model_mixup import Model
import cifar10_input

with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = 100000
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = 1e-4
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']
eval_batch_size = 100

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()

scope = tf.get_variable_scope()
with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    model = Model(mode='train')
    eval_model = Model(mode='eval')

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = 0.1
xent_loss = model.xent
weight_decay_loss = model.weight_decay_loss
total_loss = xent_loss + weight_decay * weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(
    total_loss,
    global_step=global_step)

# Setting up the Tensorboard and checkpoint outputs
model_dir = 'models/mixup'
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
tf.summary.scalar('xent train', model.xent / batch_size)
tf.summary.scalar('xent', model.xent / batch_size)
tf.summary.image('images train', model.x_input)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

with tf.Session() as sess:

  # initialize data augmentation
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

  sess.run(tf.global_variables_initializer())

  for ii in range(max_num_training_steps):
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                       multiple_passes=True)

    x, w, l, _ = sess.run([xent_loss, weight_decay_loss, total_loss, train_step],
                          feed_dict={model.x_input: x_batch,
                                     model.y_input: y_batch})
    print('loss: {:.4f}, xent: {:.4f}, weight decay loss: {:.4f}'.format(l, x, w))

    if ii != 0 and ii % 500 == 0:
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

        print('step: {}, accuracy: {}%'.format(ii, accuracy))

        print('saving...')
        saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
