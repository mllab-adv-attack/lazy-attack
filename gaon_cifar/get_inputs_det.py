"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import json
import os
import shutil

import tensorflow as tf
import numpy as np

from model import Model
import cifar10_input
from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])
 
# Setting up training parameters
max_num_training_steps = 80000
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_input)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

with tf.Session() as sess:

  # initialize data augmentation
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  x_full_batch = []
  y_full_batch = []

  start = time.time()

  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                       multiple_passes=True)

    x_full_batch.append(x_batch.astype('uint8'))
    y_full_batch.append(y_batch)

    if ((ii+1)%100==0):
      end = time.time()
      saving_time = end-start
      print('{} th batch saved'.format(ii+1))
      print('time took: {:.4}'.format(saving_time))
      start = time.time()

    if ((ii+1)%100==0):

      x_full_batch = np.concatenate(x_full_batch)
      y_full_batch = np.concatenate(y_full_batch)

      np.save('../cifar10_data/train_fixed/x_train_fixed_{}.npy'.format(ii+1), x_full_batch)
      np.save('../cifar10_data/train_fixed/y_train_fixed_{}.npy'.format(ii+1), y_full_batch)

      x_full_batch = []
      y_full_batch = []
