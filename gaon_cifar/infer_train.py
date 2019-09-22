"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import sys
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

from model import Model
import cifar10_input

from infer_model import Model as Safe_model

with open('infer_config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
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

model = Model(mode='eval')
full_model = Safe_model('train', model, eps=40)

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
#total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
total_loss = full_model.mean_xent
total_acc = full_model.accuracy
train_step = tf.train.AdamOptimizer(learning_rate).minimize(
    total_loss,
    global_step=global_step)

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
infer_model_dir = config['infer_model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver()
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_input)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

model_file = tf.train.latest_checkpoint('models/' + model_dir)
if model_file is None:
    print('No model found')
    sys.exit()

with tf.Session() as sess:

    # initialize data augmentation
    cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

    # Initialize the summary writer, global variables, and our time counter.
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())

    # restore (partial)
    variables_can_be_restored = list(
        set(tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)).intersection(tf.train.list_variables(model_file)))

    temp_saver = tf.train.Saver(variables_can_be_restored)
    ckpt_state = tf.train.get_checkpoint_state(model_file)
    print('Loading checkpoint %s' % ckpt_state.model_checkpoint_path)
    temp_saver.restore(sess, ckpt_state.model_checkpoint_path)
    training_time = 0.0

    # Main training loop
    for ii in range(max_num_training_steps):
        x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                           multiple_passes=True)


        # Actual training step
        nat_dict = {full_model.x_input: x_batch,
                    full_model.y_input: y_batch}

        start = timer()
        _, loss, acc = \
            sess.run([train_step, total_loss, total_acc], feed_dict=nat_dict)
        end = timer()

        training_time += end - start

        # Output to stdout
        if ii % num_output_steps == 0:
            print('Step {}:    ({})'.format(ii, datetime.now()))
            print('    training accuracy {:.4}%'.format(acc * 100))
            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * batch_size / training_time))
                training_time = 0.0

        # Tensorboard summaries
        '''
        if ii % num_summary_steps == 0:
            summary = sess.run(merged_summaries, feed_dict=nat_dict)
            summary_writer.add_summary(summary, global_step.eval(sess))
        '''

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
            saver.save(sess,
                       os.path.join(infer_model_dir, 'checkpoint'),
                       global_step=global_step)

        training_time += end - start
