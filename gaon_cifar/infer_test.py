"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf

import cifar10_input
#from model import Model

#from cleverhans.model_zoo.madry_lab_challenges.cifar10_model import make_wresnet
from infer_model_base import Model

# Global constants
with open('config.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']
data_path = config['data_path']

model_dir = config['model_dir']

# Set upd the data, hyperparameters, and the model
cifar = cifar10_input.CIFAR10Data(data_path)

if eval_on_cpu:
  with tf.device("/cpu:0"):
    model = Model('eval')
else:
  model = Model('eval')

x_input = tf.placeholder(
    tf.float32,
    shape=[None, 32, 32, 3])
y_input = tf.placeholder(tf.int64, shape=None)
pre_softmax = model.fprop(x_input)
predictions = tf.argmax(pre_softmax, 1)
correct_prediction = tf.equal(predictions, y_input)
num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=pre_softmax, labels=y_input)
xent = tf.reduce_sum(y_xent)
mean_xent = tf.reduce_mean(y_xent)

global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(eval_dir)

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  with tf.Session() as sess:

    
    # Restore the checkpoint
    var_stored = [v[0] for v in tf.train.list_variables(filename)]
    var_to_restore = [n.name for n in tf.trainable_variables()]
    print(len(var_stored))
    print()
    print(len(var_to_restore))
    saver.restore(sess, filename)

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_corr_nat = 0

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]

      dict_nat = {x_input: x_batch,
                  y_input: y_batch}

      cur_corr_nat, cur_xent_nat = sess.run(
                                      [num_correct, xent],
                                      feed_dict = dict_nat)

      print(eval_batch_size)
      print("Correctly classified natural examples: {}".format(cur_corr_nat))
      total_xent_nat += cur_xent_nat
      total_corr_nat += cur_corr_nat

    avg_xent_nat = total_xent_nat / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples

    summary = tf.Summary(value=[
          tf.Summary.Value(tag='xent nat', simple_value= avg_xent_nat),
          tf.Summary.Value(tag='accuracy nat', simple_value= acc_nat)])
    summary_writer.add_summary(summary, global_step.eval(sess))

    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))

# Infinite eval loop
while True:
  cur_checkpoint = tf.train.latest_checkpoint(model_dir)

  # Case 1: No checkpoint yet
  if cur_checkpoint is None:
    if not already_seen_state:
      print('No checkpoint yet, waiting ...', end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
  # Case 2: Previously unseen checkpoint
  elif cur_checkpoint != last_checkpoint_filename:
    print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
                                                          datetime.now()))
    sys.stdout.flush()
    last_checkpoint_filename = cur_checkpoint
    already_seen_state = False
    evaluate_checkpoint(cur_checkpoint)
  # Case 3: Previously evaluated checkpoint
  else:
    if not already_seen_state:
      print('Waiting for the next checkpoint ...   ({})   '.format(
            datetime.now()),
            end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
