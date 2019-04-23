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
import time

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import cifar10_input

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()

  parser.add_argument('--eps', default='8', help='Attack eps', type=int)
  parser.add_argument('--sample_size', default=100, help='sample size', type=int)
  parser.add_argument('--samples_per_batch', default=500, help='samples per batch', type=int)
  parser.add_argument('--loss_func', default='cw', help='loss func', type=str)
  parser.add_argument('--model_dir', default='adv_trained', help='model name', type=str)
  params = parser.parse_args()
  for key, val in vars(params).items():
    print('{}={}'.format(key,val))

class Submodular:
  def __init__(self, model, epsilon, loss_func):
    """Attack parameter initialization. The attack performs k steps of
    size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.loss_func = loss_func

    label_mask = tf.one_hot(model.y_input,
                            10,
                            on_value=1.0,
                            off_value=0.0,
                            dtype=tf.float32)
    if loss_func == 'xent':
      self.loss = model.y_xent
    elif loss_func == 'cw':
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax, axis=1)
      self.loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    elif loss_func == 'gt':
      self.loss = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
    elif loss_func == 'xent_v2':
      softmax = tf.nn.softmax(model.pre_softmax, axis=1)
      correct_predict = tf.reduce_sum(label_mask * softmax, axis=1)
      wrong_predict = tf.reduce_max((1-label_mask) * softmax, axis=1)
      self.loss = -tf.nn.relu(correct_predict - wrong_predict + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      self.loss = model.y_xent

    self.grad = tf.gradients(self.loss, model.x_input)[0]

  def test(self, x_nat, y, sess, ibatch):
    _, xt, yt, zt = x_nat.shape
    batch_size = 50
    assert params.samples_per_batch % batch_size == 0
    num_batches = int(math.ceil(params.samples_per_batch / batch_size))
    samples = [(xi, yi, zi) for xi in range(xt) for yi in range(yt) for zi in range(zt)]
    
    count = 0
    for i in range(num_batches):
      img_batch = np.tile(x_nat, (4*batch_size, 1, 1, 1))
      label_batch = np.tile(y, (4*batch_size))
      
      for j in range(batch_size):
        A_size = np.random.randint(len(samples))
        shuffled = np.random.permutation(samples)
        A = shuffled[:A_size]
        
        B_size = np.random.randint(len(samples))
        shuffled = np.random.permutation(samples)
        B = shuffled[:B_size]

        A = set([tuple(x) for x in A])
        B = set([tuple(x) for x in B])
        intersect = A & B
        union = A | B
        
        A_noise = np.ones_like(np.reshape(x_nat, (xt, yt, zt))) * (- params.eps)
        for tup in A:
          xi, yi, zi = tup
          A_noise[xi, yi, zi] *= -1
        img_batch[j] = np.clip(x_nat + A_noise, 0, 255)
        
        B_noise = np.ones_like(np.reshape(x_nat, (xt, yt, zt))) * (- params.eps)
        for tup in B:
          xi, yi, zi = tup
          B_noise[xi, yi, zi] *= -1
        img_batch[batch_size+j] = np.clip(x_nat + B_noise, 0, 255)

        intersect_noise = np.ones_like(np.reshape(x_nat, (xt, yt, zt))) * (- params.eps)
        for tup in intersect:
          xi, yi, zi = tup
          intersect_noise[xi, yi, zi] *= -1
        img_batch[batch_size*2+j] = np.clip(x_nat + intersect_noise, 0, 255)
        
        union_noise = np.ones_like(np.reshape(x_nat, (xt, yt, zt))) * (- params.eps)
        for tup in union:
          xi, yi, zi = tup
          union_noise[xi, yi, zi] *= -1
        img_batch[batch_size*3+j] = np.clip(x_nat + union_noise, 0, 255)
      
      losses = sess.run(self.loss, feed_dict={self.model.x_input: img_batch,
                                              self.model.y_input: label_batch})
      for j in range(batch_size):
        if losses[j] + losses[batch_size+j] >= losses[batch_size*2+j] + losses[batch_size*3+j]:
          count += 1
    
    return count / params.samples_per_batch

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
  tester = Submodular(model,
                      params.eps,
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
    eval_batch_size = 1
    #num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    x_adv = [] # adv accumulator

    bstart = 0
    while(True):
      x_candid = cifar.eval_data.xs[bstart:bstart+100]
      y_candid = cifar.eval_data.ys[bstart:bstart+100]
      mask = sess.run(model.correct_prediction, feed_dict = {model.x_input: x_candid,
                                                             model.y_input: y_candid})
      x_masked = x_candid[mask]
      y_masked = y_candid[mask]
      if bstart == 0:
        x_full_batch = x_masked[:min(num_eval_examples, len(x_masked))]
        y_full_batch = y_masked[:min(num_eval_examples, len(y_masked))]
      else:
        index = min(num_eval_examples-len(x_full_batch), len(x_masked))
        x_full_batch = np.concatenate((x_full_batch, x_masked[:index]))
        y_full_batch = np.concatenate((y_full_batch, y_masked[:index]))
      bstart += 100
      if len(x_full_batch) >= num_eval_examples:
        break

    percentages = []
    print('Iterating over {} batches\n'.format(num_batches))
    for ibatch in range(num_batches):
      print('attacking {}th image...'.format(ibatch))
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      x_batch = x_full_batch[bstart:bend, :]
      y_batch = y_full_batch[bstart:bend]

      start = time.time()

      percentage = tester.test(x_batch, y_batch, sess, ibatch)
      percentages.append(percentage)
      percentage_mean = sum(percentages) / len(percentages)
      print('submodularity percentage:{:.2f}% over {} samples'.format(percentage*100, params.samples_per_batch))
      print('percentage mean:{:.2f}%'.format(percentage_mean*100))
      end = time.time()
      print('time taken:{}'.format(end - start))
      print()
    
    print('eps:{}, loss function:{}'.format(params.eps, params.loss_func))
    print('percentage mean:{:.2f}%'.format(percentage_mean*100))
