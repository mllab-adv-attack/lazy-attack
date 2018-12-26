import numpy as np
import tensorflow as tf

from tools.utils import *
from tools.inception_v3_imagenet import model


class NES_Attack(object):
  def __init__(self, sess, args):
    # Hyperparameter setting
    self.max_queries = args.max_queries
    self.epsilon = args.epsilon
    self.batch_size = args.batch_size
    self.sigma = args.sigma
    self.max_lr = args.max_lr
    self.min_lr = args.min_lr
    self.plateau_length = args.plateau_length
    self.plateau_drop = args.plateau_drop
    self.momentum = args.momentum

    self.targeted = 1 if args.targeted else -1
 
    # Network setting
    self.x_input = tf.placeholder(dtype=tf.float32, shape=[1, 299, 299, 3])
    self.y_input = tf.placeholder(dtype=tf.int32, shape=[1])
    self.logits, self.preds = model(sess, self.x_input)
    
    noise_pos = tf.random_normal([self.batch_size//2, 299, 299, 3])
    noise = tf.concat([noise_pos, -noise_pos], axis=0)
    image_batch = self.x_input + self.sigma*noise
    label_batch = tf.tile(self.y_input, [self.batch_size])

    logits, _ = model(sess, image_batch)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=label_batch)
    losses_tiled = tf.tile(tf.reshape(losses, [-1, 1, 1, 1]), [1, 299, 299, 3])
    self.grad_estimate = tf.reduce_mean(losses_tiled*noise, axis=0)/self.sigma
    self.loss = tf.reduce_mean(losses, axis=0)

  def perturb(self, image, label, sess):
    # Setting
    lower = np.clip(image-self.epsilon, 0., 1.)
    upper = np.clip(image+self.epsilon, 0., 1.)
    adv_image = np.copy(image)
   
    # Main loop 
    num_queries = 0
    last_losses = []
    lr = self.max_lr
    grad = 0
    i = 0
     
    while True:
      prev_grad = grad

      loss, grad = sess.run([self.loss, self.grad_estimate],
        feed_dict={self.x_input: adv_image, self.y_input: label}) 
      num_queries += self.batch_size
      grad = self.momentum*prev_grad + (1.0-self.momentum)*grad
      
      # Plateau lr annealing
      last_losses.append(loss)
      last_losses = last_losses[-self.plateau_length:]
      if self.targeted*last_losses[-1] > self.targeted*last_losses[0] and len(last_losses) == self.plateau_length:
        if lr > self.min_lr:
          lr = max(lr/self.plateau_drop, self.min_lr)
        last_losses = []
      
      if num_queries >= self.max_queries:
        return adv_image, num_queries, False

      adv_image = adv_image - self.targeted*lr*np.sign(grad)
      adv_image = np.clip(adv_image, lower, upper)

      preds = sess.run(self.preds, feed_dict={self.x_input: adv_image})
      num_queries += 1

      if self.targeted == 1:
        if preds == label:
          return adv_image, num_queries, True
      else:
        if preds != label:
          return adv_image, num_queries, True

      if i % 10 == 0: 
        tf.logging.info('Iter {}, lr: {:.5f}, loss: {:.4f}, num queries: {}'.format(
          i, lr, loss, num_queries))
      
      i += 1

