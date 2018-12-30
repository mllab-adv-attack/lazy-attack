import numpy as np
import tensorflow as tf

from tools.inception_v3_imagenet import model
from tools.utils import *

IMAGENET_SL = 299

class BanditAttack(object):
  def __init__(self, args, sess):
    self.max_queries = args.max_queries
    self.epsilon = args.epsilon
    self.gradient_iters = args.gradient_iters
    self.batch_size = args.batch_size
    self.prior_size = args.tile_size
    self.exploration = args.exploration
    self.fd_eta = args.fd_eta
    self.online_lr = args.online_lr
    self.image_lr = args.image_lr

    self.x_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, IMAGENET_SL, IMAGENET_SL, 3])
    self.y_input = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
    logits, self.preds = model(sess, self.x_input)
    self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_input)

    self.prior = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.prior_size, self.prior_size, 3])
    dim = self.prior_size*self.prior_size*3
    exp_noise = self.exploration*tf.random_normal([self.batch_size, self.prior_size, self.prior_size, 3])/(dim**0.5)
    q1 = tf.image.resize_images(self.prior+exp_noise, [IMAGENET_SL, IMAGENET_SL], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    q2 = tf.image.resize_images(self.prior-exp_noise, [IMAGENET_SL, IMAGENET_SL], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    def norm(t):
      t = tf.reshape(t, [self.batch_size, -1])
      return tf.reshape(tf.norm(t, axis=1), [-1, 1, 1, 1])

    logit1, _ = model(sess, self.x_input+self.fd_eta*q1/norm(q1))
    logit2, _ = model(sess, self.x_input+self.fd_eta*q2/norm(q2))

    loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logit1, labels=self.y_input)
    loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logit2, labels=self.y_input)

    est_deriv = (loss1-loss2)/(self.fd_eta*self.exploration)
    est_grad = tf.reshape(est_deriv, [-1, 1, 1, 1])*exp_noise
    
    def eg_step(x, g, lr):
      real_x = (x+1)/2
      pos = real_x*tf.exp(lr*g)
      neg = (1-real_x)*tf.exp(-lr*g)
      new_x = pos/(pos+neg)
      return new_x*2-1

    self.prior_new = eg_step(self.prior, est_grad, self.online_lr)
    self.image_new = self.x_input+self.image_lr*tf.sign(tf.image.resize_images(self.prior_new, [IMAGENET_SL, IMAGENET_SL], tf.image.ResizeMethod.NEAREST_NEIGHBOR))
    
  def perturb(self, image, label, sess):
      total_queries = np.zeros([self.batch_size], np.int32)
      max_iters = self.max_queries // (2*self.gradient_iters) 
      adv_image = np.copy(image)
      upper = np.clip(image+self.epsilon, 0., 1.)
      lower = np.clip(image-self.epsilon, 0., 1.)

      prior = np.zeros([self.batch_size, self.prior_size, self.prior_size, 3], np.float32)
      not_dones_mask = np.ones([self.batch_size], np.int32)

      for i in range(max_iters):
        prior_new, image_new = sess.run([self.prior_new, self.image_new], 
          feed_dict={self.x_input: adv_image, self.y_input: label, self.prior: prior})
        image_new = np.clip(image_new, lower, upper)
        prior = np.reshape(not_dones_mask, [-1, 1, 1, 1])*prior_new + \
          (1-np.reshape(not_dones_mask, [-1, 1, 1, 1]))*prior
        adv_image = np.reshape(not_dones_mask, [-1, 1, 1, 1])*image_new + \
          (1-np.reshape(not_dones_mask, [-1, 1, 1, 1]))*adv_image
        total_queries += 2*self.gradient_iters*not_dones_mask
        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: adv_image, self.y_input: label})
        successes = np.not_equal(preds, label)
        not_dones_mask = (1-successes)*not_dones_mask
        
        num_successes = np.sum(1-not_dones_mask)
        average_queries = 0 if num_successes == 0 else np.sum((1-not_dones_mask)*total_queries)/num_successes
        tf.logging.info('Iter {}/{}, total success: {}/{}, average queries: {:.4f}'.format(
          i, max_iters, num_successes, len(not_dones_mask), average_queries)) 


        
    
    
    
    
    
    
    
    
      
  
   
