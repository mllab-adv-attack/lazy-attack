import cv2
import tensorflow as tf
import numpy as np
import heapq
import math
import time
import itertools

from attacks.lazy_local_search_helper import LazyLocalSearchHelper

np.random.seed(0)

class LazyLocalSearchBatchAttack(object):
  def __init__(self, model, epsilon, max_queries=10000, **kwargs):
    # Setting
    self.max_queries = max_queries
    self.epsilon = epsilon
     
    # Load lazy local search helper
    self.lazy_local_search = LazyLocalSearchHelper(model, epsilon)
 
  def _perturb_image(self, image, noise):
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0, 1)
    return adv_image

  def _split_block(self, block, block_size):
    blocks = []
    upper_left, lower_right = block
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):
      blocks.append([[x, y], [x+block_size, y+block_size]])
    return blocks
  
  def _add_noise(self, noise, block, channel, direction):
    noise_new = np.copy(noise)
    upper_left, lower_right = block
    noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] = direction*self.epsilon
    return noise_new

  def perturb(self, image, label, sess):		
    # Class variable
    self.width = image.shape[1]
    self.height = image.shape[2]

    # Local variables
    priority_queue = []
    num_queries = 0
    block_size = 32
    upper_left = [0, 0]
    lower_right = [256, 256]

    # Split image into blocks   
    blocks = self._split_block([upper_left, lower_right], block_size) 

    # Noise initialization
    noise = -self.epsilon*np.ones([1, 256, 256, 3], dtype=np.float32)

    # Variables
    num_blocks = len(blocks)
    batch_size = 32
    curr_order = np.random.permutation(num_blocks)

    # Main loop
    while block_size > 0:
      # Run batch
      num_batches = num_blocks // batch_size
      for i in range(num_batches):
        bstart = i*batch_size
        bend = bstart + batch_size
        blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]
        noise, queries, loss, success = self.lazy_local_search.perturb(image, noise, label, sess, blocks_batch)
        num_queries += queries
        tf.logging.info("Block size: {}, batch: {}, loss: {:.4f}, num queries: {}".format(
          block_size, i, loss, num_queries))
        if num_queries > self.max_queries:
          return adv_image, num_queries
        adv_image = self._perturb_image(image, noise)
        if success:
          return adv_image, num_queries
      
      # Create Next batch
      block_size //= 2
      if block_size <= 0:
        return adv_image, num_queries
      blocks = self._split_block([upper_left, lower_right], block_size)
      num_blocks = len(blocks)
      curr_order = np.random.permutation(num_blocks)

