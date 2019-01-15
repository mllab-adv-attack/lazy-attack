import cv2
import tensorflow as tf
import numpy as np
import heapq
import math
import time
import itertools

from attacks.lazy_local_search_helper import LazyLocalSearchHelper


class LazyLocalSearchBatchAttack(object):
  def __init__(self, model, args, **kwargs):
    # Setting
    self.loss_func = args.loss_func
    self.max_queries = args.max_queries
    self.epsilon = args.epsilon
    self.batch_size = args.batch_size
    self.block_size = args.block_size
    self.no_hier = args.no_hier
    self.max_iters = args.max_iters
         
    # Load lazy local search helper
    self.lazy_local_search = LazyLocalSearchHelper(model, self.loss_func, self.epsilon, self.max_iters)
 
  def _perturb_image(self, image, noise):
    adv_image = image + noise
    adv_image = np.clip(adv_image, 0, 255)
    return adv_image

  def _split_block(self, upper_left, lower_right, block_size):
    blocks = []
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):
      for c in range(3):
        blocks.append([[x, y], [x+block_size, y+block_size], c])
    return blocks
  
  def perturb(self, image, label, index, sess):		
    # Set random seed by index for the reproducibility
    np.random.seed(index)
    
    # Local variables
    num_queries = 0
    block_size = self.block_size
    upper_left = [0, 0]
    lower_right = [32, 32]
    
    # Split image into blocks
    blocks = self._split_block(upper_left, lower_right, block_size) 
    
    # Initialize noise
    noise = -self.epsilon*np.ones_like(image, dtype=np.int32)

    # Variables
    num_blocks = len(blocks)
    batch_size = self.batch_size if self.batch_size > 0 else num_blocks
    curr_order = np.random.permutation(num_blocks)

    while True:
      # Run batch
      num_batches = int(math.ceil(num_blocks/batch_size))
      for i in range(num_batches):
        bstart = i*batch_size
        bend = min(bstart + batch_size, num_blocks)
        blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]
        noise, queries, loss, success = self.lazy_local_search.perturb(
          image, noise, label, sess, blocks_batch)
        num_queries += queries
        tf.logging.info("Block size: {}, batch: {}, loss: {:.4f}, num queries: {}".format(
          block_size, i, loss, num_queries))
        if num_queries > self.max_queries:
          return adv_image, num_queries, False
        adv_image = self._perturb_image(image, noise)
        if success:
          return adv_image, num_queries, True
      
      if self.no_hier:
        return adv_image, num_queries, False
      
      # If blocks are splittable, split blocks
      if block_size >= 2:
        block_size //= 2
        blocks = self._split_block(upper_left, lower_right, block_size)
        num_blocks = len(blocks)
        batch_size = self.batch_size if self.batch_size > 0 else num_blocks
        curr_order = np.random.permutation(num_blocks)
      # Otherwise, shuffle the order of batches
      else:
        curr_order = np.random.permutation(num_blocks)
