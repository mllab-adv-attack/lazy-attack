import cv2
import collections
import tensorflow as tf
import numpy as np
import heapq
import math
import time
import itertools

from attacks.lazy_local_search_helper_new import LazyLocalSearchHelperNew

np.random.seed(0)
SubBlock = collections.namedtuple('SubBlock', 'upper_left, lower_right, xs, ys')

class LazyLocalSearchSplitAttack(object):
  def __init__(self, model, epsilon, max_queries=10000, **kwargs):
    # Setting
    self.max_queries = max_queries
    self.epsilon = epsilon
     
    # Load lazy local search helper
    self.lazy_local_search = LazyLocalSearchHelperNew(model, epsilon)
 
  def _perturb_image(self, image, noise):
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0, 1)
    return adv_image
  
  def _add_noise(self, noise, block, channel, direction):
    noise_new = np.copy(noise)
    xs = block.xs
    ys = block.ys
    for x, y in zip(xs, ys):
      noise_new[0, x, y, channel] = direction*self.epsilon 

    return noise_new

  def perturb(self, image, label, sess):		
    # Class variable
    self.width = image.shape[1]
    self.height = image.shape[2]

    # Local variables
    priority_queue = []
    num_queries = 0
    subblocks = []
    block_size = 32
    num_subblocks = 1

    for x in np.arange(0, 256, block_size):
      for y in np.arange(0, 256, block_size):
        upper_left = [x, y]
        lower_right = [x+block_size, y+block_size]
        xs, ys = np.where(np.zeros([block_size, block_size], np.int32)==0)
        xs += x
        ys += y
        #curr_order = np.random.permutation(len(xs))
        curr_order = np.arange(0, len(xs))
        subblock_size = len(xs)//num_subblocks
        for i in range(num_subblocks):
          start = i*subblock_size
          end = start+subblock_size
          xs_new = xs[curr_order[start:end]]
          ys_new = ys[curr_order[start:end]]
          subblocks.append(SubBlock(upper_left, lower_right, xs_new, ys_new))
    
    """    
    noise = np.zeros([1, 256, 256, 3], np.float32)
    for subblock in subblocks:
      for channel in range(3):
        direction = np.int32(np.sign(np.random.uniform(-1, 1)))
        noise = self._add_noise(noise, subblock, channel, direction)
    """
    noise = -np.ones([1, 256, 256, 3], np.float32)*self.epsilon
    batch_size = 32
    
    while True:
      num_batches = len(subblocks)//batch_size
      curr_order = np.random.permutation(len(subblocks))
      for i in range(num_batches):
        start = i*batch_size
        end = start+batch_size
        subblocks_batch = [subblocks[i] for i in curr_order[start:end]]
        noise, queries, loss, success = self.lazy_local_search.perturb(image, noise, label, sess, subblocks_batch)
        num_queries += queries
     
        tf.logging.info("batch: {}, loss: {}, num queries: {}".format(
          i, loss, num_queries))
     
        if num_queries > self.max_queries:
          return adv_image, num_queries
        adv_image = self._perturb_image(image, noise)
        if success:
          return adv_image, num_queries
      
      block_size //= 2
      subblocks_new = []
      for subblock in subblocks:
        upper_left = subblock.upper_left
        lower_right = subblock.lower_right
        xs = subblock.xs
        ys = subblock.ys
        count = 0
        for x in np.arange(upper_left[0], lower_right[0], block_size):
          for y in np.arange(upper_left[1], lower_right[1], block_size):
            upper_left_new = [x, y]
            lower_right_new = [x+block_size, y+block_size]
            xs_indices = np.where((xs>=x) & (xs<x+block_size))
            ys_indices = np.where((ys>=y) & (ys<y+block_size))
            indices = np.intersect1d(xs_indices[0], ys_indices[0])
            xs_new = xs[indices]
            ys_new = ys[indices]
            count += len(indices)
            if len(xs_new) > 0:
              subblocks_new.append(SubBlock(upper_left_new, lower_right_new, xs_new, ys_new))
      subblocks = subblocks_new
