import cv2
import tensorflow as tf
import numpy as np
import time
import itertools

from attacks.lazy_local_search_helper_new import LazyLocalSearchHelperNew


class LazyLocalSearchBatchAttackNew(object):
  def __init__(self, model, args, **kwargs):
    # Setting
    self.loss_func = args.loss_func
    self.max_queries = args.max_queries
    self.epsilon = args.epsilon
    self.batch_size = args.batch_size

    # Load lazy local search helper
    self.lazy_local_search = LazyLocalSearchHelperNew(model, self.loss_func, self.epsilon)

  def l2_norm(self, noise):
    noise_centered = noise-np.mean(noise)
    l2_val = np.linalg.norm(noise_centered)
    if l2_val == 0:
        return noise_centered
    return self.epsilon * noise_centered / l2_val

  def _perturb_image(self, image, noise):
    adv_image = image + self.l2_norm(cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST))
    adv_image = np.clip(adv_image, 0, 1)
    return adv_image

  def _split_block(self, block, block_size):
    blocks = []
    upper_left, lower_right = block
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):
      for c in range(3):
        blocks.append([[x, y], [x+block_size, y+block_size], c])
    return blocks

  def perturb(self, image, label, index, sess):
    # Set random seed
    np.random.seed(index)

    self.lazy_local_search.weight = 1

    # Class variable
    self.width = image.shape[1]
    self.height = image.shape[2]

    # Local variables
    priority_queue = []
    num_queries = 0
    block_size = 16
    upper_left = [0, 0]
    lower_right = [256, 256]

    # Split image into blocks   
    blocks = self._split_block([upper_left, lower_right], block_size) 

    # Noise initialization
    noise = 100 * np.ones([1, 256, 256, 3], dtype=np.float32)

    # Variables
    num_blocks = len(blocks)
    batch_size = self.batch_size
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
          return adv_image, num_queries, False
        adv_image = self._perturb_image(image, noise)
        if success:
          return adv_image, num_queries, True
      
      self.lazy_local_search.weight += 0
      
      # Create Next batch
      if block_size >= 2:
        block_size //= 2
        blocks = self._split_block([upper_left, lower_right], block_size)
        num_blocks = len(blocks)
      
      curr_order = np.random.permutation(num_blocks)

