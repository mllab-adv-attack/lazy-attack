import tensorflow as tf
import numpy as np
import itertools

from attacks.lazy_local_search_helper_l2 import LazyLocalSearchHelperL2


class LazyLocalSearchBatchAttackL2(object):
  def __init__(self, model, args, **kwargs):
    # Setting
    self.loss_func = args.loss_func
    self.max_queries = args.max_queries
    self.epsilon = args.epsilon
    self.batch_size = args.batch_size

    # Load lazy local search helper
    self.lazy_local_search = LazyLocalSearchHelperL2(model, self.loss_func, self.epsilon)

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
    num_queries = 0
    block_size = 4
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
      batch_size = self.batch_size
      num_batches = num_blocks // batch_size
      while num_batches == 0:
        batch_size = batch_size // 2
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
        adv_image = sess.run(self.lazy_local_search.x_adv,
                            feed_dict={self.lazy_local_search.x_input: image, self.lazy_local_search.y_input: label, self.lazy_local_search.noise: noise})
        if success:
          return adv_image, num_queries, True
      
      self.lazy_local_search.weight += 0
      
      # Create Next batch
      if block_size >= 2:
        block_size //= 2
        blocks = self._split_block([upper_left, lower_right], block_size)
        num_blocks = len(blocks)
      
      curr_order = np.random.permutation(num_blocks)

