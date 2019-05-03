import cv2
import itertools
import math
import numpy as np
import queue
import tensorflow as tf
import threading
import time

from attacks.lazy_local_search_helper import LazyLocalSearchHelper


class LazyLocalSearchAttack(object):
  """Lazy Local Search Attack with Hierarchical Method and Mini-batch Technique"""

  def __init__(self, models, sesses, args, **kwargs):
    """Initialize attack method.
    
    Args:
      model: tensorflow model
      args: arguments
    """
    # Hyperparameter setting
    self.targeted = args.targeted
    self.max_queries = args.max_queries
    self.epsilon = args.epsilon
    self.block_size = args.block_size
    self.max_iters = args.max_iters
    self.batch_size = args.batch_size
    self.num_steps = args.num_steps

    # Network setting
    self.model = models[0]
    self.sess = sesses[0]
    
    # Load lazy local search helper
    self.lazy_local_search = LazyLocalSearchHelper(models, sesses, args)

 
  def _perturb_image(self, image, noise):
    """Perturb an image with a noise. First, resize the noise with the size of the image. Then add 
    the resized noise to the image. Finally, clip the value of the perturbed image into [0, 1].
    
    Args:
      image: numpy array with size [1, 299, 299, 3], original image
      noise: numpy array with size [1, 256, 256, 3], noise
      
    Returns:
      adv_iamge: numpy array with size [1, 299, 299, 3], perturbed image   
    """
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0., 1.)
    return adv_image


  def _construct_blocks_list(self, upper_left, lower_right, block_size):
    """Construct the list of blocks. Note that each block consists of 
    [upper_left, lower_right, channel]
    
    Args:
      upper_left: [x, y], the coordinate of the upper left of an image
      lower_right: [x, y], the coordinate of the lower right of an image
      block_size: int, the size of a block

    Returns:
      blocks: list, the list of blocks
    """
    blocks_list = []
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)

    for i in range(4):
      blocks = []
      xs_block = xs[:len(xs)//2] if i%2 == 0 else xs[len(xs)//2:] 
      ys_block = ys[:len(ys)//2] if i//2 == 0 else ys[len(ys)//2:] 
      for x, y in itertools.product(xs_block, ys_block):
        for c in range(3):
          blocks.append([[x, y], [x+block_size, y+block_size], c])
      
      num_blocks = len(blocks)
      curr_order = np.random.permutation(num_blocks)
      blocks = [blocks[idx] for idx in curr_order] 
      
      blocks_list.append(blocks)
    
    return blocks_list

 
  def _update_noise(self, noise, block_noise, blocks):
    """ Update the noise vector
    
    Args:
      noise: a numpy array of size [1, 256, 256, 3], noise vector
      block_noise: a numpy array of size [1, 256, 256, 3], updated noise vector
      blocks: list, the location to be updated
    
    Returns:
      noise: a numpy array of size [1, 256, 256, 3]. noise vector
    """
    mask = np.zeros([1, 256, 256, 3], dtype=np.int32)

    for block in blocks:
      upper_left = block[0]
      lower_right = block[1]
      c = block[2]
      mask[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], c] = 1
    
    noise = mask*block_noise+(1-mask)*noise

    return noise

  
  def perturb(self, image, label, index):
    """Perturb an image with (or without) target label
    
    Args:
      image: a numpy array of size [1, 299, 299, 3], original image
      label: anumpy array of size [1], the label of image (or target label)
      index: int, the index of image
      sess: tensorflow session
       
    Returns:
      adv_image: numpy array with size [1, 299, 299, 3], adversarial image
      num_queries: int, query count
      success: boolean, True if attack is successful
    """
    # Set random seed by index for the reproducibility
    np.random.seed(index)

    # Class variables
    self.width = image.shape[1]
    self.height = image.shape[2]

    # Local variables
    adv_image = np.copy(image)
    upper_left = [0, 0]
    lower_right = [256, 256]
    block_size = self.block_size
    num_queries = 0
    step = 0
    success = False
      
    # Split image into the list of block groups   
    blocks_list = self._construct_blocks_list(upper_left, lower_right, block_size) 
    
    # Initialize noise
    noise = -self.epsilon*np.ones([1, 256, 256, 3], dtype=np.float32)
     
    losses, preds = self.sess.run(
      [self.model.losses, self.model.preds], 
      feed_dict={self.model.x_input: adv_image, self.model.y_input: label}
    )

    # Main loop
    while True:
      # Run batch
      batch_size = self.batch_size if self.batch_size > 0 else len(blocks_list[0])
      num_blocks = len(blocks_list[0])
      num_batches = int(math.ceil(num_blocks/batch_size)) 
     
      for batch in range(num_batches):
        bstart = batch*batch_size
        bend = min(bstart+batch_size, num_blocks)
        
        # Run the block groups in parallel
        threads = [None]*4
        result_queue = queue.Queue()
        noise_new_list = np.zeros([4, 256, 256, 3], np.float32)
        success_list = np.zeros([4], bool)
        loss_diff_list = np.zeros([4], np.float32)
        self.lazy_local_search.success = False
        
        start = time.time()
        
        blocks_total = []
                 
        for gpu in range(4):
          blocks = blocks_list[gpu][bstart:bend] 
          blocks_total += blocks
          threads[gpu] = threading.Thread(
            target=self.lazy_local_search.perturb,
            args=(image, np.copy(noise), label, blocks, gpu, result_queue)
          )
        
        for gpu in range(4):
          threads[gpu].daemon = True
          threads[gpu].start()

        for _ in range(4):
          noise_new, queries, curr_loss, gpu, success = result_queue.get()
          num_queries += queries
          noise_new_list[gpu] = noise_new
          success_list[gpu] = success
          loss_diff_list[gpu] = losses[0]-curr_loss if self.targeted else curr_loss-losses[0]
        
        end = time.time()

        # If all fail, update noise. Else, only update the noise of successful block.
        if np.sum(success_list) == 0: 
          for block_idx in range(4):
            noise_new = np.expand_dims(noise_new_list[block_idx, ...], axis=0)
            blocks = blocks_list[block_idx]
            noise = self._update_noise(noise, noise_new, blocks)
        else:
          success_indices = np.where(success_list==True)[0] 
          noise = np.expand_dims(noise_new_list[success_indices[0], ...], axis=0)

        # If query count exceeds max queries, then return False
        if num_queries > self.max_queries:
          return adv_image, num_queries, False
        
        # Update adversarial image
        adv_image = self._perturb_image(image, noise)
        
        # Calculate loss and prediction
        losses_old = np.copy(losses)
        losses, preds = self.sess.run(
          [self.model.losses, self.model.preds], 
          feed_dict={self.model.x_input: adv_image, self.model.y_input: label}
        )
        loss_diff = losses_old[0]-losses[0] if self.targeted else losses[0]-losses_old[0]

        # Logging
        tf.logging.info('block size: {}, step: {}, batch: {}, loss: {:.4f}, num queries: {}'.format(
          block_size, step, batch, losses[0], num_queries))
        #tf.logging.info('loss diff (all): {}, loss diff (blocks): {}'.format(loss_diff, np.sum(loss_diff_list)))
        tf.logging.info('time: {}'.format(end-start))  
        
        # Check if the attack is successful
        if self.targeted:
          if preds == label:
            success = True
        else:
          if preds != label:
            success = True
        
        # If succeeds, return True
        if success:
          return adv_image, num_queries, True
        
      step += 1
      
      if step == self.num_steps:
        block_size = block_size//2 if block_size > 1 else block_size
        blocks_list = self._construct_blocks_list(upper_left, lower_right, block_size) 
        step = 0
