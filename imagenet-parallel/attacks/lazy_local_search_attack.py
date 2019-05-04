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
  
  def __init__(self, models, sesses, args, **kwargs):
    # Hyperparameter setting
    self.targeted = args.targeted
    self.max_queries = args.max_queries
    self.epsilon = args.epsilon

    # Network setting
    self.model = models[0]
    self.sess = sesses[0]
    
    # ADMM setting
    self.admm = args.admm
    self.admm_block_size = args.admm_block_size
    self.admm_iter = args.admm_iter
    self.overlap = args.overlap
    self.admm_rho = args.admm_rho
    self.admm_tau = args.admm_tau
    self.num_steps = args.num_steps

    # Local search setting
    self.lls_block_size = args.lls_block_size
    self.batch_size = args.batch_size
    
    # Set noise size
    self.upper_left = [0, 0]
    self.lower_right = [256, 256]

    # Construct blocks
    self.blocks = self._construct_blocks()

    # Load lazy local search helper
    self.lazy_local_search = LazyLocalSearchHelper(models, sesses, args)

 
  def _perturb_image(self, image, noise):
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0., 1.)
    return adv_image

  def _construct_blocks(self, upper_left, lower_right, block_size, overlap):
    blocks = []
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):
      x0 = max(x-overlap, upper_left[0])
      y0 = max(y-overlap, upper_left[1])
      x1 = min(x+block_size+overlap, lower_right[0])
      y1 = min(y+block_size+overlap, lower_right[1])
      blocks.append([[x0, y0], [x1, y1]])

    return blocks
  
  def _split_block(self, block, block_size):
    blocks = []
    upper_left, lower_right = block
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):
      for c in range(3):
        blocks.append([[x, y], [x+block_size, y+block_size], c])
   
    num_blocks = len(blocks)
    curr_order = np.random.permutation(num_blocks)
    blocks = [blocks[idx] for idx in curr_order]
    
    return blocks

  def perturb(self, image, label, index):
    # Set random seed by index for the reproducibility
    np.random.seed(index)

    # Store the size of an image
    self.width = image.shape[1]
    self.height = image.shape[2]

    # Local variables
    num_queries = 0
    lls_block_size = self.lls_block_size
    results = [None]*4
    prev_results = [None]*4
     
    # Initialize noise
    noise = np.zeros([1, 256, 256, 3], dtype=np.float32)
    
    # Initialize ADMM variables
    yk = []
    for block in self.blocks:
      upper_left, lower_right = block
      yk_i = np.zeros([lower_right[0]-upper_left[0], lower_right[1]-upper_left[1], 3], np.float32)  
      yk.append(yk_i) 
    
    rho = self.admm_rho
    tau = self.admm_tau
  
    # Initialize success flag
    self.lazy_local_search.success = False
    
    # Main loop
    while True:
      subblocks_list = [self._split_block(block) for block in self.blocks] 
      
      # Run step
      for step in range(num_steps)
        batch_size = self.batch_size if self.batch_size > 0 else len(blocks_list[0])
        num_blocks = len(blocks_list[0])
        num_batches = int(math.ceil(num_blocks/batch_size)) 
        
        # Run batch 
        for batch in range(num_batches):
          bstart = batch*batch_size
          bend = min(bstart+batch_size, num_blocks)
          
          # Run the block groups in parallel
          threads = [None]*4
          results = [None]*4
          result_queue = queue.Queue()
           
          for gpu in range(4):
            if prev_results[gpu] is None:
              block_noise = -self.epsilon*np.ones_like(noise, dtype=np.float32)  
            else:
              prev_block_noise, _, _, block, _ = prev_results[gpu]
              upper_left, lower_right = block
              block_noise = np.copy(noise)
              block_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :] = \
                prev_block_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :]
            blocks = subblock_list[gpu][bstart:bend]
            y = yk[gpu]
            threads[gpu] = threading.Thread(
              target=self.lazy_local_search.perturb,
              args=(image, block_noise, noise, label, blocks, y, rho, gpu, result_queue)
            )
        
          for gpu in range(4):
            threads[gpu].daemon = True
            threads[gpu].start()

          for _ in range(4):
            block_noise, block_queries, block_loss, block_success, gpu = result_queue.get()
            
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
