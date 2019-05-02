""" Borrowed from gaon's inplementation """
import copy
import cv2
import itertools
import math
import numpy as np
import queue
import tensorflow as tf
import threading
import time

from attacks.lazy_local_search_helper import LazyLocalSearchHelper

class SuccessChecker(object):
  def __init__(self, success=False):
    self.flag = success

  def set(self):
    self.flag = True

  def reset(self):
    self.flag = False

  def check(self):
    return self.flag


class LazyLocalSearchAttack(object):
  """Lazy Local Search Attack with ADMM"""

  def __init__(self, models, sesses, args, **kwargs):
    """Initialize attack method.
    
    Args:
      model: TensorFlow model
      args: arguments
    """
    # Basic setting
    self.targeted = args.targeted
    self.max_queries = args.max_queries
    self.epsilon = args.epsilon

    # Network setting
    self.model = models[0]
    self.sess = sesses[0]
    
    # ADMM setting
    self.admm_block_size = args.admm_block_size
    self.partition = args.partition
    self.admm_iter = args.admm_iter
    self.admm = args.admm
    self.overlap = args.overlap
    self.admm_rho = args.admm_rho
    self.admm_tau = args.admm_tau
    self.parallel = args.parallel

    # Local search setting
    self.lls_iter = args.lls_iter
    self.lls_block_size= args.lls_block_size
    self.batch_size = args.batch_size
    self.no_hier = args.no_hier

    # Set image size
    self.upper_left = [0, 0]
    self.lower_right = [256, 256]

    # Create success checker
    self.success_checker = SuccessChecker()

    # Construct blocks
    self.blocks = self._construct_blocks()

    # Build local search helper
    self.lazy_local_search = [LazyLocalSearchHelper(models[i%self.parallel], sesses[i%self.parallel], args) for i in range(len(self.blocks))]
    
  # Perturb image with given noise 
  def _perturb_image(self, image, noise):
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0., 1.)
  
    return adv_image

  # Calculate per-gpu queries
  def _parallel_queries(self, queries, non_parallel_queries):
    parallel_queries = (queries-non_parallel_queries)/self.parallel+non_parallel_queries

    return parallel_queries

  # Block(image) splitting function
  def _split_block(self, upper_left, lower_right, block_size, overlap):
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

  # Construct blocks for ADMM
  def _construct_blocks(self):
    block_size = self.admm_block_size
    overlap = self.overlap

    # Grid
    if self.partition == 'basic':
      blocks = self._split_block(self.upper_left, self.lower_right, block_size, overlap)

    # Centered
    elif self.partition == 'centered':
      print("unimplemented partition method!")
      raise Exception

    else:
      print("unimplemented partition method!")
      raise Exception

    return blocks

  # Perturb an image
  def perturb(self, image, label, index):
    # Wall clock time (total)
    total_time = 0
    total_start = time.time()
    
    # Set random seed by index for the reproducibility
    np.random.seed(index)

    # Store the size of an image
    self.width = image.shape[1]
    self.height = image.shape[2]

    # Local variables
    num_queries = 0
    non_parallel_queries = 0
    lls_block_size = self.lls_block_size
    results, prev_results = None, None
    
    # Initialize global noise
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
    self.success_checker.reset()
    
    # Run ADMM iterations
    for step in range(self.admm_iter):
      
      # Wall clock time (round)
      start = time.time()
      
      # Initialize threads
      threads = []
      prev_block_noises = [None]*len(self.blocks)
      results = [None]*len(self.blocks)
      result_queue = queue.Queue()
      
      for i in range(len(self.blocks)):    
        # Solve local search on a block
        if step == 0:
          prev_block_noises[i] = -self.epsilon*np.ones_like(noise, dtype=np.int32) 
        else:
          prev_block_noise, _, _, block, _ = prev_results[i]
          upper_left, lower_right = block
          
          # Initialize to (x; z\x)
          prev_block_noises[i] = np.copy(noise)
          prev_block_noises[i][:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :] = \
            prev_block_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :]
      
        threads.append(threading.Thread(
          target=self.lazy_local_search[i].perturb,
          args=(image, prev_block_noises[i], noise, label, self.blocks[i], lls_block_size, self.success_checker, yk[i], rho, i, result_queue)
        ))
 
      # Run threads
      num_running = 0     
      for i in range(len(self.blocks)):
        threads[i].daemon = True
        threads[i].start()
        num_running += 1 
      
        # If all gpus are used, wait for results
        if num_running == self.parallel:
          
          for _ in range(self.parallel):
            block_noise, block_queries, block_loss, block, block_success, i = result_queue.get()
            results[i] = (block_noise, block_queries, block_loss, block, block_success)
            num_queries += block_queries
            
            # Early stop checking
            if block_success:
              noise = block_noise
              curr_loss = block_loss
              adv_image = self._perturb_image(image, noise)
          
          # Max query checking
          parallel_queries = self._parallel_queries(num_queries, non_parallel_queries)

          if parallel_queries > self.max_queries:
            end = time.time()
            total_time = end-total_start
            tf.logging.info('Step {}, loss: {:.5f}, total queries: {}, per-gpu queries: {:.0f}, time taken: {:.2f}'.format(
              step, curr_loss, num_queries, parallel_queries, end-start))
            return adv_image, num_queries, parallel_queries, False, total_time

          # Early stop checking
          if self.success_checker.check():
            end = time.time()
            total_time = end-total_start
            tf.logging.info('Step {}, loss: {:.5f}, total queries: {}, per-gpu queries: {:.0f}, time taken: {:.2f}'.format(
              step, curr_loss, num_queries, parallel_queries, end-start))
            return adv_image, num_queries, parallel_queries, False, total_time
        
          num_running = 0
      
      # Update global variable by averaging
      overlap_count = np.zeros_like(noise, np.float32)
      new_noise = np.zeros_like(noise, np.float32)

      for block_noise, _, _, block, _ in results:
        upper_left, lower_right = block
        new_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :] += \
          block_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :]
        overlap_count[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :] += \
          np.ones_like(block_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :], np.float32)
      
      new_noise = new_noise/overlap_count

      # Check convergence (global)
      change_ratio = np.mean(new_noise != noise) 
      noise = new_noise
      
      # Update ADMM variables
      if self.admm:
        for i in range(len(self.blocks)):
          block_noise, _, _, block, _ = results[i]
          upper_left, lower_right = block
          dist = (block_noise-noise)[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :]
          yk[i] += rho*dist
        
        rho *= tau
      
      # Check early stop
      noise_threshold = np.where(noise==0, -1, noise)
      noise_threshold = self.epsilon*np.sign(noise_threshold)
      
      adv_image = self._perturb_image(image, noise_threshold)
      losses, preds = self.sess.run([self.model.losses, self.model.preds], feed_dict={
        self.model.x_input: adv_image,
        self.model.y_input: label
        })
      
      num_queries += 1
      non_parallel_queries += 1
      curr_loss = losses[0]

      # Max query checking
      parallel_queries = self._parallel_queries(num_queries, non_parallel_queries)
      
      # Save previous results for next round
      prev_results = copy.deepcopy(results)

      # Max query checking
      if parallel_queries > self.max_queries:
        end = time.time()
        total_time = end-total_start
        tf.logging.info('Step {}, loss: {:.5f}, total queries: {}, per-gpu queries: {:.0f}, change ratio: {:.4f}, time taken: {:.2f}'.format(
          step, curr_loss, num_queries, parallel_queries, change_ratio, end-start))
        return adv_image, num_queries, parallel_queries, False, total_time

      # Check early stop
      if self.targeted:
        if preds == label:
          end = time.time()
          total_time = end-total_start
          tf.logging.info('Step {}, loss: {:.5f}, total queries: {}, per-gpu queries: {:.0f}, change ratio: {:.4f}, time taken: {:.2f}'.format(
            step, curr_loss, num_queries, parallel_queries, change_ratio, end-start))
          return adv_image, num_queries, parallel_queries, True, total_time
      else:
        if preds != label:
          end = time.time()
          total_time = end-total_start
          tf.logging.info('Step {}, loss: {:.5f}, total queries: {}, per-gpu queries: {:.0f}, change ratio: {:.4f}, time taken: {:.2f}'.format(
            step, curr_loss, num_queries, parallel_queries, change_ratio, end-start))
          return adv_image, num_queries, parallel_queries, True, total_time
        
      end = time.time()
      tf.logging.info('Step {}, loss: {:.5f}, total queries: {}, per-gpu queries: {:.0f}, change ratio: {:.4f}, time taken: {:.2f}'.format(
        step, curr_loss, num_queries, parallel_queries, change_ratio, end-start))

      # Divide lls_block_size if hierarchical is used
      if not self.no_hier and ((step+1)%self.lls_iter == 0) and lls_block_size > 1:
        lls_block_size //= 2

    # Attack failed
    total_end = time.time()
    total_time = total_end-total_start
    return adv_image, num_queries, parallel_queries, False, total_time

