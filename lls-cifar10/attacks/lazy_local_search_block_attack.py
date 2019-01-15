import heapq
import itertools
import math
import numpy as np
import queue
import tensorflow as tf
import threading
import time

class LazyLocalSearchBlockAttack(object):
  def __init__(self, models, epsilon, loss_func, num_steps_outer=5, num_steps_inner=2, **kwargs):
    # Setting
    self.models = models
    self.model = models[0]
    self.epsilon = epsilon
    self.num_steps_outer = num_steps_outer
    self.num_steps_inner = num_steps_inner
    self.loss_func = loss_func
    
    # Global variables
    self.success = False
    self.lock = threading.Lock()
    
    # Construct blocks
    self.blocks = self.construct_blocks()
  
  def construct_blocks(self):
    blocks = []
    
    # block 0
    block_0 = np.zeros([1, 32, 32, 3], np.int32)
    #block_0[0, 0:16, 0:16, ...] = 1
    block_0[0, 0:8, 0:32, ...] = 1
    blocks.append(block_0)
    
    # block 1
    block_1 = np.zeros([1, 32, 32, 3], np.int32)
    #block_1[0, 0:16, 16:32, ...] = 1
    block_1[0, 8:24, 8:24, ...] = 1 
    blocks.append(block_1)
    
    # block 2
    block_2 = np.zeros([1, 32, 32, 3], np.int32)
    #block_2[0, 16:32, 0:16, ...] = 1
    block_2[0, 8:24, 0:8, ...] = 1
    block_2[0, 24:32, 0:16, ...] = 1
    blocks.append(block_2)
    
    # block 3
    block_3 = np.zeros([1, 32, 32, 3], np.int32)
    #block_3[0, 16:32, 16:32, ...] = 1
    block_3[0, 8:24, 24:32, ...] = 1
    block_3[0, 24:32, 16:32, ...] = 1
    blocks.append(block_3)
    """
    B = np.random.permutation(32*32*3)

    for i in range(32*32*3):
      w = B[i] // (32*3)
      h = (B[i] % (32*3)) // 3
      c = B[i] % 3
      blocks[i%4][0, w, h, c] = 1
    """
    return blocks
    
  def perturb(self, image, label, sesses):		
    adv_image = np.copy(image)
    sess = sesses[0]
    num_queries = 0
    self.success = False

    adv_image = np.copy(image)
    initial_attack = np.ndarray.astype(np.sign(np.random.uniform(-1, 1, adv_image.shape)), np.int32)
    adv_image += initial_attack * self.epsilon
    adv_image = np.clip(adv_image, 0, 255)

    # Run steps
    for step in range(self.num_steps_outer):
      start = time.time()
      threads = [None] * 4
      result_queue = queue.Queue()
      cw_loss = 0

      # Run blocks
      for block_idx in range(4):
        # Solve lazy greedy on the block
        block_image = np.copy(adv_image)
        threads[block_idx] = threading.Thread(target=self._lazy_local_search, args=(image, block_image, initial_attack, label, sesses, block_idx, result_queue)) 

      for block_idx in range(4):
        threads[block_idx].daemon = True
        threads[block_idx].start()
      
      for _ in range(4):
        block_image, queries, block_idx, update = result_queue.get()
        end = time.time()
        num_queries += queries
        if update:
          block = self.blocks[block_idx]
          adv_image = block * block_image + (1-block) * adv_image
      
      feed = {
        self.model.x_input: adv_image,
        self.model.y_input: label
      }
      cw_loss = sess.run(self.model.y_cw, feed)
      num_queries += 1
      end = time.time()

      tf.logging.info('Step {0}, Loss: {1}, Time taken: {2}'.format(step, cw_loss[0], end-start))
        
      if cw_loss[0] > 0:
        break

    return adv_image, num_queries

  def _lazy_local_search(self, image, adv_image, initial_attack, label, sesses, block_idx, result_queue):
    start = time.time()
    
    priority_queue = []
    num_queries = 0
    
    model = self.models[block_idx]
    sess = sesses[block_idx]
    block = self.blocks[block_idx]

    if self.loss_func == 'xent':
      loss = -model.y_xent
    elif self.loss_func == 'cw':
      loss = -model.y_cw
    else:
      tf.logging.info("Unknown loss function. Defaulting to cross-entropy")
      loss = -model.y_xent 
   
    # Evaluate the current loss
    feed = {
    	model.x_input: adv_image,
    	model.y_input: label,
    }
    curr_loss = sess.run(loss, feed)
    num_queries += 1
    
    # Get the current set
    initial_image = image + initial_attack * self.epsilon
    initial_image = np.clip(initial_image, 0, 255)
    A = np.ndarray.astype(np.not_equal(adv_image, initial_image), np.int32)
    A = np.logical_and(block, A)+block-1
   
    # Run lazy local search algorithm 
    for _ in range(self.num_steps_inner):
      # Add elementss
      _, ws, hs, cs = np.where(A==0) 
      batch_size = 100
      num_batches = int(math.ceil(len(ws)/batch_size))

      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(ws))

        image_batch = np.tile(adv_image, [bend-bstart, 1, 1, 1])
        label_batch = np.tile(label, bend-bstart)

        for i, (w, h, c) in enumerate(zip(ws[bstart:bend], hs[bstart:bend], cs[bstart:bend])):
          image_batch[i, w, h, c] = image[0, w, h, c] - initial_attack[0, w, h, c]*self.epsilon
        image_batch = np.clip(image_batch, 0, 255)
      
        feed = {
          model.x_input: image_batch,
          model.y_input: label_batch,
        }
        losses = sess.run(loss, feed)
        num_queries += bend-bstart
        
        # Push into the queue
        for i in range(bend-bstart):
          w = ws[bstart+i]
          h = hs[bstart+i]
          c = cs[bstart+i]
          margin = losses[i] - curr_loss
          heapq.heappush(priority_queue, (margin, w, h, c))
      
      # Pop the best element in the priority queue		
      best_margin, best_w, best_h, best_c = heapq.heappop(priority_queue)
      curr_loss += best_margin
      adv_image[0, best_w, best_h, best_c] = image[0, best_w, best_h, best_c] - initial_attack[0, best_w, best_h, best_c] * self.epsilon
      adv_image = np.clip(adv_image, 0, 255)
      A[0, best_w, best_h, best_c] = 1

      while len(priority_queue) > 0:
        # Pick a candidate
        cand_margin, cand_w, cand_h, cand_c = heapq.heappop(priority_queue)
        # Construct a batch
        image_batch = np.copy(adv_image)
        label_batch = np.copy(label)
        image_batch[0, cand_w, cand_h, cand_c] = image[0, cand_w, cand_h, cand_c] - initial_attack[0, cand_w, cand_h, cand_c]*self.epsilon
        image_batch = np.clip(image_batch, 0, 255)
        # Re-evaluate the margin of the candidate
        feed = {
      	  model.x_input: image_batch,
      	  model.y_input: label_batch,
        }
        losses, cw_losses = sess.run([loss, model.y_cw], feed)
        num_queries += 1
        margin = losses[0]-curr_loss

        # Check if the cardinality of the queue has been changed
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # Stop adding elements
          if margin >= 0:
            break

          # Perturb image
          curr_loss = losses[0]
          adv_image[0, cand_w, cand_h, cand_c] = image[0, cand_w, cand_h, cand_c] - initial_attack[0, cand_w, cand_h, cand_c] * self.epsilon
          adv_image = np.clip(adv_image, 0, 255)
          A[0, cand_w, cand_h, cand_c] = 1
          
          # Early stopping
          if cw_losses[0] > 0 or self.success:
            self.lock.acquire()
            result_queue.put((adv_image, num_queries, block_idx, not self.success))
            self.success = True
            self.lock.release()
            return

        else:
          heapq.heappush(priority_queue, (margin, cand_w, cand_h, cand_c))
      
      priority_queue = []
      
      # Delete elements
      _, ws, hs, cs = np.where(A==1)  
      batch_size = 100
      num_batches = int(math.ceil(len(ws)/batch_size))

      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(ws))

        image_batch = np.tile(adv_image, [bend-bstart, 1, 1, 1])
        label_batch = np.tile(label, bend-bstart)

        for i, (w, h, c) in enumerate(zip(ws[bstart:bend], hs[bstart:bend], cs[bstart:bend])):
          image_batch[i, w, h, c] = image[0, w, h, c] + initial_attack[0, w, h, c]*self.epsilon
        image_batch = np.clip(image_batch, 0, 255)
      
        feed = {
          model.x_input: image_batch,
          model.y_input: label_batch,
        }
        losses = sess.run(loss, feed)
        num_queries += bend-bstart
        
        # Push into the queue
        for i in range(bend-bstart):
          w = ws[bstart+i]
          h = hs[bstart+i]
          c = cs[bstart+i]
          margin = losses[i] - curr_loss
          heapq.heappush(priority_queue, (margin, w, h, c))
    
      # Pop the best element in the priority queue		
      best_margin, best_w, best_h, best_c = heapq.heappop(priority_queue)
      curr_loss += best_margin
      adv_image[0, best_w, best_h, best_c] = image[0, best_w, best_h, best_c] + initial_attack[0, best_w, best_h, best_c]*self.epsilon
      adv_image = np.clip(adv_image, 0, 255)
      A[0, best_w, best_h, best_c] = 0

      while len(priority_queue) > 0:
        # Pick a candidate
        cand_margin, cand_w, cand_h, cand_c = heapq.heappop(priority_queue)
        # Construct a batch
        image_batch = np.copy(adv_image)
        label_batch = np.copy(label)
        image_batch[0, cand_w, cand_h, cand_c] = image[0, cand_w, cand_h, cand_c] + initial_attack[0, cand_w, cand_h, cand_c]*self.epsilon
        image_batch = np.clip(image_batch, 0, 255)
        # Re-evaluate the margin of the candidate
        feed = {
      	  model.x_input: image_batch,
      	  model.y_input: label_batch,
        }
        losses, cw_losses = sess.run([loss, model.y_cw], feed)
        num_queries += 1
        margin = losses[0]-curr_loss
        
        # Check if the cardinality of the queue has been changed
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # Stop adding elements
          if margin >= 0:
            break
          
          # Perturb image
          curr_loss = losses[0]
          adv_image[0, cand_w, cand_h, cand_c] = image[0, cand_w, cand_h, cand_c] + initial_attack[0, cand_w, cand_h, cand_c]*self.epsilon
          adv_image = np.clip(adv_image, 0, 255)
          A[0, cand_w, cand_h, cand_c] = 0
          
          # Early stopping
          if cw_losses[0] > 0 or self.success:
            self.lock.acquire()
            result_queue.put((adv_image, num_queries, block_idx, not self.success))
            self.success = True
            self.lock.release()
            return

        else:
      	  heapq.heappush(priority_queue, (margin, cand_w, cand_h, cand_c))

      priority_queue = []
  
    result_queue.put((adv_image, num_queries, block_idx, True))
    return 
