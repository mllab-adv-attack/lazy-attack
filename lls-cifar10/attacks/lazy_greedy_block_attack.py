import tensorflow as tf
import numpy as np
import heapq
import itertools
import math
import queue
import threading
import time

class LazyGreedyBlockAttack(object):
  def __init__(self, models, epsilon, loss_func, num_steps=5, **kwargs):
    # Setting
    self.models = models
    self.model = models[0]
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.loss_func = loss_func

    # Global variables
    self.success = False
    self.lock = threading.Lock()
    
    # Construct blocks
    self.blocks = self.construct_blocks()
  
  def construct_blocks(self):
    blocks = []
    
    # block 0
    block = np.zeros([1, 32, 32, 3], np.int32)
    block[0, 0:16, 0:16, ...] = 1
    blocks.append(block)
    
    # block 1
    block = np.zeros([1, 32, 32, 3], np.int32)
    block[0, 0:16, 16:32, ...] = 1
    blocks.append(block)
    
    # block 2
    block = np.zeros([1, 32, 32, 3], np.int32)
    block[0, 16:32, 0:16, ...] = 1
    blocks.append(block)
    
    # block 3
    block = np.zeros([1, 32, 32, 3], np.int32)
    block[0, 16:32, 16:32, ...] = 1
    blocks.append(block)

    
  def perturb(self, image, label, sesses):		
    adv_image = np.copy(image)
    num_queries = 0
    self.success = False

    # Run steps
    for step in range(self.num_steps):
      start = time.time()
      threads = [None] * 4
      result_queue = queue.Queue()

      # Run blocks
      for i in range(4):
        # Solve lazy greedy on the block
        threads[i] = threading.Thread(target=self._lazy_greedy_block, args=(image, adv_image, label, sesses, i, result_queue)) 

      for i in range(4):
        threads[i].daemon = True
        threads[i].start()
      
      for _ in range(4):
        block_image, queries, block_idx, update = result_queue.get()
        num_queries += queries
        if update:
          block = self.blocks[block_idx]
          adv_image = (1-block) * adv_image + block * block_image
                
      end = time.time()

      # Early stopping
      feed = {
        self.model.x_input: adv_image,
        self.model.y_input: label,
      }
      cw_loss = sess.run([self.model.y_cw], feed)
      num_queries += 1
      if cw_loss[0] > 0:
        break

    return adv_image, num_queries

  def _lazy_greedy_block(self, image, adv_image, label, sesses, block_idx, result_queue):
    
    tf.logging.info("block {0} starts".format(block_idx))
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
    
    # replace block with the original
    adv_image = (1-block) * adv_image + block * image

    # Evaluate the current loss
    feed = {
    	model.x_input: image,
    	model.y_input: label,
    }
    curr_loss = sess.run(loss, feed)
    num_queries += 1
    
    # First forward passes
    _, ws, hs, cs = np.where(block==1)  
    batch_size = 50
    num_batches = int(math.ceil(len(ws)/batch_size))

    for ibatch in range(num_batches):
      bstart = ibatch * batch_size
      bend = min(bstart + batch_size, len(ws))

      image_batch = np.tile(adv_image, [2*(bend-bstart), 1, 1, 1])
      label_batch = np.tile(label, bend-bstart)

      for i, (w, h, c) in enumerate(zip(ws[bstart:bend], hs[bstart:bend], cs[bstart:bend])):
        image_batch[2*i, w, h, c] -= self.epsilon
        image_batch[2*i+1, w, h, c] += self.epsilon
      image_batch = np.clip(image_batch, 0, 255)
      
      feed = {
        model.x_input: image_batch,
        model.y_input: label_batch,
      }
      losses = sess.run(loss, feed)
      num_queries += 2*(bend-bstart)

      # Push into the queue
      for i in range(bend-bstart):
        w = ws[bstart+i]
        h = hs[bstart+i]
        c = cs[bstart+i]
        margin_neg = losses[2*i] - curr_loss
        margin_pos = losses[2*i+1] - curr_loss
        margin = min(margin_neg, margin_pos)
        heapq.heappush(priority_queue, (margin, margin_neg, margin_pos, w, h, c))
    
    # Pop the best element in the priority queue		
    best_margin, best_margin_neg, best_margin_pos, best_w, best_h, best_c = heapq.heappop(priority_queue)
    curr_loss += best_margin
    adv_image[0, best_w, best_h, best_c] += -self.epsilon if best_margin_neg < best_margin_pos else self.epsilon
    adv_image = np.clip(adv_image, 0, 255)

    while len(priority_queue) > 0:
      # Pick a candidate
      cand_margin, cand_margin_neg, cand_margin_pos, cand_w, cand_h, cand_c = heapq.heappop(priority_queue)
      # Construct a batch
      image_batch = np.tile(adv_image, [2, 1, 1, 1])
      label_batch = np.tile(label, 2)
      image_batch[0, cand_w, cand_h, cand_c] -= self.epsilon
      image_batch[1, cand_w, cand_h, cand_c] += self.epsilon
      image_batch = np.clip(image_batch, 0, 255)
      # Re-evaluate the margin of the candidate
      feed = {
      	model.x_input: image_batch,
      	model.y_input: label_batch,
      }
      losses, cw_losses = sess.run([loss, model.y_cw], feed)
      num_queries += 2
      
      margin_neg = losses[0] - curr_loss
      margin_pos = losses[1] - curr_loss
      margin = min(margin_neg, margin_pos)
      
      # Check if the cardinality of the queue has been changed
      if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
        curr_loss += margin
        adv_image[0, cand_w, cand_h, cand_c] += -self.epsilon if margin_neg < margin_pos else self.epsilon
        adv_image = np.clip(adv_image, 0, 255)
        

      else:
      	heapq.heappush(priority_queue, (margin, margin_neg, margin_pos, cand_w, cand_h, cand_c))

    

