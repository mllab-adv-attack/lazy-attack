""" Borrowed from gaon's implementation """
import cv2
import itertools
import tensorflow as tf
import numpy as np
import heapq
import math
import sys
import time
import threading
import time


class LazyLocalSearchHelper(object):

  def __init__(self, model, sess, args, **kwargs):
    # Hyperparameter setting 
    self.epsilon = args.epsilon
    self.targeted = args.targeted
    self.admm = args.admm
    self.batch_size = args.batch_size
    self.merge_per_batch = args.merge_per_batch
   
    # Network setting
    self.model = model
    self.sess = sess
    
  # Perturb image with given noise   
  def _perturb_image(self, image, noise):
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0., 1.)
    return adv_image 

  # Block(image) splitting function
  def _split_block(self, upper_left, lower_right, block_size):
    blocks = []
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):
      for c in range(3):
        blocks.append([[x, y], [x+block_size, y+block_size], c])

    return blocks
  
  # Flip part of the noise (-eps <--> eps)
  def _flip_noise(self, noise, block):
    noise_new = np.copy(noise)
    upper_left, lower_right, channel = block 
    noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] *= -1
   
    return noise_new

  # Compute loss term involving (xi - Si z)
  def admm_loss(self, block, x, z, yk, rho):
    upper_left, lower_right = block
    dist = (x-z)[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :]
    
    return np.sum(np.multiply(yk, dist), axis=(1, 2, 3))+(rho/2)*np.sum(np.multiply(dist, dist), axis=(1, 2, 3))

  # Perturb an image within an ADMM block (iterate all batches)
  def perturb(self, 
              image, 
              prev_block_noise,
              noise, 
              label, 
              admm_block, 
              lls_block_size, 
              success_checker,
              yk,
              rho,
              index,
              results):		

    # Get the size of image
    self.width = np.shape(image)[1]
    self.height = np.shape(image)[2]

    # Split to lls blocks
    upper_left, lower_right = admm_block
    blocks = self._split_block(upper_left, lower_right, lls_block_size)
   
    # Initialize local noise
    block_noise = prev_block_noise

    # Copy global noise
    noise = np.copy(noise)
 
    # Initialize query count
    num_queries = 0

    # Random permute mini-batches
    num_blocks = len(blocks)
    if self.batch_size == 0:
      self.batch_size = num_blocks
    curr_order = np.random.permutation(num_blocks)
    num_batches = int(math.ceil(num_blocks/self.batch_size)) 
    
    # Perform mini-batch lls
    for i in range(num_batches):
      bstart = i*self.batch_size
      bend = min((i+1)*self.batch_size, num_blocks)
      blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]

      success_checker.inc_run()
      
      block_noise, queries, loss, success = self._perturb_one_batch(
        image, block_noise, noise, label, admm_block, blocks_batch, success_checker, yk, rho)
      
      num_queries += queries

      results[index] = [block_noise, num_queries, loss, admm_block, success]

      success_checker.dec_run()

      if success_checker.check():
        results[index] = [block_noise, num_queries, loss, admm_block, success]
        return

      if self.merge_per_batch:
        while success_checker.if_run():
          time.sleep(5)

        # Update global variable by averaging
        overlap_count = np.zeros_like(noise, np.float32)
        new_noise = np.zeros_like(noise, np.float32)

        for block_noise, _, _, block, _ in results:
          upper_left, lower_right = block
          new_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :] += \
            block_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :]
          overlap_count[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :] += \
            np.ones_like(block_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :], np.float32)

        noise = new_noise / overlap_count



    results[index] = [block_noise, num_queries, loss, admm_block, success]
    return
    
  def _perturb_one_batch(self,
                         image, 
                         block_noise,
                         noise,
                         label,
                         admm_block,
                         blocks_batch,
                         success_checker,
                         yk,
                         rho): 
   
    model = self.model
    sess = self.sess

    blocks = blocks_batch
    priority_queue = []
    num_queries = 0
    
    # Check if blocks are in the working set
    A = np.zeros([len(blocks)], np.int32)
    for i, block in enumerate(blocks):
      upper_left, _, channel = block
      x = upper_left[0]
      y = upper_left[1]
      # If flipped, set A to be 1.
      if block_noise[0, x, y, channel] > 0:
        A[i] = 1

    # Calculate current loss
    image_batch = self._perturb_image(image, block_noise)
    label_batch = np.copy(label)
    losses, preds = sess.run([model.losses, model.preds],
      feed_dict={model.x_input: image_batch, model.y_input: label_batch})
    num_queries += 1
    
    if self.admm:
      losses += self.admm_loss(admm_block, block_noise, noise, yk, rho)
    
    curr_loss = losses[0]

    # Early stopping
    if self.targeted:
      if preds == label:
        success_checker.set()
        return block_noise, num_queries, curr_loss, True
    else:
      if preds != label:
        success_checker.set()
        return block_noise, num_queries, curr_loss, True
   
    if success_checker.check():
      return block_noise, num_queries, curr_loss, False
   
    # Main loop
    for _ in range(1):
      # Lazy Greedy Insert
      indices,  = np.where(A==0)
      
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))
      
      for ibatch in range(num_batches):
        bstart = ibatch*batch_size
        bend = min(bstart+batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, self.width, self.height, 3], np.float32) 
        noise_batch = np.zeros([bend-bstart, 256, 256, 3], np.float32)
        label_batch = np.tile(label, bend-bstart)
         
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i:i+1, ...] = self._flip_noise(block_noise, blocks[idx])
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])
        
        losses, preds = sess.run([model.losses, model.preds], 
          feed_dict={model.x_input: image_batch, model.y_input: label_batch})
       
        if self.admm:
          losses += self.admm_loss(admm_block, noise_batch, noise, yk, rho)
        
        # Early stopping 
        success_indices,  = np.where(preds == label) if self.targeted else np.where(preds != label)
        if len(success_indices) > 0:
          block_noise[0, ...] = noise_batch[success_indices[0], ...]
          num_queries += success_indices[0] + 1
          curr_loss = losses[success_indices[0]]
          success_checker.set()
          return block_noise, num_queries, curr_loss, True

        if success_checker.check():
          return block_noise, num_queries, curr_loss, False
        
        num_queries += bend-bstart

        # Push into the priority queue
        for i in range(bend-bstart):
          idx = indices[bstart+i]
          margin = losses[i]-curr_loss
          heapq.heappush(priority_queue, (margin, idx))
      
      # Pick the best element and insert it into working set   
      if len(priority_queue) > 0:
        best_margin, best_idx = heapq.heappop(priority_queue)
        curr_loss += best_margin
        block_noise = self._flip_noise(block_noise, blocks[best_idx])
        A[best_idx] = 1
      
      # Add elements into working set
      while len(priority_queue) > 0:
        # Pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        # Re-evalulate the element
        image_batch = self._perturb_image(
          image, self._flip_noise(block_noise, blocks[cand_idx]))
        label_batch = np.copy(label)

        losses, preds = sess.run([model.losses, model.preds], 
          feed_dict={model.x_input: image_batch, model.y_input: label_batch})
        
        if self.admm:
          losses += self.admm_loss(admm_block, self._flip_noise(block_noise, blocks[cand_idx]), noise, yk, rho)
        
        num_queries += 1
        margin = losses[0]-curr_loss
       
        # If the cardinality has not changed, add the element
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin > 0:
            break
          # Update noise
          curr_loss = losses[0]
          block_noise = self._flip_noise(block_noise, blocks[cand_idx])
          A[cand_idx] = 1
          # Early stopping
          if self.targeted:
            if preds == label:
              success_checker.set()
              return block_noise, num_queries, curr_loss, True
          else:
            if preds != label:
              success_checker.set()
              return block_noise, num_queries, curr_loss, True
          
          if success_checker.check():
            return block_noise, num_queries, curr_loss, False

        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))
	    
      priority_queue = []
      
      # Lazy Greedy Delete
      indices,  = np.where(A==1)
       
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))   
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, self.width, self.height, 3], np.float32) 
        noise_batch = np.zeros([bend-bstart, 256, 256, 3], np.float32)
        label_batch = np.tile(label, bend-bstart)
        
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i:i+1, ...] = self._flip_noise(block_noise, blocks[idx])
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])
        
        losses, preds = sess.run([model.losses, model.preds],
          feed_dict={model.x_input: image_batch, model.y_input: label_batch})
       
        if self.admm:
          losses += self.admm_loss(admm_block, noise_batch, noise, yk, rho) 
           
        # Early stopping 
        success_indices,  = np.where(preds == label) if self.targeted else np.where(preds != label)
        if len(success_indices) > 0:
          block_noise[0, ...] = noise_batch[success_indices[0], ...]
          num_queries += success_indices[0] + 1
          curr_loss = losses[success_indices[0]]
          success_checker.set()
          return block_noise, num_queries, curr_loss, True 
       
        if success_checker.check():
          return block_noise, num_queries, curr_loss, False
           
        num_queries += bend-bstart

        # Push into the priority queue
        for i in range(bend-bstart):
          idx = indices[bstart+i]
          margin = losses[i]-curr_loss
          heapq.heappush(priority_queue, (margin, idx))

      # Pick the best element and remove it from working set   
      if len(priority_queue) > 0:
        best_margin, best_idx = heapq.heappop(priority_queue)
        curr_loss += best_margin
        noise = self._flip_noise(noise, blocks[best_idx])
        A[best_idx] = 0
      
      # Delete elements into working set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        # Re-evalulate the element
        image_batch = self._perturb_image(
          image, self._flip_noise(noise, blocks[cand_idx]))
        label_batch = np.copy(label)
        
        losses, preds = sess.run([model.losses, model.preds], 
          feed_dict={model.x_input: image_batch, model.y_input: label_batch})
       
        if self.admm:
          losses += self.admm_loss(admm_block, self._flip_noise(block_noise, blocks[cand_idx]), noise, yk, rho)
       
        num_queries += 1 
        margin = losses[0]-curr_loss
      
        # If the cardinality has not changed, remove the element
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin >= 0:
            break
          # Update noise
          curr_loss = losses[0]
          block_noise = self._flip_noise(block_noise, blocks[cand_idx])
          A[cand_idx] = 0
          # Early stopping
          if self.targeted:
            if preds == label:
              success_checker.set()
              return block_noise, num_queries, curr_loss, True
          else:
            if preds != label:
              success_checker.set()
              return block_noise, num_queries, curr_loss, True
         
          if success_checker.check():
            return block_noise, num_queries, curr_loss, False 
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))
      
      priority_queue = []
    
    return block_noise, num_queries, curr_loss, False

