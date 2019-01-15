import cv2
import tensorflow as tf
import numpy as np
import heapq
import math
import time


class LazyLocalSearchHelper(object):
  def __init__(self, model, loss_func, epsilon, max_iters, **kwargs):
    # Hyperparameter Setting
    self.epsilon = epsilon
    self.max_iters = max_iters
    
    # Network Setting
    self.model = model
    self.x_input = model.x_input
    self.y_input = model.y_input
    self.logits = self.model.logits
    self.losses = -tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=self.logits, labels=self.model.y_input)
    self.preds = model.predictions
     
  def _perturb_image(self, image, noise):
    adv_image = image + noise
    adv_image = np.clip(adv_image, 0, 255)
    return adv_image 

  def _flip_noise(self, noise, block):
    noise_new = np.copy(noise)
    upper_left, lower_right, channel = block 
    noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] *= -1
    return noise_new

  def perturb(self, image, noise, label, sess, blocks):		
    # Local variables
    priority_queue = []
    num_queries = 0
    
    # Setting
    A = np.zeros([len(blocks)], np.int32)
    for i, block in enumerate(blocks):
      upper_left, _, channel = block
      x = upper_left[0]
      y = upper_left[1]
      # If flipped, set A to be 1
      if noise[0, x, y, channel] > 0:
        A[i] = 1

    # Calculate current loss
    image_batch = self._perturb_image(image, noise)
    label_batch = np.copy(label)
    losses, preds = sess.run([self.losses, self.preds],
      feed_dict={self.x_input: image_batch, self.y_input: label_batch})
    num_queries += 1
    curr_loss = losses[0]
      
    if preds != label:
      return noise, num_queries, curr_loss, True

    for _ in range(self.max_iters):
      # First forward passes
      indices,  = np.where(A==0)
      
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, 32, 32, 3], np.int32) 
        noise_batch = np.zeros([bend-bstart, 32, 32, 3], np.int32)
        label_batch = np.tile(label, bend-bstart)
        
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i:i+1, ...] = self._flip_noise(noise, blocks[idx])
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])
        
        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        
        # Early stopping
        success_indices, = np.where(preds != label)
        if len(success_indices) > 0:
          noise[0, ...] = noise_batch[success_indices[0], ...]
          curr_loss = losses[success_indices[0]]
          num_queries += success_indices[0] + 1
          return noise, num_queries, curr_loss, True
        num_queries += bend-bstart
        
        # Push into the priority queue
        for i in range(bend-bstart):
          idx = indices[bstart+i]
          margin = losses[i]-curr_loss
          heapq.heappush(priority_queue, (margin, idx))
      
      # Pick the best element and perturb the image   
      best_margin, best_idx = heapq.heappop(priority_queue)
      curr_loss += best_margin
      noise = self._flip_noise(noise, blocks[best_idx])
      A[best_idx] = 1
      
      # Add element into the set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        
        # Re-evalulate the element
        image_batch = self._perturb_image(
          image, self._flip_noise(noise, blocks[cand_idx]))
        label_batch = np.copy(label)

        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        num_queries += 1
        margin = losses[0]-curr_loss
        
        # If the cardinality has not changed, perturb the image
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin >= 0:
            break
          # Perturb image
          curr_loss = losses[0]
          noise = self._flip_noise(noise, blocks[cand_idx])
          A[cand_idx] = 1
          # Early stopping
          if preds != label:
            return noise, num_queries, curr_loss, True
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))
	    
      priority_queue = []

      # Now delete element
      indices,  = np.where(A==1)
       
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))   
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, 32, 32, 3], np.int32) 
        noise_batch = np.zeros([bend-bstart, 32, 32, 3], np.int32)
        label_batch = np.tile(label, bend-bstart)
        
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i:i+1, ...] = self._flip_noise(noise, blocks[idx])
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])
        
        losses, preds = sess.run([self.losses, self.preds],
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        
        # Early stopping
        success_indices, = np.where(preds != label)
        if len(success_indices) > 0:
          noise[0, ...] = noise_batch[success_indices[0], ...]
          curr_loss = losses[success_indices[0]]
          num_queries += success_indices[0] + 1
          return noise, num_queries, curr_loss, True
        num_queries += bend-bstart
        
        # Push into the priority queue
        for i in range(bend-bstart):
          idx = indices[bstart+i]
          margin = losses[i]-curr_loss
          heapq.heappush(priority_queue, (margin, idx))
    
      # Pick the best element and perturb the image   
      best_margin, best_idx = heapq.heappop(priority_queue)
      curr_loss += best_margin
      noise = self._flip_noise(noise, blocks[best_idx])
      A[best_idx] = 0
      
      # Add element into the set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        # Re-evalulate the element
        image_batch = self._perturb_image(
          image, self._flip_noise(noise, blocks[cand_idx]))
        label_batch = np.copy(label)
        
        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        num_queries += 1 
        margin = losses[0]-curr_loss
      
        # If the cardinality has not changed, perturb the image
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin >= 0:
            break
          # Perturb image
          curr_loss = losses[0]
          noise = self._flip_noise(noise, blocks[cand_idx])
          A[cand_idx] = 0
          # Early stopping
          if preds != label:
            return noise, num_queries, curr_loss, True
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))
      
      priority_queue = []
    
    return noise, num_queries, curr_loss, False

