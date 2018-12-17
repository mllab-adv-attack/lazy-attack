import cv2
import tensorflow as tf
import numpy as np
import heapq
import math
import time

SIZE = 299

class LazyLocalSearchHelper(object):
  def __init__(self, model, epsilon, **kwargs):
    # Setting
    self.x_input = model['x_input']
    self.y_input = model['y_input']
    self.logits = model['logits']
    self.preds = model['preds']
    self.targeted = model['targeted']

    self.probs = tf.nn.softmax(self.logits)

    batch_nums = tf.range(0, limit=tf.shape(self.probs)[0])
    indices = tf.stack([batch_nums, self.y_input], axis=1)

    ground_truth_probs = tf.gather_nd(params=self.probs, indices=indices)
    top_2 = tf.nn.top_k(self.probs, k=2)
    max_indices = tf.where(tf.equal(top_2.indices[:, 0], self.y_input), top_2.indices[:, 1], top_2.indices[:, 0])
    max_indices = tf.stack([batch_nums, max_indices], axis=1)
    max_probs = tf.gather_nd(params=self.probs, indices=max_indices)
    self.losses = tf.log(ground_truth_probs) - tf.log(max_probs)
    if self.targeted:
        self.losses = -self.losses
    
    self.epsilon = epsilon
 
  def _perturb_image(self, image, noise):
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0, 1)
    return adv_image 

  def _flip_noise(self, noise, block, channel):
    noise_new = np.copy(noise)
    upper_left, lower_right = block 
    noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] *= -1
    return noise_new

  def perturb(self, image, noise, label, sess, blocks):		
    # Class variables
    self.width = image.shape[1]
    self.height = image.shape[2]
   
    # Local variables
    priority_queue = []
    num_queries = 0
    
    # Setting
    A = np.zeros([len(blocks), 3], np.int32)

    # Calculate current loss
    image_batch = self._perturb_image(image, noise)
    label_batch = np.copy(label)
    losses, preds = sess.run([self.losses, self.preds],
      feed_dict={self.x_input: image_batch, self.y_input: label_batch})
    num_queries += 1
    curr_loss = losses[0]
    
    if not self.targeted:
      if preds[0] != label:
        return noise, num_queries, curr_loss, True
    else:
      if preds[0] == label:
        return noise, num_queries, curr_loss, True

    for _ in range(2):
      # First forward passes
      indices, channels = np.where(A==0)
      
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, self.width, self.height, 3], np.float32) 
        label_batch = np.tile(label, bend-bstart)
        
        for i, (idx, c) in enumerate(zip(indices[bstart:bend], channels[bstart:bend])):
          image_batch[i:i+1, ...] = self._perturb_image(
            image, self._flip_noise(noise, blocks[idx], c))
        
        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        num_queries += bend-bstart
        
        if not self.targeted:
          success_indices,  = np.where(preds!=label)
        else:
          success_indices,  = np.where(preds==label)

        if len(success_indices) > 0:
          idx = indices[bstart+success_indices[0]]
          c = channels[bstart+success_indices[0]]
          noise = self._flip_noise(noise, blocks[idx], c)
          curr_loss = losses[success_indices[0]]
          return noise, num_queries, curr_loss, True

        # Push into the priority queue
        for i in range(bend-bstart):
          idx = indices[bstart+i]
          c = channels[bstart+i]
          margin = losses[i]-curr_loss
          heapq.heappush(priority_queue, (margin, idx, c))
      
      # Pick the best element and perturb the image   
      best_margin, best_idx, best_c = heapq.heappop(priority_queue)
      curr_loss += best_margin
      noise = self._flip_noise(noise, blocks[best_idx], best_c)
      A[best_idx, best_c] = 1
      
      # Add element into the set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_idx, cand_c = heapq.heappop(priority_queue)
        
        # Re-evalulate the element
        image_batch = self._perturb_image(
          image, self._flip_noise(noise, blocks[cand_idx], cand_c))
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
          noise = self._flip_noise(noise, blocks[cand_idx], cand_c)
          A[cand_idx, cand_c] = 1
          # Early stopping
          if not self.targeted:
            if preds[0] != label:
              return noise, num_queries, curr_loss, True
          else:
            if preds[0] == label:
              return noise, num_queries, curr_loss, True

        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx, cand_c))
	    
      priority_queue = []

      # Now delete element
      indices, channels = np.where(A==1)
       
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))   
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, self.width, self.height, 3], np.float32) 
        label_batch = np.tile(label, bend-bstart)
        
        for i, (idx, c) in enumerate(zip(indices[bstart:bend], channels[bstart:bend])):
          image_batch[i:i+1, ...] = self._perturb_image(
            image, self._flip_noise(noise, blocks[idx], c))
        
        losses, preds = sess.run([self.losses, self.preds],
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        num_queries += bend-bstart
        
        if not self.targeted:
          success_indices,  = np.where(preds!=label)
        else:
          success_indices,  = np.where(preds==label)
        
        if len(success_indices) > 0:
          idx = indices[bstart+success_indices[0]]
          c = channels[bstart+success_indices[0]]
          noise = self._flip_noise(noise, blocks[idx], c)
          curr_loss = losses[success_indices[0]]
          return noise, num_queries, curr_loss, True

        # Push into the priority queue
        for i in range(bend-bstart):
          idx = indices[bstart+i]
          c = channels[bstart+i]
          margin = losses[i] - curr_loss
          #heapq.heappush(priority_queue, (margin, idx))
          heapq.heappush(priority_queue, (margin, idx, c))
    
      # Pick the best element and perturb the image   
      best_margin, best_idx, best_c = heapq.heappop(priority_queue)
      curr_loss += best_margin
      noise = self._flip_noise(noise, blocks[best_idx], best_c)
      A[best_idx, best_c] = 0
      
      # Add element into the set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_idx, cand_c = heapq.heappop(priority_queue)
        # Re-evalulate the element
        image_batch = self._perturb_image(
          image, self._flip_noise(noise, blocks[cand_idx], cand_c))
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
          noise = self._flip_noise(noise, blocks[cand_idx], cand_c)
          A[cand_idx, cand_c] = 0
          # Early stopping
          if not self.targeted:
            if preds[0] != label:
              return noise, num_queries, curr_loss, True
          else:
            if preds[0] == label:
              return noise, num_queries, curr_loss, True
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx, cand_c))
      
      priority_queue = []
    
    return noise, num_queries, curr_loss, False  	
