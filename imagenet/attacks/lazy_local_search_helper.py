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

    probs = tf.nn.softmax(self.logits)

    batch_nums = tf.range(0, limit=tf.shape(probs)[0])
    indices = tf.stack([batch_nums, self.y_input], axis=1)

    ground_truth_probs = tf.gather_nd(params=probs, indices=indices)
    
    top_2 = tf.nn.top_k(probs, k=2)
    max_indices = tf.where(tf.equal(top_2.indices[:, 0], self.y_input), top_2.indices[:, 1], top_2.indices[:, 0])
    max_indices = tf.stack([batch_nums, max_indices], axis=1)
    max_probs = tf.gather_nd(params=probs, indices=max_indices)
    
    if self.targeted:
      self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.logits, labels=self.y_input)
    else:
      #self.losses = -tf.nn.sparse_softmax_cross_entropy_with_logits(
      #  logits=self.logits, labels=self.y_input)
      self.losses = tf.log(ground_truth_probs) - tf.log(max_probs)
    
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
    for i, block in enumerate(blocks):
      for channel in range(3):
        upper_left, _ = block
        x = upper_left[0]
        y = upper_left[1]
        # If flipped, set A to be 1.
        if noise[0, x, y, channel] > 0:
          A[i, channel] = 1

    # Calculate current loss
    image_batch = self._perturb_image(image, noise)
    label_batch = np.copy(label)
    losses, preds = sess.run([self.losses, self.preds],
      feed_dict={self.x_input: image_batch, self.y_input: label_batch})
    num_queries += 1
    curr_loss = losses[0]
    
    if self.targeted:
      if preds == label:
        return noise, num_queries, curr_loss, True
    else:
      if preds != label:
        return noise, num_queries, curr_loss, True

    for _ in range(1):
      # First forward passes
      indices, channels = np.where(A==0)
      
      batch_size = 1
      num_batches = int(math.ceil(len(indices)/batch_size))
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, self.width, self.height, 3], np.float32) 
        label_batch = np.tile(label, bend-bstart)
        noise_batch = np.zeros([bend-bstart, 256, 256, 3], np.float32)
         
        for i, (idx, c) in enumerate(zip(indices[bstart:bend], channels[bstart:bend])):
          noise_batch[i:i+1, ...] = self._flip_noise(noise, blocks[idx], c)
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])
        
        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        num_queries += bend-bstart
        
        # Early stopping 
        if (self.targeted and preds == label) or (not self.targeted and preds != label):
          return noise_batch, num_queries, losses[0], True

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
          if self.targeted:
            if preds == label:
              return noise, num_queries, curr_loss, True
          else:
            if preds != label:
              return noise, num_queries, curr_loss, True

        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx, cand_c))
	    
      priority_queue = []

      # Now delete element
      indices, channels = np.where(A==1)
       
      batch_size = 1
      num_batches = int(math.ceil(len(indices)/batch_size))   
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, self.width, self.height, 3], np.float32) 
        label_batch = np.tile(label, bend-bstart)
        noise_batch = np.zeros([bend-bstart, 256, 256, 3], np.float32)
        
        for i, (idx, c) in enumerate(zip(indices[bstart:bend], channels[bstart:bend])):
          noise_batch[i:i+1, ...] = self._flip_noise(noise, blocks[idx], c)
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])
        
        losses, preds = sess.run([self.losses, self.preds],
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        num_queries += bend-bstart
        
        # Early stopping 
        if (self.targeted and preds == label) or (not self.targeted and preds != label):
          return noise_batch, num_queries, losses[0], True

        # Push into the priority queue
        for i in range(bend-bstart):
          idx = indices[bstart+i]
          c = channels[bstart+i]
          margin = losses[i] - curr_loss
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
          if self.targeted:
            if preds == label:
              return noise, num_queries, curr_loss, True
          else:
            if preds != label:
              return noise, num_queries, curr_loss, True
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx, cand_c))
      
      priority_queue = []
    
    return noise, num_queries, curr_loss, False  	
