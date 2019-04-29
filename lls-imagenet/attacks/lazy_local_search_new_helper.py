import cv2
import tensorflow as tf
import numpy as np
import heapq
import math
import sys
import time

SIZE = 299

class LazyLocalSearchNewHelper(object):

  def __init__(self, model, loss_func, epsilon, max_iters, **kwargs):
    # Hyperparameter setting 
    self.epsilon = epsilon
    self.max_iters = max_iters

    # Network setting
    self.x_input = model['x_input']
    self.y_input = model['y_input']
    self.logits = model['logits']
    self.preds = model['preds']
    self.targeted = model['targeted']

    probs = tf.nn.softmax(self.logits)
    batch_num = tf.range(0, limit=tf.shape(probs)[0])
    indices = tf.stack([batch_num, self.y_input], axis=1)
    ground_truth_probs = tf.gather_nd(params=probs, indices=indices)
    top_2 = tf.nn.top_k(probs, k=2)
    max_indices = tf.where(tf.equal(top_2.indices[:, 0], self.y_input), top_2.indices[:, 1], top_2.indices[:, 0])
    max_indices = tf.stack([batch_num, max_indices], axis=1)
    max_probs = tf.gather_nd(params=probs, indices=max_indices)
    
    if self.targeted:
      if loss_func == 'xent':
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.y_input)
      else:
        tf.logging.info('Loss function must be xent')
        sys.exit() 
    else:
      if loss_func == 'xent':
        self.losses = -tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.y_input)
      elif loss_func == 'cw':
        self.losses = tf.log(ground_truth_probs+1e-10) - tf.log(max_probs+1e-10)
      else:
        tf.logging.info('Loss function must be xent or cw')
        sys.exit() 
 
  def _perturb_image(self, image, noise):
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0., 1.)
    return adv_image 

  def _flip_noise(self, noise, block):
    noise_new = np.copy(noise)
    upper_left, lower_right, channel, _ = block 
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
    A = np.zeros([len(blocks)], np.int32)
    for i, block in enumerate(blocks):
      _, _, _, flipped = block
      # If flipped, set A to be 1.
      if flipped:
        A[i] = 1

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

    for _ in range(self.max_iters):
      # First forward passes
      indices,  = np.where(A==0)
      
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, self.width, self.height, 3], np.float32) 
        noise_batch = np.zeros([bend-bstart, 256, 256, 3], np.float32)
        label_batch = np.tile(label, bend-bstart)
         
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i:i+1, ...] = self._flip_noise(noise, blocks[idx])
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])
        
        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        
        # Early stopping 
        success_indices,  = np.where(preds == label) if self.targeted else np.where(preds != label)
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
      blocks[best_idx][3] = True
      
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
          blocks[cand_idx][3] = True
          # Early stopping
          if self.targeted:
            if preds == label:
              return noise, num_queries, curr_loss, True
          else:
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
        
        image_batch = np.zeros([bend-bstart, self.width, self.height, 3], np.float32) 
        noise_batch = np.zeros([bend-bstart, 256, 256, 3], np.float32)
        label_batch = np.tile(label, bend-bstart)
        
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i:i+1, ...] = self._flip_noise(noise, blocks[idx])
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])
        
        losses, preds = sess.run([self.losses, self.preds],
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        
        # Early stopping 
        success_indices,  = np.where(preds == label) if self.targeted else np.where(preds != label)
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
      blocks[best_idx][3] = False
      
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
          blocks[cand_idx] = False
          # Early stopping
          if self.targeted:
            if preds == label:
              return noise, num_queries, curr_loss, True
          else:
            if preds != label:
              return noise, num_queries, curr_loss, True
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))
      
      priority_queue = []
    
    return noise, num_queries, curr_loss, False

