import tensorflow as tf
import numpy as np
import heapq
import math
import sys

SIZE = 299

class LazyLocalSearchHelperL2(object):
  def __init__(self, model, loss_func, epsilon, **kwargs):
    # Hyperparameter setting 
    self.epsilon = epsilon
    self.weight = 1

    # Network setting
    self.x_input = model['x_input']
    self.y_input = model['y_input']
    self.noise = model['noise']
    self.x_adv = model['x_adv']
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

  def _add_noise(self, noise, block):
    noise_new = np.copy(noise)
    upper_left, lower_right, channel = block 
    noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] += self.weight
    return noise_new

  def _sub_noise(self, noise, block):
    noise_new = np.copy(noise)
    upper_left, lower_right, channel = block 
    noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] -= self.weight
    noise_new[noise_new < 0] = 0
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
      for channel in range(3):
        upper_left, _, channel = block
        x = upper_left[0]
        y = upper_left[1]
        # If flipped, set A to be 1.
        '''
        if noise[0, x, y, channel] > 0:
          A[i] = 1
        '''
        A[i] = noise[0, x, y, channel]

    # Calculate current loss

    label_batch = np.copy(label)
    losses, preds = sess.run([self.losses, self.preds],
      feed_dict={self.x_input: image, self.y_input: label_batch, self.noise: noise})
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
      indices,  = np.where(A>=0)
      
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, self.width, self.height, 3], np.float32) 
        label_batch = np.tile(label, bend-bstart)
        noise_batch = np.zeros([bend-bstart, 256, 256, 3], np.float32)
         
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i:i+1, ...] = self._add_noise(noise, blocks[idx])
          image_batch[i:i+1, ...] = image
        
        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: image_batch, self.y_input: label_batch, self.noise: noise_batch})
        
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
      noise = self._add_noise(noise, blocks[best_idx])
      A[best_idx] += self.weight
      
      heapq.heappush(priority_queue, (margin, best_idx))
      
      # Add element into the set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        # Re-evalulate the element
        label_batch = np.copy(label)

        losses, preds = sess.run([self.losses, self.preds], 
                                 feed_dict={self.x_input: image, self.y_input: label_batch, self.noise: self._add_noise(noise, blocks[cand_idx])})
        num_queries += 1
        margin = losses[0]-curr_loss
        
        # If the cardinality has not changed, perturb the image
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin >= 0:
            break
          # Perturb image
          curr_loss = losses[0]
          noise = self._add_noise(noise, blocks[cand_idx])
          A[cand_idx] += self.weight
          # Early stopping
          if self.targeted:
            if preds == label:
              return noise, num_queries, curr_loss, True
          else:
            if preds != label:
              return noise, num_queries, curr_loss, True

        # If the cardinality has changed, push the element into the priority queue
        '''
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))
        '''
        heapq.heappush(priority_queue, (margin, cand_idx))
	    
      priority_queue = []

      # Now delete element
      indices,  = np.where(A>0)
      #indices,  = np.where(A<=np.median(A))
       
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))   
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = np.zeros([bend-bstart, self.width, self.height, 3], np.float32)
        label_batch = np.tile(label, bend-bstart)
        noise_batch = np.zeros([bend-bstart, 256, 256, 3], np.float32)
        
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i:i+1, ...] = self._sub_noise(noise, blocks[idx])
          image_batch[i:i+1, ...] = image
        
        losses, preds = sess.run([self.losses, self.preds],
                                 feed_dict={self.x_input: image_batch, self.y_input: label_batch, self.noise: noise_batch})
        
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
      noise = self._sub_noise(noise, blocks[best_idx])
      A[best_idx] -= self.weight
      A[A<0] = 0
      
      heapq.heappush(priority_queue, (margin, best_idx))
      
      # Add element into the set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)

        if A[cand_idx] == 0:
          continue

        # Re-evalulate the element
        label_batch = np.copy(label)
        
        losses, preds = sess.run([self.losses, self.preds], 
                                 feed_dict={self.x_input: image, self.y_input: label_batch, self.noise: self._sub_noise(noise, blocks[cand_idx])})
        num_queries += 1 
        margin = losses[0]-curr_loss
      
        # If the cardinality has not changed, perturb the image
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin >= 0:
            break
          # Perturb image
          curr_loss = losses[0]
          noise = self._sub_noise(noise, blocks[cand_idx])
          A[cand_idx] -= self.weight
          A[A<0] = 0
          # Early stopping
          if self.targeted:
            if preds == label:
              return noise, num_queries, curr_loss, True
          else:
            if preds != label:
              return noise, num_queries, curr_loss, True
        # If the cardinality has changed, push the element into the priority queue
        '''
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))
        '''
        heapq.heappush(priority_queue, (margin, cand_idx))

      priority_queue = []
    
    return noise, num_queries, curr_loss, False
