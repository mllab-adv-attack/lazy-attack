import tensorflow as tf
import numpy as np
import heapq
import math
import time

SIZE = 299

class LazyLocalSearchAttack(object):
  def __init__(self, model, epsilon, max_queries, block_size=8, **kwargs):
    # Setting
    self.x_input = model['x_input']
    self.y_input = model['y_input']
    self.logits = model['logits']
    self.preds = model['preds']
    self.probs = tf.nn.softmax(self.logits)
    
    batch_nums = tf.range(0, limit=tf.shape(self.logits)[0])
    indices = tf.stack([batch_nums, self.y_input], axis=1)
    
    self.ground_truth_probs = tf.gather_nd(params=self.probs, indices=indices)
    top_2 = tf.nn.top_k(self.probs, k=2)
    max_indices = tf.where(tf.equal(top_2.indices[:, 0], self.y_input), top_2.indices[:, 1], top_2.indices[:, 0])
    max_indices = tf.stack([batch_nums, max_indices], axis=1)
    max_probs = tf.gather_nd(params=self.probs, indices=max_indices)
    
    #self.losses = -tf.log(max_probs) + tf.log(self.ground_truth_probs)
    self.losses = -model['losses']
    
    self.epsilon = epsilon
    self.max_queries = max_queries
    self.block_size = block_size

  def _perturb_block(self, image, adv_image, w, h, c, direction):
    x = self.xs[w]
    y = self.ys[h]
    adv_image[0, x:x+self.block_size, y:y+self.block_size, c] = image[0, x:x+self.block_size, y:y+self.block_size, c] + direction * self.epsilon
    return adv_image 
  
  def perturb(self, image, label, sess):		
    priority_queue = []
    num_queries = 0
 
    self.xs = np.arange(0, image.shape[1], self.block_size)
    self.ys = np.arange(0, image.shape[2], self.block_size)
 
    adv_image = np.copy(image)
    initial_attack = np.ndarray.astype(np.sign(np.random.uniform(-1, 1, [1, len(self.xs), len(self.ys), 3])), np.int32)
    #initial_attack = -np.ones([1, len(self.xs), len(self.ys), 3], np.int32)
    for w in range(len(self.xs)):
      for h in range(len(self.ys)):
        for c in range(3):
          adv_image = self._perturb_block(image, adv_image, w, h, c, initial_attack[0, w, h, c])
    adv_image = np.clip(adv_image, 0, 1)

    A = np.zeros([1, len(self.xs), len(self.ys), 3], np.int32)

    for _ in range(5):
      # Calculate current loss
      feed = {
        self.x_input: adv_image,
        self.y_input: label,
      }
      curr_loss, pred = sess.run([self.losses, self.preds], feed)
      num_queries += 1

      if pred[0] != label:
        return adv_image, num_queries
      
      # First forward passes
      _, ws, hs, cs = np.where(A==0)
      
      batch_size = 100
      num_batches = int(math.ceil(len(ws)/batch_size))
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(ws))
        
        image_batch = np.tile(adv_image, [bend-bstart, 1, 1, 1]) 
        label_batch = np.tile(label, bend-bstart)

        for i, (w, h, c) in enumerate(zip(ws[bstart:bend], hs[bstart:bend], cs[bstart:bend])):
          image_batch[i:i+1, ...] = self._perturb_block(image, image_batch[i:i+1, ...], w, h, c, -initial_attack[0, w, h, c]) 
        image_batch = np.clip(image_batch, 0, 1)

        feed = {
          self.x_input: image_batch,
          self.y_input: label_batch
        }
        losses = sess.run(self.losses, feed)
        num_queries += bend-bstart
        
        if num_queries >= self.max_queries:
          return adv_image, num_queries
        
        # Push into the priority queue
        for i in range(bend-bstart):
          w = ws[bstart+i]
          h = hs[bstart+i]
          c = cs[bstart+i]
          margin = losses[i] - curr_loss
          heapq.heappush(priority_queue, (margin, w, h, c))
      
      # Pick the best element and perturb the image   
      best_margin, best_w, best_h, best_c = heapq.heappop(priority_queue)
      curr_loss += best_margin
      adv_image = self._perturb_block(image, adv_image, best_w, best_h, best_c, -initial_attack[0, best_w, best_h, best_c])
      adv_image = np.clip(adv_image, 0, 255)
      A[0, best_w, best_h, best_c] = 1
      
      # Add element into the set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_w, cand_h, cand_c = heapq.heappop(priority_queue)
       
        # Re-evalulate the element
        image_batch = np.copy(adv_image)
        label_batch = np.copy(label)
        image_batch[0:1, ...] = self._perturb_block(image, image_batch[0:1, ...], cand_w, cand_h, cand_c, -initial_attack[0, cand_w, cand_h, cand_c])
        image_batch = np.clip(image_batch, 0, 255)
        feed = {
          self.x_input: image_batch,
          self.y_input: label_batch
        }
        loss, pred = sess.run([self.losses, self.preds], feed)
        num_queries += 1
        margin = loss[0]-curr_loss
        
        if num_queries >= self.max_queries:
          return adv_image, num_queries
      
        # If the cardinality has not changed, perturb the image
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin >= 0:
            break
          # Perturb image
          curr_loss += margin
          adv_image = image_batch
          A[0, cand_w, cand_h, cand_c] = 1
          # Early stopping
          if pred[0] != label:
            return adv_image, num_queries

        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_w, cand_h, cand_c))
	    
      priority_queue = []

      # Now delete element
      _, ws, hs, cs = np.where(A == 1)
       
      batch_size = 100
      num_batches = int(math.ceil(len(ws)/batch_size))   
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(ws))
        
        image_batch = np.tile(adv_image, [bend-bstart, 1, 1, 1]) 
        label_batch = np.tile(label, bend-bstart)

        for i, (w, h, c) in enumerate(zip(ws[bstart:bend], hs[bstart:bend], cs[bstart:bend])):
          image_batch[i:i+1, ...] = self._perturb_block(image, image_batch[i:i+1, ...], w, h, c, initial_attack[0, w, h, c])
        image_batch = np.clip(image_batch, 0, 255)
        
        feed = {
          self.x_input: image_batch,
          self.y_input: label_batch
        }
        losses = sess.run(self.losses, feed)
        num_queries += bend-bstart

        if num_queries >= self.max_queries:
          return adv_image, num_queries
      
        # Push into the priority queue
        for i in range(bend-bstart):
          w = ws[bstart+i]
          h = hs[bstart+i]
          c = cs[bstart+i]
          margin = losses[i] - curr_loss
          heapq.heappush(priority_queue, (margin, w, h, c))
    
       # Pick the best element and perturb the image   
      best_margin, best_w, best_h, best_c = heapq.heappop(priority_queue)
      curr_loss += best_margin
      adv_image = self._perturb_block(image, adv_image, best_w, best_h, best_c, initial_attack[0, best_w, best_h, best_c])
      adv_image = np.clip(adv_image, 0, 255)
      A[0, best_w, best_h, best_c] = 0

      # Add element into the set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_w, cand_h, cand_c = heapq.heappop(priority_queue)
        # Re-evalulate the element
        image_batch = np.copy(adv_image)
        label_batch = np.copy(label)
        image_batch[0:1, ...] = self._perturb_block(image, image_batch[0:1, ...], cand_w, cand_h, cand_c, initial_attack[0, cand_w, cand_h, cand_c])
        image_batch = np.clip(image_batch, 0, 255)
        
        feed = {
          self.x_input: image_batch,
          self.y_input: label_batch
        }
        num_queries += 1
        loss, pred = sess.run([self.losses, self.preds], feed)
        margin = loss[0]-curr_loss
      
        if num_queries >= self.max_queries:
          return adv_image, num_queries

        # If the cardinality has not changed, perturb the image
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin >= 0:
            break
          # Perturb image
          curr_loss += margin
          adv_image = image_batch
          A[0, cand_w, cand_h, cand_c] = 0
          # Early stopping
          if pred[0] != label:
            return adv_image, num_queries
        
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_w, cand_h, cand_c))
      
      priority_queue = []
    
    return adv_image, num_queries  	
