import tensorflow as tf
import numpy as np
import heapq
import math
import time


class LazyLocalSearchAttack(object):
  def __init__(self, model, args, **kwargs):
    # Hyperparameter setting (general)
    self.epsilon = args.epsilon
    self.max_queries = args.max_queries
    self.loss_func = args.loss_func

    # Network setting
    self.x_input = model.x_input
    self.y_input = model.y_input
  
    logits = model.logits
    self.preds = model.predictions
    probs = tf.nn.softmax(logits)
    batch_num = tf.range(0, limit=tf.shape(probs)[0])
    indices = tf.stack([batch_num, self.y_input], axis=1)
    ground_truth_probs = tf.gather_nd(params=probs, indices=indices)
    top_2 = tf.nn.top_k(probs, k=2)
    max_indices = tf.where(tf.equal(top_2.indices[:, 0], self.y_input), 
      top_2.indices[:, 1], top_2.indices[:, 0])
    max_indices = tf.stack([batch_num, max_indices], axis=1)
    max_probs = tf.gather_nd(params=probs, indices=max_indices)
    
    if self.loss_func == 'xent':
      self.losses = -tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=self.y_input)
    elif self.loss_func == 'cw':
      self.losses = tf.log(ground_truth_probs+1e-10) - tf.log(max_probs+1e-10) 
    
  def _perturb_image(self, image, noise):
    adv_image = image + noise
    adv_image = np.clip(adv_image, 0., 255.)
    return adv_image   
     
  def _flip_noise(self, noise, w, h, c):
    noise_new = np.copy(noise)
    noise_new[0, w, h, c] *= -1
    return noise_new

  def perturb(self, image, label, sess):		
    # Local variable
    priority_queue = []
    num_queries = 0
    success = False
    step = 0
    
    adv_image = np.copy(image)
    A = np.zeros_like(image, np.int32)
    noise = -self.epsilon*np.ones_like(image, np.float32)

    # Calculate current loss
    image_batch = self._perturb_image(image, noise)
    label_batch = np.copy(label)
    losses, preds = sess.run([self.losses, self.preds], 
      feed_dict={self.x_input: image_batch, self.y_input: label_batch})
    num_queries += 1
    curr_loss = losses[0]

    if preds != label:
      adv_image = self._perturb_image(image, noise)
      return adv_image, num_queries, True
    
    while True:
      # First forward passes
      _, ws, hs, cs = np.where(A==0)
      
      batch_size = 100
      num_batches = int(math.ceil(len(ws)/batch_size))   
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(ws))
        
        image_batch = np.zeros([bend-bstart, 32, 32, 3], np.float32) 
        noise_batch = np.zeros([bend-bstart, 32, 32, 3], np.float32)
        label_batch = np.tile(label, bend-bstart)

        for i, (w, h, c) in enumerate(zip(ws[bstart:bend], hs[bstart:bend], cs[bstart:bend])):
          noise_batch[i:i+1, ...] = self._flip_noise(noise, w, h, c)
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])

        losses, preds = sess.run([self.losses, self.preds],
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        
        # Early stopping
        success_indices, = np.where(preds != label)
        if len(success_indices) > 0:
          noise = noise_batch[success_indices[0], ...]
          num_queries += success_indices[0]+1
          if num_queries > self.max_queries:
            return adv_image, num_queries, False
          adv_image = self._perturb_image(image, noise)
          return adv_image, num_queries, True
        num_queries += bend-bstart
        
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
      noise = self._flip_noise(noise, best_w, best_h, best_c)
      A[0, best_w, best_h, best_c] = 1
      
      # Add element into the set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_w, cand_h, cand_c = heapq.heappop(priority_queue)
       
        # Re-evalulate the element
        image_batch = self._perturb_image(
          image, self._flip_noise(noise, cand_w, cand_h, cand_c))
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
          noise = self._flip_noise(noise, cand_w, cand_h, cand_c)
          A[0, cand_w, cand_h, cand_c] = 1
          # Early stopping
          if preds != label:
            success = True
            break

        # If the cardinality has changed, push the element into the priority queue
        else: 
          heapq.heappush(priority_queue, (margin, cand_w, cand_h, cand_c))
	    
      tf.logging.info('Adding elements, step {}, loss: {:.4f}, num_queries: {}'.format(
        step, curr_loss, num_queries))

      if num_queries > self.max_queries:
        return adv_image, num_queries, False
      adv_image = self._perturb_image(image, noise)
      if success:
        return adv_image, num_queries, True
       
      priority_queue = []

      # Now delete element
      _, ws, hs, cs = np.where(A == 1)
       
      batch_size = 100
      num_batches = int(math.ceil(len(ws)/batch_size))   
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(ws))
        
        image_batch = np.zeros([bend-bstart, 32, 32, 3], np.float32) 
        noise_batch = np.zeros([bend-bstart, 32, 32, 3], np.float32)
        label_batch = np.tile(label, bend-bstart)

        for i, (w, h, c) in enumerate(zip(ws[bstart:bend], hs[bstart:bend], cs[bstart:bend])):
          noise_batch[i:i+1, ...] = self._flip_noise(noise, w, h, c)
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])

        losses, preds = sess.run([self.losses, self.preds],
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
       
        # Early stopping 
        success_indices, = np.where(preds != label)
        if len(success_indices) > 0:
          noise = noise_batch[success_indices[0], ...]
          num_queries += success_indices[0]+1
          if num_queries > self.max_queries:
            return adv_image, num_queries, False
          adv_image = self._perturb_image(image, noise)
          return adv_image, num_queries, True
        num_queries += bend-bstart
        
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
      noise = self._flip_noise(noise, best_w, best_h, best_c)
      A[0, best_w, best_h, best_c] = 0

      # Add element into the set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_w, cand_h, cand_c = heapq.heappop(priority_queue)
        
        # Re-evalulate the element
        image_batch = self._perturb_image(
          image, self._flip_noise(noise, cand_w, cand_h, cand_c))
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
          noise = self._flip_noise(noise, cand_w, cand_h, cand_c)
          A[0, cand_w, cand_h, cand_c] = 0
          # Early stopping
          if preds != label:
            success = True
            break

        # If the cardinality has changed, push the element into the priority queue
        else: 
          heapq.heappush(priority_queue, (margin, cand_w, cand_h, cand_c))
      
      tf.logging.info("Deleting elements, step {}, loss: {:.4f}, num_queries: {}".format(
        step, curr_loss, num_queries)) 
      if num_queries > self.max_queries:
        return adv_image, num_queries, False
      adv_image = self._perturb_image(image, noise)
      if success:
        return adv_image, num_queries, True

      priority_queue = []
      step += 1

