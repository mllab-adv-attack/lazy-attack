import cv2
import tensorflow as tf
import numpy as np
import heapq
import math
import sys
import time

SIZE = 299

class LazyLocalSearchHelper(object):
  """Lazy Local Search Helper
  Note that since heapq library only supports min heap,
  we flip the sign of loss function.
  """

  def __init__(self, model, args, **kwargs):
    """Initalize lazy socal search helper
    
    Args:
      model: tensorflow model
      loss_func: str, the type of loss function
      epsilon: float, the maximum perturbation of pixel value
    """
    # Hyperparameter setting 
    self.epsilon = args.epsilon
    self.max_iters = args.max_iters
    self.targeted = args.targeted
    self.loss_func = args.loss_func
    
    # Network setting
    self.x_input = model['x_input']
    self.y_input = model['y_input']
    self.logits = model['logits']
    self.preds = model['preds']

    probs = tf.nn.softmax(self.logits)
    batch_num = tf.range(0, limit=tf.shape(probs)[0])
    indices = tf.stack([batch_num, self.y_input], axis=1)
    ground_truth_probs = tf.gather_nd(params=probs, indices=indices)
    top_2 = tf.nn.top_k(probs, k=2)
    max_indices = tf.where(tf.equal(top_2.indices[:, 0], self.y_input), top_2.indices[:, 1], top_2.indices[:, 0])
    max_indices = tf.stack([batch_num, max_indices], axis=1)
    max_probs = tf.gather_nd(params=probs, indices=max_indices)

    # Uncomment if you want to get a deterministic result    
    if self.targeted:
      if self.loss_func == 'xent':
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.y_input)
        #self.losses = -tf.log(ground_truth_probs+1e-10)
      elif self.loss_func == 'cw':
        self.losses = tf.log(max_probs+1e-10) - tf.log(ground_truth_probs+1e-10)
      else:
        tf.logging.info('Loss function must be xent or cw')
        sys.exit() 
    else:
      if self.loss_func == 'xent':
        self.losses = -tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.y_input)
        #self.losses = tf.log(ground_truth_probs+1e-10)
      elif self.loss_func == 'cw':
        self.losses = tf.log(ground_truth_probs+1e-10) - tf.log(max_probs+1e-10)
      else:
        tf.logging.info('Loss function must be xent or cw')
        sys.exit() 
 
  def _perturb_image(self, image, noise):
    """Perturb an image with a noise. First, resize the noise with the size of the image. Then add 
    the resized noise to the image. Finally, clip the value of the perturbed image into [0, 1]
    
    Args:
      image: numpy array with size [1, 299, 299, 3], original image
      noise: numpy array with size [1, 256, 256, 3]

    Returns:
      adv_iamge: numpy array with size [1, 299, 299, 3], perturbed image
    """ 
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0., 1.)
    return adv_image 

  def _flip_noise(self, noise, block):
    """Flip the sign of perturbation on a block
    Args:
      noise: numpy array with size [1, 256, 256, 3]
      block: [upper_left, lower_right, channel]
    
    Returns:
      noise: numpy array with size [1, 256, 256, 3], updated noise 
    """
    noise_new = np.copy(noise)
    upper_left, lower_right, channel = block 
    noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] *= -1
    return noise_new

  def perturb(self, image, noise, label, sess, blocks):		
    """Update noise with lazy local search
    
    Args:
      image: numpy array with size [1, 299, 299, 3], original image
      noise: numpy array with size [1, 256, 256, 3]
      label: numpy array with size [1], the label of image (or target label)
      sess: tensorflow session
      blocks: list, a set of blocks

    Returns: 
      noise: numpy array with size [1, 256, 256, 3], updated noise
      num_queries: int, query count
      curr_loss: float, the value of loss function
      success: boolean, True if attack is successful
    """
    # Class variables
    self.width = image.shape[1]
    self.height = image.shape[2]
   
    # Local variables
    priority_queue = []
    num_queries = 0
    
    # Check if blocks are in the working set
    A = np.zeros([len(blocks)], np.int32)
    for i, block in enumerate(blocks):
      upper_left, _, channel = block
      x = upper_left[0]
      y = upper_left[1]
      # If flipped, set A to be 1.
      if noise[0, x, y, channel] > 0:
        A[i] = 1

    # Calculate current loss
    image_batch = self._perturb_image(image, noise)
    label_batch = np.copy(label)
    losses, preds = sess.run([self.losses, self.preds],
      feed_dict={self.x_input: image_batch, self.y_input: label_batch})
    num_queries += 1
    curr_loss = losses[0]
    
    # Early stopping
    if self.targeted:
      if preds == label:
        return noise, num_queries, curr_loss, True
    else:
      if preds != label:
        return noise, num_queries, curr_loss, True
    
    # Main loop
    for _ in range(self.max_iters):
      # Lazy Greedy Insert
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
      
      # Pick the best element and insert it into working set   
      if len(priority_queue) > 0:
        best_margin, best_idx = heapq.heappop(priority_queue)
        if best_margin <= 0:
          curr_loss += best_margin
          noise = self._flip_noise(noise, blocks[best_idx])
          A[best_idx] = 1
      
      # Add elements into working set
      while len(priority_queue) > 0:
        # Pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        # Re-evalulate the element
        image_batch = self._perturb_image(
          image, self._flip_noise(noise, blocks[cand_idx]))
        label_batch = np.copy(label)

        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        num_queries += 1
        margin = losses[0]-curr_loss
        
        # If the cardinality has not changed, add the element
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin > 0:
            break
          # Update noise
          curr_loss = losses[0]
          noise = self._flip_noise(noise, blocks[cand_idx])
          A[cand_idx] = 1
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

      # Pick the best element and remove it from working set   
      if len(priority_queue) > 0:
        best_margin, best_idx = heapq.heappop(priority_queue)
        if best_margin < 0:
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
        
        losses, preds = sess.run([self.losses, self.preds], 
          feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        num_queries += 1 
        margin = losses[0]-curr_loss
      
        # If the cardinality has not changed, remove the element
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin >= 0:
            break
          # Update noise
          curr_loss = losses[0]
          noise = self._flip_noise(noise, blocks[cand_idx])
          A[cand_idx] = 0
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

