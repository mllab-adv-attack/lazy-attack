import tensorflow as tf
import numpy as np
import heapq
import math
import sys
import itertools


# lls solver for an admm block
class LazyLocalSearchBlockHelper(object):

  def __init__(self, model, args, **kwargs):
    # Hyperparameter Setting
    self.epsilon = args.epsilon
    self.lls_iter = args.lls_iter
    self.targeted = args.targeted
    self.loss_func = args.loss_func
    self.admm = args.admm
    self.batch_size = args.batch_size

    # Network Setting
    self.model = model
    self.x_input = model.x_input
    self.y_input = model.y_input
    self.logits = self.model.logits
    self.preds = model.predictions

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
          logits=self.logits, labels=self.model.y_input)
        #self.losses = -tf.log(ground_truth_probs+1e-10)
      elif self.loss_func == 'cw':
        self.losses = tf.log(max_probs+1e-10) - tf.log(ground_truth_probs+1e-10)
      else:
        tf.logging.info('Loss function must be xent or cw')
        sys.exit()
    else:
      if self.loss_func == 'xent':
        self.losses = -tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.model.y_input)
        #self.losses = tf.log(ground_truth_probs+1e-10)
      elif self.loss_func == 'cw':
        self.losses = tf.log(ground_truth_probs+1e-10) - tf.log(max_probs+1e-10)
      else:
        tf.logging.info('Loss function must be xent or cw')
        sys.exit()

  # Perturb image with given noise
  def _perturb_image(self, image, noise):
    adv_image = image + noise
    adv_image = np.clip(adv_image, 0, 255)
    return adv_image

  # block(image) splitting function
  def _split_block(self, upper_left, lower_right, block_size):
    blocks = []
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):
      for c in range(3):
        blocks.append([[x, y], [x + block_size, y + block_size], c])
    return blocks

  # flip part of the noise (-eps <--> eps)
  def _flip_noise(self, noise, block):
    noise_new = np.copy(noise)
    upper_left, lower_right, channel = block
    noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] *= -1
    return noise_new

  # compute loss term involving (xi - Si z)
  def admm_loss(self, block, x, z, yk, rho):
    upper_left, lower_right = block
    dist = (x-z)[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :]

    return np.sum(np.multiply(yk, dist), axis=(1, 2, 3)) + (rho/2) * np.sum(np.multiply(dist, dist), axis=(1, 2, 3))

  # perturb an image within an admm block (iterate all batches)
  def perturb(self,
              image,
              prev_block_noise,
              noise,
              label,
              sess,
              admm_block,
              lls_block_size,
              success_checker,
              yk,
              rho,
              index,
              results):

    # split to lls blocks
    upper_left, lower_right = admm_block
    blocks = self._split_block(upper_left, lower_right, lls_block_size)

    # initialize local noise
    block_noise = prev_block_noise

    # initialize query count
    num_queries = 0

    # random permute mini-batches
    num_blocks = len(blocks)
    if self.batch_size == 0:
        self.batch_size = num_blocks
    curr_order = np.random.permutation(num_blocks)
    
    num_batches = int(math.ceil(num_blocks/self.batch_size))

    # perform mini-batch lls
    for i in range(num_batches):
      bstart = i*self.batch_size
      bend = min((i+1)*self.batch_size, num_blocks)
      blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]

      block_noise, queries, loss, success = self.perturb_one_batch(image,
                                                                   block_noise,
                                                                   noise,
                                                                   label,
                                                                   sess,
                                                                   admm_block,
                                                                   blocks_batch,
                                                                   success_checker,
                                                                   yk,
                                                                   rho)
      num_queries += queries
    
      if success_checker.check():
        results[index] = [block_noise, num_queries, loss, admm_block, success]
        return

    results[index] = [block_noise, num_queries, loss, admm_block, success]
    return

  # perturb an image within an admm block (one batch)
  # from lazy_local_search_helper.perturb
  def perturb_one_batch(self,
                        image,
                        block_noise,
                        noise,
                        label,
                        sess,
                        admm_block,
                        blocks_batch,
                        success_checker,
                        yk,
                        rho):
    # Set random seed by index for the reproducibility

    blocks = blocks_batch

    # Local variables
    priority_queue = []
    num_queries = 0

    # Check if block are in the working set
    A = np.zeros([len(blocks)], np.int32)
    for i, block in enumerate(blocks):
      upper_left, _, channel = block
      x = upper_left[0]
      y = upper_left[1]
      # If flipped, set A to be 1
      if block_noise[0, x, y, channel] > 0:
        A[i] = 1

    # Calculate current loss
    image_batch = self._perturb_image(image, block_noise)
    label_batch = np.copy(label)
    losses, preds = sess.run([self.losses, self.preds],
                             feed_dict={self.x_input: image_batch, self.y_input: label_batch})

    if self.admm:
      losses += self.admm_loss(admm_block, block_noise, noise, yk, rho)

    num_queries += 1
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
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))

        image_batch = np.zeros([bend-bstart, 32, 32, 3], np.int32)
        noise_batch = np.zeros([bend-bstart, 32, 32, 3], np.int32)
        label_batch = np.tile(label, bend-bstart)

        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i:i+1, ...] = self._flip_noise(block_noise, blocks[idx])
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])

        losses, preds = sess.run([self.losses, self.preds],
                                 feed_dict={self.x_input: image_batch, self.y_input: label_batch})

        if self.admm:
          losses += self.admm_loss(admm_block, noise_batch, noise, yk, rho)

        # Early stopping
        success_indices, = np.where(preds == label) if self.targeted else np.where(preds != label)
        if len(success_indices) > 0:
          block_noise[0, ...] = noise_batch[success_indices[0], ...]
          curr_loss = losses[success_indices[0]]
          num_queries += success_indices[0] + 1

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

        losses, preds = sess.run([self.losses, self.preds],
                                 feed_dict={self.x_input: image_batch, self.y_input: label_batch})

        if self.admm:
          losses += self.admm_loss(admm_block, self._flip_noise(block_noise, blocks[cand_idx]), noise, yk, rho)

        num_queries += 1
        margin = losses[0]-curr_loss

        # If the cardinality has not changed, add the element
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin > 0:
            break
          # Perturb image
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

        image_batch = np.zeros([bend-bstart, 32, 32, 3], np.int32)
        noise_batch = np.zeros([bend-bstart, 32, 32, 3], np.int32)
        label_batch = np.tile(label, bend-bstart)

        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i:i+1, ...] = self._flip_noise(block_noise, blocks[idx])
          image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])

        losses, preds = sess.run([self.losses, self.preds],
                                 feed_dict={self.x_input: image_batch, self.y_input: label_batch})

        if self.admm:
          losses += self.admm_loss(admm_block, noise_batch, noise, yk, rho)

        # Early stopping
        success_indices, = np.where(preds == label) if self.targeted else np.where(preds != label)
        if len(success_indices) > 0:
          block_noise[0, ...] = noise_batch[success_indices[0], ...]
          curr_loss = losses[success_indices[0]]
          num_queries += success_indices[0] + 1
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
        block_noise = self._flip_noise(block_noise, blocks[best_idx])
        A[best_idx] = 0

      # Delete elements from working set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        # Re-evalulate the element
        image_batch = self._perturb_image(
          image, self._flip_noise(block_noise, blocks[cand_idx]))
        label_batch = np.copy(label)

        losses, preds = sess.run([self.losses, self.preds],
                                 feed_dict={self.x_input: image_batch, self.y_input: label_batch})

        if self.admm:
          losses += self.admm_loss(admm_block, self._flip_noise(block_noise, blocks[cand_idx]), noise, yk, rho)

        num_queries += 1
        margin = losses[0]-curr_loss

        # If the cardinality has not changed, remove the element
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin > 0:
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

