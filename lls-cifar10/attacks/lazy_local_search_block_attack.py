import itertools
import numpy as np
import tensorflow as tf
import threading
import time
import sys

from attacks.lazy_local_search_block_helper import LazyLocalSearchBlockHelper


# checker for checking if others thread successes
class SuccessChecker(object):
  def __init__(self, success=False):
    self.flag = success

  def set(self):
    self.flag = True

  def reset(self):
    self.flag = False

  def check(self):
    return self.flag


# lls-admm solver
class LazyLocalSearchBlockAttack(object):
  def __init__(self, models, args, **kwargs):
    # Setting
    self.models = models
    self.model = models[0]
    self.epsilon = args.epsilon
    self.admm_iter = args.admm_iter
    self.loss_func = args.loss_func
    self.targeted = args.targeted
    self.max_queries = args.max_queries

    self.admm_block_size = args.admm_block_size
    self.partition = args.partition
    self.admm_iter = args.admm_iter
    self.admm = args.admm
    self.overlap = args.overlap
    self.admm_rho = args.admm_rho
    self.admm_tau = args.admm_tau
    self.gpus = args.gpus

    self.lls_iter = args.lls_iter
    self.lls_block_size = args.lls_block_size
    self.batch_size = args.batch_size
    self.no_hier = args.no_hier

    # Set Image size
    self.upper_left = [0, 0]
    self.lower_right = [32, 32]

    # Global variables
    self.success_checker = SuccessChecker()

    # Construct blocks
    self.blocks = self.construct_blocks()

    # build lls helpers
    self.lazy_local_search = [LazyLocalSearchBlockHelper(models[i%self.gpus], args) for i in range(len(self.blocks))]

    # Network Setting
    self.x_input = self.model.x_input
    self.y_input = self.model.y_input
    self.logits = self.model.logits
    self.preds = self.model.predictions

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
        # self.losses = -tf.log(ground_truth_probs+1e-10)
      elif self.loss_func == 'cw':
        self.losses = tf.log(max_probs + 1e-10) - tf.log(ground_truth_probs + 1e-10)
      else:
        tf.logging.info('Loss function must be xent or cw')
        sys.exit()
    else:
      if self.loss_func == 'xent':
        self.losses = -tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.model.y_input)
        # self.losses = tf.log(ground_truth_probs+1e-10)
      elif self.loss_func == 'cw':
        self.losses = tf.log(ground_truth_probs + 1e-10) - tf.log(max_probs + 1e-10)
      else:
        tf.logging.info('Loss function must be xent or cw')
        sys.exit()

  # perturb image with given noise
  def _perturb_image(self, image, noise):
    adv_image = image + noise
    adv_image = np.clip(adv_image, 0, 255)
    return adv_image

  # block(image) splitting function
  def _split_block(self, upper_left, lower_right, block_size, overlap):
    blocks = []
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):
      for c in range(3):
        x0 = max(x-overlap, upper_left[0])
        y0 = max(y-overlap, upper_left[1])
        x1 = min(x+block_size+overlap, lower_right[0])
        y1 = min(y+block_size+overlap, lower_right[1])
        blocks.append([[x0, y0], [x1, y1], c])
    return blocks

  # Construct blocks for admm
  def construct_blocks(self):
    block_size = self.admm_block_size
    overlap = self.overlap

    # grid
    if self.partition == 'basic':
      blocks = self._split_block(self.upper_left, self.lower_right, block_size, overlap)

    # centered
    elif self.partition == 'centered':
      print("unimplemented partition method!")
      raise Exception

    else:
      print("unimplemented partition method!")
      raise Exception

    return blocks

  # perturb an image (full process)
  def perturb(self, image, label, index, sesses):

    total_time = 0

    # Set random seed
    np.random.seed(index)

    # Initialize query count, lls_block_size, main session
    num_queries = 0
    lls_block_size = self.lls_block_size
    sess = sesses[0]

    # Initialize noise
    noise = -self.epsilon * np.ones_like(image, dtype=np.int32)

    # Initialize admm variables
    yk = []
    for block in self.blocks:
      upper_left, lower_right, c = block
      yk_i = np.zeros((lower_right[0]-upper_left[0], lower_right[1]-upper_left[1]))
      yk.append(yk_i)

    rho = self.admm_rho
    tau = self.admm_tau

    # Initialize success flag
    self.success_checker.reset()

    # Run admm iterations
    for step in range(self.admm_iter):

      # Check wall clock time
      start = time.time()

      # Initialize threads
      threads = []
      results = [None] * len(self.blocks)

      for i in range(len(self.blocks)):
        # Solve lazy greedy on the block
        threads.append(threading.Thread(target=self.lazy_local_search[i].perturb, args=(
        image, noise, label, sesses[i%self.gpus], self.blocks[i], lls_block_size, self.success_checker, yk[i], rho, i, results)))

      # Run threads
      num_running = 0
      for i in range(len(self.blocks)):
        threads[i].start()
        num_running += 1

        # If all gpus are used, wait for results
        if num_running == self.gpus:

          for j in range(i-self.gpus+1, i+1):
            threads[j].join()

          # Gather results
          for j in range(i-self.gpus+1, i+1):
            block_noise, block_queries, block_loss, block, block_success = results[j]

            num_queries += block_queries

            # Early stop checking
            if block_success:
              noise = block_noise
              curr_loss = block_loss
              adv_image = self._perturb_image(image, noise)

          # Max query checking
          if num_queries > self.max_queries:
            end = time.time()
            total_time += (end-start)
            
            tf.logging.info('Step {}, Loss: {:.4f}, num queries: {}, Time taken: {:.4f}'.format(step, curr_loss, num_queries, end - start))
            return adv_image, num_queries, False, total_time

          # Early stop checking
          if self.success_checker.check():
            end = time.time()
            total_time += (end-start)
            
            tf.logging.info('Step {}, Loss: {:.4f} num queries: {}, Time taken: {:.4f}'.format(step, curr_loss, num_queries, end - start))
            return adv_image, num_queries, True, total_time

          num_running = 0

      # Update global variable by averaging
      overlap_count = np.zeros_like(noise)
      new_noise = np.zeros_like(noise)

      for block_noise, _, _, block, _ in results:

        upper_left, lower_right, c = block

        new_noise[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], c] += \
          block_noise[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], c]

        overlap_count[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], c] += \
          np.ones_like(block_noise[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], c])

      new_noise = new_noise / overlap_count
      noise = new_noise

      # Update admm variables
      if self.admm:

        for i in range(len(self.blocks)):
          block_noise, _, _, block, _ = results[i]

          upper_left, lower_right, c = block
          dist = (block_noise - noise)[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], c]

          yk[i] += rho * dist

        rho *= tau

      # Check early stop
      adv_image = self._perturb_image(image, noise)
      feed = {
        self.model.x_input: adv_image,
        self.model.y_input: label
      }
      losses, preds = sess.run([self.losses, self.preds],
                               feed_dict=feed)
      num_queries += 1

      curr_loss = losses[0]

      # Max queries checking
      if num_queries > self.max_queries:
        end = time.time()
        total_time += (end-start)
        
        tf.logging.info('Step {}, Loss: {:.4f}, num queries: {}, Time taken: {:.4f}'.format(step, curr_loss, num_queries, end - start))
        return adv_image, num_queries, False, total_time

      # Check early stop
      if self.targeted:
        if preds == label:
          end = time.time()
          total_time += (end-start)
          tf.logging.info('Step {}, Loss: {:.4f}, num queries: {}, Time taken: {:.4f}'.format(step, curr_loss, num_queries, end - start))
          return adv_image, num_queries, True, total_time
      else:
        if preds != label:
          end = time.time()
          total_time += (end-start)
          tf.logging.info('Step {}, Loss: {:.4f}, num queries: {}, Time taken: {:.4f}'.format(step, curr_loss, num_queries, end - start))
          return adv_image, num_queries, True, total_time

      end = time.time()

      total_time += (end-start)

      tf.logging.info('Step {}, Loss: {:.4f}, num queries: {}, Time taken: {:.4f}'.format(step, curr_loss, num_queries, end - start))

      # Divide lls_block_size if hierarchical is used
      if not self.no_hier and ((step+1)% self.lls_iter == 0) and lls_block_size > 1:
        lls_block_size //= 2

    # Attack failed
    return adv_image, num_queries, False, total_time

