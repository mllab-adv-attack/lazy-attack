import itertools
import numpy as np
import tensorflow as tf
import threading
import time
import sys
import copy

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
    # basic settings
    self.models = models
    self.model = models[0]
    self.epsilon = args.epsilon
    self.loss_func = args.loss_func
    self.targeted = args.targeted
    self.max_queries = args.max_queries

    # admm settings
    self.admm_block_size = args.admm_block_size
    self.partition = args.partition
    self.admm_iter = args.admm_iter
    self.admm = args.admm
    self.overlap = args.overlap
    self.admm_rho = args.admm_rho
    self.admm_tau = args.admm_tau
    self.adam = args.adam
    self.adam_adapt = args.adam_adapt
    self.parallel = args.parallel
    self.merge_per_batch = args.merge_per_batch

    # lazy local search settings
    self.lls_iter = args.lls_iter
    self.lls_block_size = args.lls_block_size
    self.no_hier = args.no_hier

    # Set Image size
    self.upper_left = [0, 0]
    self.lower_right = [32, 32]

    # Global variables
    self.success_checker = SuccessChecker()

    # Construct blocks
    self.blocks = self.construct_blocks()

    # build lls helpers
    self.lazy_local_search = [LazyLocalSearchBlockHelper(models[i%self.parallel], args)
                              for i in range(len(self.blocks))]

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

  # calculate per-gpu queries
  def _parallel_queries(self, queries, non_parallel_queries):

    parallel_queries = (queries-non_parallel_queries)/self.parallel + non_parallel_queries

    return parallel_queries

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
      x0 = max(x-overlap, upper_left[0])
      y0 = max(y-overlap, upper_left[1])
      x1 = min(x+block_size+overlap, lower_right[0])
      y1 = min(y+block_size+overlap, lower_right[1])
      blocks.append([[x0, y0], [x1, y1]])
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

    # wall clock time (full)
    total_time = 0
    total_start = time.time()

    # Set random seed
    np.random.seed(index)

    # Initialize query count, lls_block_size, main session, results & pre_results array
    num_queries = 0
    non_parallel_queries = 0 # for counting non parallel queries
    lls_block_size = self.lls_block_size
    sess = sesses[0]
    results, prev_results, = None, None

    # Initialize global noise
    noise = np.zeros_like(image, dtype=np.int32)

    # Initialize admm & adam variables
    yk = []
    mk = []
    vk = []
    beta1 = 0.9
    beta2 = 0.999
    for block in self.blocks:
      upper_left, lower_right = block

      yk_i = np.zeros((lower_right[0]-upper_left[0], lower_right[1]-upper_left[1], 3))
      yk.append(yk_i)

      if self.adam:
        mk_i = np.zeros_like(yk_i)
        vk_i = np.zeros_like(yk_i)
        mk.append(mk_i)
        vk.append(vk_i)

    rho = self.admm_rho
    tau = self.admm_tau

    # Initialize success flag
    self.success_checker.reset()

    # Run admm iterations
    for step in range(self.admm_iter):

      # wall clock time (round)
      start = time.time()

      # initialize lls blocks
      for i in range(len(self.blocks)):
        num_merge_batches = self.lazy_local_search[i].split_lls_blocks(self.blocks[i], lls_block_size)

      # run threads
      for ibatch in range(num_merge_batches):

        # Initialize threads and new block noises
        threads = []
        new_block_noises = [None] * len(self.blocks)

        # make results object to receive results from threads
        results = [None] * len(self.blocks)

        for i in range(len(self.blocks)):
          # Solve lazy greedy on the block
          if step == 0 and ibatch == 0:
            new_block_noises[i] = -self.epsilon * np.ones_like(noise, dtype=np.int32)
          else:
            prev_block_noise, _, _, block, _ = prev_results[i]

            upper_left, lower_right = block

            # initialize to (x ; z \ x)
            new_block_noises[i] = np.copy(noise)
            new_block_noises[i][:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :] = \
              prev_block_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :]

          threads.append(threading.Thread(target=self.lazy_local_search[i].perturb, args=(image,
                                                                                          new_block_noises[i],
                                                                                          noise,
                                                                                          label,
                                                                                          sesses[i%self.parallel],
                                                                                          self.blocks[i],
                                                                                          self.success_checker,
                                                                                          yk[i],
                                                                                          rho,
                                                                                          i,
                                                                                          ibatch,
                                                                                          results)))

        # Run threads
        num_running = 0
        for i in range(len(self.blocks)):
          threads[i].start()
          num_running += 1

          # If all gpus are used, wait for results
          if num_running == self.parallel:

            for j in range(i-self.parallel+1, i+1):
              threads[j].join()

            # Gather results
            for j in range(i-self.parallel+1, i+1):
              block_noise, block_queries, block_loss, block, block_success = results[j]

              num_queries += block_queries

              # Early stop checking
              if block_success:
                noise = block_noise
                curr_loss = block_loss
                adv_image = self._perturb_image(image, noise)

            # Max query checking (use per-gpu queries)
            parallel_queries = self._parallel_queries(num_queries, non_parallel_queries)

            if parallel_queries > self.max_queries:
              end = time.time()
              total_time = end - total_start
              tf.logging.info('Step {}, Loss: {:.5f}, total queries: {}, '
                              'per-gpu queries: {:.0f}, Time taken: {:.2f}'.format(
                step, curr_loss, num_queries, parallel_queries, end - start))
              return adv_image, num_queries, parallel_queries, False, total_time

            # Early stop checking
            if self.success_checker.check():
              end = time.time()
              total_time = end - total_start
              tf.logging.info('Step {}, Loss: {:.5f}, total queries: {}, '
                              'per-gpu queries: {:.0f}, Time taken: {:.2f}'.format(
                step, curr_loss, num_queries, parallel_queries, end - start))
              return adv_image, num_queries, parallel_queries, True, total_time

            num_running = 0

        # Update global variable by averaging
        overlap_count = np.zeros_like(noise)
        new_noise = np.zeros_like(noise)

        for block_noise, _, _, block, _ in results:

          upper_left, lower_right = block

          new_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :] += \
              block_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :]

          overlap_count[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :] += \
              np.ones_like(block_noise[:, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :])

        new_noise = new_noise / overlap_count

        # check convergence (global)
        change_ratio = np.mean(new_noise != noise)

        noise = new_noise

        # Check early stop
        noise_threshold = np.where(noise==0, -1, noise)
        noise_threshold = self.epsilon * np.sign(noise_threshold)

        adv_image = self._perturb_image(image, noise_threshold)
        feed = {
          self.model.x_input: adv_image,
          self.model.y_input: label
        }
        losses, preds = sess.run([self.losses, self.preds],
                                 feed_dict=feed)
        num_queries += 1
        non_parallel_queries += 1

        curr_loss = losses[0]

        # Max queries checking (use per-gpu queries)
        parallel_queries = self._parallel_queries(num_queries, non_parallel_queries)

        # save previous results for next round
        prev_results = copy.deepcopy(results)

        # max query checking
        if parallel_queries > self.max_queries:
          end = time.time()
          total_time = end - total_start
          tf.logging.info('Step {}, Loss: {:.5f}, total queries: {}, '
                          'per-gpu queries: {:.0f}, change ratio: {:.4f}, Time taken: {:.2f}'.format(
            step, curr_loss, num_queries, parallel_queries, change_ratio, end - start))
          return adv_image, num_queries, parallel_queries, False, total_time

        # Check early stop
        if self.targeted:
          if preds == label:
            end = time.time()
            total_time = end - total_start
            tf.logging.info('Step {}, Loss: {:.5f}, total queries: {}, '
                            'per-gpu queries: {:.0f}, change ratio: {:.4f}, Time taken: {:.2f}'.format(
              step, curr_loss, num_queries, parallel_queries, change_ratio, end - start))
            return adv_image, num_queries, parallel_queries, True, total_time
        else:
          if preds != label:
            end = time.time()
            total_time = end - total_start
            tf.logging.info('Step {}, Loss: {:.5f}, total queries: {}, '
                            'per-gpu queries: {:.0f}, change ratio: {:.4f}, Time taken: {:.2f}'.format(
              step, curr_loss, num_queries, parallel_queries, change_ratio, end - start))
            return adv_image, num_queries, parallel_queries, True, total_time

        end = time.time()
        tf.logging.info('Step {}, Loss: {:.5f}, total queries: {}, '
                        'per-gpu queries: {:.0f}, change ratio: {:.4f}, Time taken: {:.2f}'.format(
          step, curr_loss, num_queries, parallel_queries, change_ratio, end - start))

      # Update admm variables
      if self.admm:

        for i in range(len(self.blocks)):
          block_noise, _, _, block, _ = results[i]

          upper_left, lower_right = block
          dist = (block_noise - noise)[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], :]

          # update by adam optimizer
          if self.adam:
            lr = rho * np.sqrt(1-beta2**(step+1))/(1-beta1**(step+1))
            mk[i] = beta1 * mk[i] + (1-beta1) * dist
            vk[i] = beta2 * vk[i] + (1-beta2) * (dist**2)
            yk[i] += lr * mk[i] / (np.sqrt(vk[i]) + 1e-8)
          # else, update by rho tuned with tau (or not)
          else:
            yk[i] += rho * dist

        # tune rho with tau
        if not self.adam or self.adam_adapt:
          rho *= tau

      # Divide lls_block_size if hierarchical is used
      if not self.no_hier and ((step+1)% self.lls_iter == 0) and lls_block_size > 1:
        lls_block_size //= 2

    # Attack failed
    total_end = time.time()
    total_time = total_end - total_start
    return adv_image, num_queries, parallel_queries, False, total_time

