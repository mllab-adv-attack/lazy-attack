import tensorflow as tf
import numpy as np
import heapq

class LazyGreedyAttack(object):
  def __init__(self, model, epsilon, loss_func, **kwargs):
    self.model = model
    self.epsilon = epsilon
    
    if loss_func == 'xent':
      self.loss = -self.model.y_xent
    elif loss_func == 'cw':
      self.loss = -self.model.y_cw
    else:
      tf.logging.info("Unknown loss function, Defaulting to cross-entropy")
      self.loss = -self.model.y_xent

  def perturb(self, image, label, sess):		
    adv_image = np.copy(image)
    priority_queue = []
    num_queries = 0

    # Calculate current loss
    feed = {
      self.model.x_input: adv_image,
      self.model.y_input: label,
    }
    curr_loss = sess.run(self.loss, feed)
    num_queries += 1

    # First forward passes
    for w in range(32):
      image_batch = np.tile(adv_image, (32*3*2, 1, 1, 1))
      label_batch = np.tile(label, 32*3*2)
      # run forward passes over a row
      for pos in range(32*3):
        h = pos // 3
        c = pos % 3
        image_batch[2*pos, w, h, c] -= self.epsilon
        image_batch[2*pos+1, w, h, c] += self.epsilon 
      image_batch = np.clip(image_batch, 0, 255)
      feed = {
        self.model.x_input: image_batch,
        self.model.y_input: label_batch
      }
      losses = sess.run(self.loss, feed)
      num_queries += 32*3*2
      # Push into the priority queue
      for pos in range(32*3):
        h = pos // 3
        c = pos % 3
        margin_mi = losses[2*pos]-curr_loss
        margin_pl = losses[2*pos+1]-curr_loss
        heapq.heappush(priority_queue, (min(margin_mi, margin_pl), margin_mi, margin_pl, w, h, c))
    
    # Pick the best element and perturb the image   
    best_margin, best_margin_mi, best_margin_pl, best_w, best_h, best_c = heapq.heappop(priority_queue)
    curr_loss += best_margin
    adv_image[0, best_w, best_h, best_c] += -self.epsilon if best_margin_mi < best_margin_pl else self.epsilon
    adv_image = np.clip(adv_image, 0, 255)

    # Run a loop while the priority queue is empty
    while len(priority_queue) > 0:
      # pick the best element
      cand_margin, cand_margin_mi, cand_margin_pl, cand_w, cand_h, cand_c = heapq.heappop(priority_queue)
      # Re-evalulate the element
      image_batch = np.tile(adv_image, (2, 1, 1, 1))
      label_batch = np.tile(label, 2)
      image_batch[0, cand_w, cand_h, cand_c] -= self.epsilon
      image_batch[1, cand_w, cand_h, cand_c] += self.epsilon
      image_batch = np.clip(image_batch, 0, 255)
      feed = {
        self.model.x_input: image_batch,
        self.model.y_input: label_batch
      }
      num_queries += 2
      losses, cw_losses = sess.run([self.loss, self.model.y_cw], feed)
      margin_mi = losses[0]-curr_loss
      margin_pl = losses[1]-curr_loss
      margin = min(margin_mi, margin_pl)
       
      # If the cardinality has not changed, perturb the image
      if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
        curr_loss += margin
        adv_image[0, cand_w, cand_h, cand_c] += -self.epsilon if margin_mi < margin_pl else self.epsilon
        adv_image = np.clip(adv_image, 0, 255)
        # Early stopping
        if cw_losses[0] > 0 or cw_losses[1] > 0:
          break;
      # If the cardinality has changed, push the element into the priority queue
      else: 
        heapq.heappush(priority_queue, (margin, margin_mi, margin_pl, cand_w, cand_h, cand_c))
	
    return adv_image, num_queries

