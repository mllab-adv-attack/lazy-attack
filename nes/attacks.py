import numpy as np
import tensorflow as tf

from tools.utils import *
from tools.logging_utils import *
from tools.inception_v3_imagenet import model
from tools.imagenet_labels import label_to_name

IMAGENET_PATH="../data"
NUM_LABELS=1000

def main(args, gpus):
  # Set random seed
  np.random.seed(0)

  # Set verbosity
  tf.logging.set_verbosity(tf.logging.INFO)

  # Load indices
  indices = np.load('../data/indices_targeted.npy')

  # Setting 
  batch_size = args.batch_size
  epsilon = args.epsilon
  batch_per_gpu = batch_size // len(gpus)
  max_iters = int(np.ceil(args.max_queries // (args.samples_per_draw+1)))
  k = NUM_LABELS
  is_targeted = 1

  # Session initialization
  sess = tf.InteractiveSession()
  x_input = tf.placeholder(tf.float32, [None, 299, 299, 3])
  y_input = tf.placeholder(tf.int32, [None])
  eval_logits, eval_preds = model(sess, x_input)
  eval_percent_adv = tf.equal(eval_preds, tf.cast(y_input, tf.int64))

  # Loss function
  def standard_loss(eval_points, noise):
    logits, preds = model(sess, eval_points)
    labels = tf.tile(y_input, [logits.shape[0]])
    labels = tf.one_hot(labels, NUM_LABELS)
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    
    return losses, noise

  # Gradient estimation graph
  grad_estimates = []
  final_losses = []
  loss_fn = standard_loss

  for i, device in enumerate(gpus):
    with tf.device(device):
      tf.logging.info('Loading on gpu {} of {}'.format(i+1, len(gpus)))
      noise_pos = tf.random_normal((batch_per_gpu//2,) + (299, 299, 3))
      noise = tf.concat([noise_pos, -noise_pos], axis=0)
      eval_points = x_input + args.sigma * noise
      losses, noise = loss_fn(eval_points, noise)
    losses_tiled = tf.tile(tf.reshape(losses, (-1, 1, 1, 1)), (1,) + (299, 299, 3))
    grad_estimates.append(tf.reduce_mean(losses_tiled * noise, axis=0)/args.sigma)
    final_losses.append(losses)
  grad_estimate = tf.reduce_mean(grad_estimates, axis=0)
  final_losses = tf.concat(final_losses, axis=0)

  # Gradient estimation eval
  def get_grad(pt, lb, spd, bs):
      num_batches = spd // bs
      losses = []
      grads = []
      feed_dict = {x_input: pt, y_input: lb}
      for _ in range(num_batches):
          loss, dl_dx_ = sess.run([final_losses, grad_estimate], feed_dict)
          losses.append(np.mean(loss))
          grads.append(dl_dx_)
      return np.array(losses).mean(), np.mean(np.array(grads), axis=0)

  # Step condition (important for partial-info attacks)
  def robust_in_top_k(t_, prop_adv_,k_):
      if k == NUM_LABELS:
          return True
      for i in range(1):
          n = np.random.rand(*prop_adv_.shape)*args.sigma
          eval_logits_ = sess.run(eval_logits, {x_input: prop_adv_})[0]
          if not t_ in eval_logits_.argsort()[-k_:][::-1]:
              return False
      return True
  
  # Print hyperparameter
  for key, val in vars(args).items():
    tf.logging.info("{}={}".format(key, val))

  total_num_corrects = 0
  total_num_queries = []  

  for i, idx in enumerate(indices[args.img_index:args.img_index+args.sample_size]):
    tf.logging.info('')

    # Get image and label
    initial_img, orig_class = get_image(idx, IMAGENET_PATH)
    initial_img = np.expand_dims(initial_img, axis=0)
    lower = np.clip(initial_img - args.epsilon, 0., 1.)
    upper = np.clip(initial_img + args.epsilon, 0., 1.)
    adv = initial_img.copy()

    # Choose target class
    target_class = pseudorandom_target(idx, NUM_LABELS, orig_class)
    target_class = np.expand_dims(target_class, axis=0)
    tf.logging.info('{}th image, image index: {}, orig class: {}, target class: {}'.format(i+1, idx, orig_class, target_class[0]))
   
    # History variables
    num_queries = 0
    g = 0
    prev_adv = adv
    last_ls = []
    goal_epsilon = epsilon
    max_lr = args.max_lr
     
    # Main loop
    for j in range(max_iters):
      # Check if we should stop
      padv = sess.run(eval_percent_adv, feed_dict={x_input: adv, y_input: target_class})
      if padv == 1 and epsilon <= goal_epsilon:
        #tf.logging.info('Early stopping at iteration {}'.format(j))
        break

      prev_g = g
      l, g = get_grad(adv, target_class, args.samples_per_draw, batch_size)

      # Simple momentum
      g = args.momentum * prev_g + (1.0 - args.momentum) * g

      # Plateau lr annealing
      last_ls.append(l)
      last_ls = last_ls[-args.plateau_length:]
      if last_ls[-1] > last_ls[0] and len(last_ls) == args.plateau_length:
        if max_lr > args.min_lr:
          #tf.logging.info("Annealing max_lr")
          max_lr = max(max_lr / args.plateau_drop, args.min_lr)
        last_ls = []

      # Search for lr and epsilon decay
      current_lr = max_lr
      proposed_adv = adv - is_targeted * current_lr * np.sign(g)
      prop_de = 0.0
      while current_lr >= args.min_lr:
        # General line search
        proposed_adv = adv - is_targeted * current_lr * np.sign(g)
        proposed_adv = np.clip(proposed_adv, lower, upper)
        num_queries += 1
        if robust_in_top_k(target_class, proposed_adv, k):
          if prop_de > 0:
            delta_epsilon = max(prop_de, 0.1)
            last_ls = []
          prev_adv = adv
          adv = proposed_adv
          epsilon = max(epsilon - prop_de/args.conservative, goal_epsilon)
          break
        elif current_lr >= args.min_lr*2:
          current_lr = current_lr / 2
          #print("[log] backtracking lr to %3f" % (current_lr,))
        else:
          prop_de = prop_de / 2
          if prop_de == 0:
            raise ValueError("Did not converge.")
          if prop_de < 2e-3:
            prop_de = 0
          current_lr = max_lr
          #print("[log] backtracking eps to %3f" % (epsilon-prop_de,))

      # Book-keeping stuff
      num_queries += args.samples_per_draw
      if j % 10 == 0:
        tf.logging.info('Step {}: loss: {:.4f}, lr: {}, num queries: {}'.format(
          j, l, current_lr, num_queries))
    
    assert(np.amax(np.abs(adv-initial_img))<=epsilon+1e-3)

    padv = sess.run(eval_percent_adv, feed_dict={x_input: adv, y_input: target_class})
    
    if padv == 1:
      total_num_corrects += 1
      total_num_queries.append(num_queries)
      tf.logging.info('{}th image successes, average queries: {:.4f}, median queries: {}, success rate: {:.4f}'.format(
        i+1, np.sum(total_num_queries)/max(1, total_num_corrects), np.median(total_num_queries), total_num_corrects/(i+1)))
    else:
      tf.logging.info('{}th image fails, average queries: {:.4f}, median queries: {}, success rate: {:.4f}'.format(
        i+1, np.sum(total_num_queries)/max(1, total_num_corrects), np.median(total_num_queries), total_num_corrects/(i+1)))
    
if __name__ == '__main__':
  main()
