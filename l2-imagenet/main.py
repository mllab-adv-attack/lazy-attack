import argparse
import inspect
import json
import numpy as np
from PIL import Image
import sys
import tensorflow as tf

from tools.inception_v3_imagenet import model
from tools.utils import *
import attacks

ATTACK_CLASSES = [x for x in attacks.__dict__.values() if inspect.isclass(x)]
for attack in ATTACK_CLASSES:
  setattr(sys.modules[__name__], attack.__name__, attack)

IMAGENET_PATH = '../data'
NUM_LABELS = 1000

parser = argparse.ArgumentParser()

# Setting
parser.add_argument('--loss_func', default='xent', type=str)
parser.add_argument('--max_queries', default=10000, type=int)
parser.add_argument('--epsilon', default='5.0', type=float)
parser.add_argument('--img_index_start', default=0, type=int)
parser.add_argument('--sample_size', default=1000, type=int)
parser.add_argument('--save_img', dest='save_img', action='store_true')
parser.add_argument('--attack', default='LazyLocalSearchBatchAttackNew', type=str)
parser.add_argument('--targeted', action='store_true')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=64, type=int)

args = parser.parse_args()

if __name__ == '__main__':
  # Set verbosity
  tf.logging.set_verbosity(tf.logging.INFO)
    
  # Create session
  sess = tf.InteractiveSession()

  # Build graph
  x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
  y_input = tf.placeholder(dtype=tf.int32, shape=[None])
  noise = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])

  noise_resized = tf.image.resize_nearest_neighbor(noise, (299, 299))
  noise_centered = noise_resized - tf.reduce_mean(noise_resized)
  x_adv = x_input + tf.nn.l2_normalize(noise_centered, axis=(1,2,3)) * args.epsilon
  x_adv = tf.clip_by_value(x_adv, 0, 1)
  
  logits, preds = model(sess, x_adv)
  
  model = {
    'x_input': x_input,
    'noise': noise,
    'x_adv': x_adv,
    'y_input': y_input,
    'logits': logits,
    'preds': preds,
    'targeted': args.targeted,
  }

  # Print hyperparameters
  for key, val in vars(args).items():
    tf.logging.info('{}={}'.format(key, val))

  # Create attack class.
  attack_class = getattr(sys.modules[__name__], args.attack)
  lazy_local_search_attack = attack_class(model, args)

  # Load indices. 
  if args.targeted:
    indices = np.load('../data/indices_targeted.npy')
  else:
    indices = np.load('../data/indices_untargeted.npy')

  # Main loop
  count = 0
  index = args.img_index_start
  total_num_corrects = 0
  total_num_queries = []
  index_to_num_queries = {}
 
  while count < args.sample_size:
    tf.logging.info('')
    
    # Get image and label.
    initial_img, orig_class = get_image(indices[index], IMAGENET_PATH)
    initial_img = np.expand_dims(initial_img, axis=0)

    # Generate target class(the same method as in NES attack).
    if args.targeted:
      target_class = pseudorandom_target(indices[index], NUM_LABELS, orig_class)
      target_class = np.expand_dims(target_class, axis=0)
      
    orig_class = np.expand_dims(orig_class, axis=0)
   
    count += 1
    
    # Run attack.
    if not args.targeted:
      tf.logging.info('Untargeted attack on {}th image starts, index: {}, orig class: {}'.format(
        count, indices[index], orig_class[0]))
      adv_img, num_queries, success = lazy_local_search_attack.perturb(initial_img, orig_class, indices[index], sess)
    else:
      tf.logging.info('Targeted attack on {}th image starts, index: {}, orig class: {}, target class: {}'.format(
        count, indices[index], orig_class[0], target_class[0]))
      adv_img, num_queries, success = lazy_local_search_attack.perturb(initial_img, target_class, indices[index], sess)
    
    # Check if the adversarial image satisfies the constraint.
    assert(np.linalg.norm(adv_img-initial_img) <= args.epsilon+1e-3)    
    
    # Save the adversarial image.
    if args.save_img:
      nat_image = Image.fromarray(np.ndarray.astype(initial_img[0, :, :, :], np.uint8))
      nat_image.save('/data_large/unsynced_store/seungyong/output/imagenet/nat/{}_nat.jpg'.format(indices[index]))
      adv_image = Image.fromarray(np.ndarray.astype(adv_img[0, :, :, :], np.uint8))
      adv_image.save('/data_large/unsynced_store/seungyong/output/imagenet/adv/{}_adv.jpg'.format(indices[index]))
    
    if success:
      total_num_corrects += 1
      total_num_queries.append(num_queries)
      index_to_num_queries[indices[index]] = num_queries
      average_queries = 0 if len(total_num_queries) == 0 else np.mean(total_num_queries)
      median_queries = 0 if len(total_num_queries) == 0 else np.median(total_num_queries)
      tf.logging.info('Attack successes, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
        average_queries, median_queries, total_num_corrects/count))
    else:
      index_to_num_queries[indices[index]] = -1
      average_queries = 0 if len(total_num_queries) == 0 else np.mean(total_num_queries)
      median_queries = 0 if len(total_num_queries) == 0 else np.median(total_num_queries)
      tf.logging.info('Attack fails, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
        average_queries, median_queries, total_num_corrects/count))

    index += 1
  
  targeted = 'targeted' if args.targeted else 'untargeted'
  #filename = '/data_large/unsynced_store/seungyong/output/imagenet/lls_new_{}_{}_{}.npy'.format(
  #  targeted, args.loss_func, args.img_index_start+args.sample_size)
  #np.save(filename, index_to_num_queries)
  filename = '/data/home/gaon/lazy-attack/l2-imagenet/out/l2_{}_{}_32_median.npy'.format(
    args.img_index_start, args.sample_size)
  np.save(filename, index_to_num_queries)
