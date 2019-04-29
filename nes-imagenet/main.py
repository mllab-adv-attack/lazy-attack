import argparse
import inspect
import json
import numpy as np
import pickle
from PIL import Image
import sys
import tensorflow as tf

from attacks.nes_attack import NESAttack
from tools.inception_v3_imagenet import model
from tools.utils import *

IMAGENET_PATH = '../imagenet_data'
NUM_LABELS = 1000

parser = argparse.ArgumentParser()

# Experimental setting
parser.add_argument('--targeted', action='store_true')
parser.add_argument('--loss_func', default='xent', type=str)
parser.add_argument('--max_queries', default=100000, type=int)
parser.add_argument('--epsilon', default=0.05, type=float)
parser.add_argument('--img_index_start', default=0, type=int)
parser.add_argument('--sample_size', default=100, type=int)
parser.add_argument('--save_img', action='store_true')

# NES setting
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--sigma', default=1e-3, type=float)
parser.add_argument('--max_lr', default=1e-2, type=float)
parser.add_argument('--min_lr', default=5e-5, type=float)
parser.add_argument('--plateau_length', default=5, type=int)
parser.add_argument('--plateau_drop', default=2.0, type=float)
parser.add_argument('--momentum', default=0.9, type=float)

args = parser.parse_args()

if __name__ == '__main__':
  # Set verbosity
  tf.logging.set_verbosity(tf.logging.INFO)
    
  # Create session
  sess = tf.InteractiveSession()

  # Create attack class.
  attack = NESAttack(sess, args)
  
   # Print hyperparameters
  for key, val in vars(args).items():
    tf.logging.info('{}={}'.format(key, val))

  # Load indices
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
   
    # Run attack
    if args.targeted: 
      tf.logging.info('Targeted attack on {}th image starts, index: {}, orig class: {}, target class: {}'.format(
          count, indices[index], orig_class[0], target_class[0]))
      adv_img, num_queries, success = attack.perturb(initial_img, target_class, sess)
    else:
      tf.logging.info('Untargeted attack on {}th image starts, index: {}, orig class: {}'.format(
          count, indices[index], orig_class[0]))
      adv_img, num_queries, success = attack.perturb(initial_img, orig_class, sess)
    
    # Check if the adversarial image satisfies the constraint.
    assert np.amax(np.abs(adv_img-initial_img)) <= args.epsilon+1e-3
    assert np.amin(adv_img) >= 0
    assert np.amax(adv_img) <= 1 
    
    # Save the adversarial image.
    if args.save_img:
      nat_image = Image.fromarray(np.ndarray.astype(initial_img[0, :, :, :]*255, np.uint8))
      nat_image.save('/data_large/unsynced_store/seungyong/output/imagenet/nat/{}_nat.jpg'.format(indices[index]))
      adv_image = Image.fromarray(np.ndarray.astype(adv_img[0, :, :, :]*255, np.uint8))
      adv_image.save('/data_large/unsynced_store/seungyong/output/imagenet/adv/{}_adv.jpg'.format(indices[index]))
    
    if success:
      total_num_corrects += 1
      total_num_queries.append(num_queries)
      index_to_num_queries[indices[index]] = num_queries
      average_queries = 0 if len(total_num_queries) == 0 else np.mean(total_num_queries)
      median_queries = 0 if len(total_num_queries) == 0 else np.median(total_num_queries)
      tf.logging.info('Attack success, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
        average_queries, median_queries, total_num_corrects/count))
    
    else:
      index_to_num_queries[indices[index]] = -1
      average_queries = 0 if len(total_num_queries) == 0 else np.mean(total_num_queries)
      median_queries = 0 if len(total_num_queries) == 0 else np.median(total_num_queries)
      tf.logging.info('Attack fail, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
        average_queries, median_queries, total_num_corrects/count))
    
    index += 1
  
  targeted = 'targeted' if args.targeted else 'untargeted' 
  filename = '/data_large/unsynced_store/seungyong/output/imagenet/nes/targeted/nes_{}_{}_{}_{}.npy'.format(
    targeted, args.loss_func, args.momentum, args.img_index_start+args.sample_size)
  np.save(filename, index_to_num_queries)
