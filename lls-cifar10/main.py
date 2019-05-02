import argparse
import inspect
import math
import numpy as np
import sys
import time
import tensorflow as tf
import os
from PIL import Image

import attacks
import cifar10_input
from model import Model

""" The collection of all attack classes """
ATTACK_CLASSES = [
  x for x in attacks.__dict__.values() 
  if inspect.isclass(x)
]
for attack in ATTACK_CLASSES:
  setattr(sys.modules[__name__], attack.__name__, attack)

""" Arguments """
parser = argparse.ArgumentParser()

#MODEL_DIR = './models/adv_trained'
MODEL_DIR = './models/naturally_trained'
DATA_DIR = './../cifar10_data'
SAVE_DIR = './save'

# Directory
parser.add_argument('--model_dir', default=MODEL_DIR, type=str)
parser.add_argument('--data_dir', default=DATA_DIR, type=str)

# Experiment Setting
parser.add_argument('--img_index_start', default=0, type=int)
parser.add_argument('--sample_size', default=500, type=int)
parser.add_argument('--save_img', dest='save_img', action='store_true')
parser.add_argument('--save_dir', default=SAVE_DIR)

# Attack
parser.add_argument('--attack', default='LazyLocalSearchAttack', type=str)
parser.add_argument('--loss_func', default='xent', type=str)
parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--max_queries', default=20000, type=int)
parser.add_argument('--targeted', action='store_true')

# Lazy Local Search
parser.add_argument('--max_iters', default=1, type=int)
parser.add_argument('--block_size', default=4, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--no_hier', action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
  # Set verbosity
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Load pretrained model
  model_file = tf.train.latest_checkpoint(args.model_dir)
  if model_file is None:
    tf.logging.info('No model found')
    sys.exit()
  
  # Create session
  sess = tf.InteractiveSession()
   
  # Build graph
  model = Model(mode='eval')
 
  # Restore the checkpoint
  saver = tf.train.Saver()
  saver.restore(sess, model_file)
  
  # Create attack class
  attack_class = getattr(sys.modules[__name__], args.attack)
  attack = attack_class(model, args)
  
  # Create directory
  if args.save_img:
    tf.gfile.MakeDirs(args.save_dir)

  # Print hyperparameters
  for key, val in vars(args).items():
    tf.logging.info('{}={}'.format(key, val))

  # Load dataset
  cifar = cifar10_input.CIFAR10Data(args.data_dir)
  
  # Load indices
  indices = np.load('../cifar10_data/indices_untargeted.npy') 
  
  count = 0
  index = args.img_index_start
  total_num_corrects = 0
  total_queries = []
  index_to_query = {}

  while count < args.sample_size:
    tf.logging.info("")

    # Get image and label
    initial_img = cifar.eval_data.xs[indices[index]]
    initial_img = np.int32(initial_img)
    initial_img = np.expand_dims(initial_img, axis=0)
    orig_class = cifar.eval_data.ys[indices[index]]
    orig_class = np.expand_dims(orig_class, axis=0)
   
    # Generate target class (same method as in Boundary attack)
    if args.targeted:
      target_class = (orig_class+1) % 10
      target_class = np.expand_dims(target_class, axis=0)

    count += 1
   
    # Run attack
    if args.targeted:
      tf.logging.info('Targeted attack on {}th image starts, index: {}, orig class: {}, target class: {}'.format(
        count, indices[index], orig_class[0], target_class[0]))
      adv_img, num_queries, success = attack.perturb(initial_img, target_class, indices[index], sess)
    else:
      tf.logging.info('Untargeted attack on {}th image starts, index: {}, orig class: {}'.format(
        count, indices[index], orig_class[0]))
      adv_img, num_queries, success = attack.perturb(initial_img, orig_class, indices[index], sess)
    
    # Check if the adversarial image satisfies the constraint 
    assert np.amax(np.abs(adv_img-initial_img)) <= args.epsilon
    assert np.amax(adv_img) <= 255
    assert np.amin(adv_img) >= 0
    p = sess.run(model.predictions, feed_dict={model.x_input: adv_img})

    # Save the adversarial image
    if args.save_img:
      adv_image = Image.fromarray(np.ndarray.astype(adv_img[0, ...]*255, np.uint8))
      adv_image.save(os.path.join(args.save_dir, '{}_adv.jpg'.format(indices[index])))
    
    # Logging
    if success:
      total_num_corrects += 1
      total_queries.append(num_queries)
      index_to_query[indices[index]] = num_queries
      average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
      median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
      success_rate = total_num_corrects/count
      tf.logging.info('Attack success, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
        p[0], average_queries, median_queries, success_rate))   
    else:
      index_to_query[indices[index]] = -1
      average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
      median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
      success_rate = total_num_corrects/count
      tf.logging.info('Attack fail, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
        p[0], average_queries, median_queries, success_rate))   
    
    index += 1

