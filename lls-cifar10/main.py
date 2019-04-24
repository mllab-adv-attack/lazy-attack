import argparse
import inspect
import math
import numpy as np
import sys
import time
import tensorflow as tf
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

# Directory
parser.add_argument('--model_dir', default='./models/adv_trained', type=str)
parser.add_argument('--data_dir', default='../cifar10_data', type=str)

# Experiment Setting
parser.add_argument('--img_index_start', default=0, type=int)
parser.add_argument('--sample_size', default=100, type=int)
parser.add_argument('--save_img', dest='save_img', action='store_true')
parser.add_argument('--save_dir', default='/data_large/unsynced_store/seungyong/output/cifar10/lls/untargeted')

# Attack
parser.add_argument('--attack', default='LazyLocalSearchAttack', type=str)
parser.add_argument('--loss_func', default='xent', type=str)
parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--max_queries', default=20000, type=int)
parser.add_argument('--targeted', action='store_true')

# Lazy Local Search Batch
parser.add_argument('--block_size', default=4, type=int)
parser.add_argument('--max_iters', default=2, type=int)
parser.add_argument('--batch_size', default=256, type=int)
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
  total_num_queries = []
  index_to_num_queries = {}

  while count < args.sample_size:
    tf.logging.info("")

    # Get image and label
    initial_img = cifar.eval_data.xs[indices[index]]
    initial_img = np.int32(initial_img)
    initial_img = np.expand_dims(initial_img, axis=0)
    orig_class = cifar.eval_data.ys[indices[index]]
    orig_class = np.expand_dims(orig_class, axis=0)
    
    count += 1
    
    tf.logging.info('Untargeted attack on {}th image starts, img index: {}, orig class: {}'.format(
      count, indices[index], orig_class[0]))
    
    adv_img, num_queries, success = attack.perturb(initial_img, orig_class, indices[index], sess)
    assert(np.amax(np.abs(adv_img-initial_img))<=args.epsilon)
    
    if args.save_img:
      nat_image = Image.fromarray(np.ndarray.astype(initial_img[0, ...]*255, np.uint8))
      nat_image.save(args.save_dir+'/nat/{}_nat.jpg'.format(indices[index]))
      adv_image = Image.fromarray(np.ndarray.astype(adv_img[0, ...]*255, np.uint8))
      adv_image.save(args.save_dir+'/adv/{}_adv.jpg'.format(indices[index]))
     
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
  
  filename = args.save_dir+'/lls_new_new_untargeted_{}_{}_{}_{}.npy'.format(
    args.loss_func, args.batch_size, args.max_iters, args.img_index_start+args.sample_size)
  np.save(filename, index_to_num_queries) 
