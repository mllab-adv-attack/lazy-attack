import argparse
from collections import namedtuple
import inspect
import numpy as np
import os
from PIL import Image
import sys
import tensorflow as tf

import attacks
from tools.inception_v3_imagenet import model
from tools.utils import *
from tools.imagenet_labels import *

""" The collection of all attack classes """
ATTACK_CLASSES = [x for x in attacks.__dict__.values() if inspect.isclass(x)]
for attack in ATTACK_CLASSES:
  setattr(sys.modules[__name__], attack.__name__, attack)

""" For passing boolean values to slurm file in hyperparameter searching"""
def str2bool(key):
  return key.lower() in ('yes', 'true', 'y', 't')

""" Namedtuple for model """
Model = namedtuple('Model', 'x_input, y_input, logits, preds, losses')

""" Arguments """
IMAGENET_PATH = '/data_large/readonly/imagenet_data'
MODEL_DIR = './tools/data'
NUM_LABELS = 1000

parser = argparse.ArgumentParser()

# Directory
parser.add_argument('--asset_dir', default='./assets', type=str)
parser.add_argument('--save_dir', default='./saves', type=str)

# Experimental setting
parser.add_argument('--img_index_start', default=0, type=int)
parser.add_argument('--sample_size', default=1000, type=int)
parser.add_argument('--save_img', dest='save_img', action='store_true')

# Attack setting
parser.add_argument('--attack', default='LazyLocalSearchAttack', type=str, help='The type of attack')
parser.add_argument('--epsilon', default=0.05, type=float, help='The maximum perturbation')
parser.add_argument('--max_queries', default=10000, type=int, help='The query limit')
parser.add_argument('--targeted', default='False', type=str2bool, help='Targeted attack if true')

# Parimonious attack setting
parser.add_argument('--lls_block_size', default=32, type=int, help='Initial block size')
parser.add_argument('--batch_size', default=64, type=int, help='The size of batch. No batch if negative')
parser.add_argument('--lls_iter', default=2, type=int, help='The number of iterations in local search')
parser.add_argument('--no_hier', default='False', type=str2bool)

# ADMM setting
parser.add_argument('--admm', default='False', type=str2bool, help='Use admm')
parser.add_argument('--admm_block_size', default=128, type=int, help='Block size for admm')
parser.add_argument('--partition', default='basic', type=str, help='Block partitioning scheme')
parser.add_argument('--admm_iter', default=100, type=int, help='ADMM max iteration')
parser.add_argument('--overlap', default=32, type=int, help='Overlap size')
parser.add_argument('--admm_rho', default=1e-11, type=float, help='ADMM rho')
parser.add_argument('--admm_tau', default=1.5, type=float, help='ADMM tau')
parser.add_argument('--adam', default='False', help='use adam optimizer', type=str2bool)
parser.add_argument('--adam_lr', default=1e-3, help='initial learning rate in adam', type=float)
parser.add_argument('--gpus', default=2, type=int, help='The number of gpus to use')
parser.add_argument('--parallel', default=4, type=int, help='The number of parallel threads to use')
parser.add_argument('--merge_per_batch', default='False', type=str2bool, help='merge after each mini-batch')

# Graph cut setting
parser.add_argument('--alpha', default=10000, type=int)
parser.add_argument('--beta', default=1, type=int)

args = parser.parse_args()

# adam not implemented yet
assert not args.adam

if __name__ == '__main__':
  # Set verbosity
  tf.logging.set_verbosity(tf.logging.INFO)
    
  # Create config for sessions
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.gpu_options.allow_growth = True

  # Build models
  graphs = []
  sesses = []
  models = []
  
  # Assign 1~4 threads per gpu
  assert (args.parallel >= args.gpus) and (args.parallel%args.gpus == 0) and (args.parallel//args.gpus <= 4)

  for gpu in range(4):
    graph = tf.Graph()
    graphs.append(graph)
    with graph.as_default():
      with tf.device('/gpu:'+str(gpu%args.gpus)):
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        y_input = tf.placeholder(dtype=tf.int32, shape=[None])
        sess = tf.Session(config=config)
        sesses.append(sess) 
        logits, preds = model(sess, x_input)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=y_input)
        if not args.targeted:
          losses *= -1
        models.append(Model(x_input, y_input, logits, preds, losses))

  # Restore models
  for gpu in range(4):
    graph = graphs[gpu]
    sess = sesses[gpu]
    checkpoint_path = MODEL_DIR+'/inception_v3.ckpt' 
    with graph.as_default():
      optimistic_restore(sess, checkpoint_path)  

  # Create a directory
  if args.save_img:
    tf.gfile.MakeDirs(args.save_dir)

  # Print hyperparameters
  for key, val in vars(args).items():
    tf.logging.info('{}={}'.format(key, val))
  
  # Create attack
  attack_class = getattr(sys.modules[__name__], args.attack)
  attack = attack_class(models, sesses, args)

  # Load the indices
  if args.targeted:
    indices = np.load(args.asset_dir+'/indices_targeted.npy')
  else:
    indices = np.load(args.asset_dir+'/indices_untargeted.npy')

  # Main loop
  count = 0
  index = args.img_index_start
  total_num_corrects = 0
  total_queries = []
  index_to_query = {}
 
  while count < args.sample_size:
    tf.logging.info('')
    
    # Get an image and the corresponding label
    initial_img, orig_class = get_image(indices[index], IMAGENET_PATH)
    initial_img = np.expand_dims(initial_img, axis=0)

    # Generate a target class (same method as in NES attack)
    if args.targeted:
      target_class = pseudorandom_target(indices[index], NUM_LABELS, orig_class)
      target_class = np.expand_dims(target_class, axis=0)
      
    orig_class = np.expand_dims(orig_class, axis=0)
   
    count += 1
    
    # Run attack
    if args.targeted:
      tf.logging.info('Targeted attack on {}th image starts, index: {}, orig class: {}, target class: {}'.format(
        count, indices[index], label_to_name(orig_class[0]), label_to_name(target_class[0])))
      adv_img, _, num_queries, success, _, losses = attack.perturb(initial_img, target_class, indices[index])
    else: 
      tf.logging.info('Untargeted attack on {}th image starts, index: {}, orig class: {}'.format(
        count, indices[index], label_to_name(orig_class[0])))
      adv_img, _, num_queries, success, _, losses = attack.perturb(initial_img, orig_class, indices[index])
   
    #np.save('./outputs/losses_{}.npy'.format(count), losses)
     
    # Check if the adversarial image satisfies the constraint
    assert np.amax(np.abs(adv_img-initial_img)) <= args.epsilon+1e-3    
    assert np.amax(adv_img) <= 1.+1e-3
    assert np.amin(adv_img) >= 0.-1e-3
    p = sess.run(preds, feed_dict={x_input: adv_img})

    # Save the adversarial image
    if args.save_img:
      adv_image = Image.fromarray(np.ndarray.astype(adv_img[0, :, :, :]*255, np.uint8))
      adv_image.save(os.path.join(args.save_dir, '{}_adv.jpg'.format(indices[index])))
   
    # Logging 
    if success:
      total_num_corrects += 1
      total_queries.append(num_queries)
      index_to_query[indices[index]] = num_queries
      average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
      median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
      success_rate = total_num_corrects/count
      tf.logging.info('Attack succeeds, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
        label_to_name(p[0]), average_queries, median_queries, success_rate))
    else:
      index_to_query[indices[index]] = -1
      average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
      median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
      success_rate = total_num_corrects/count
      tf.logging.info('Attack fails, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
        label_to_name(p[0]), average_queries, median_queries, success_rate))

    index += 1

