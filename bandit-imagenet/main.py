import argparse
import inspect
import math
import numpy as np
import sys
import tensorflow as tf

from tools.inception_v3_imagenet import model
from tools.utils import *
import attacks

ATTACK_CLASSES = [x for x in attacks.__dict__.values() if inspect.isclass(x)]
for attack in ATTACK_CLASSES:
  setattr(sys.modules[__name__], attack.__name__, attack)

IMAGENET_PATH = '../imagenet_data'
IMAGENET_SL = 299
NUM_LABELS = 1000

parser = argparse.ArgumentParser()

# Setting
parser.add_argument('--max_queries', default=10000, type=int)
parser.add_argument('--epsilon', default='0.05', type=float)
parser.add_argument('--img_index_start', default=0, type=int)
parser.add_argument('--sample_size', default=1000, type=int)
parser.add_argument('--save_img', dest='save_img', action='store_true')
parser.add_argument('--attack', default='BanditAttack', type=str)

parser.add_argument('--gradient_iters', default=1, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--tile_size', default=50, type=int)
parser.add_argument('--exploration', default=0.01, type=float)
parser.add_argument('--fd_eta', default=0.1, type=float)
parser.add_argument('--online_lr', default=1, type=float)
parser.add_argument('--image_lr', default=0.005, type=float)

args = parser.parse_args()

if __name__ == '__main__':
  # Set verbosity
  tf.logging.set_verbosity(tf.logging.INFO)
    
  # Create session
  sess = tf.InteractiveSession()

  # Create attack class.
  attack_class = getattr(sys.modules[__name__], args.attack)
  bandit_attack = attack_class(args, sess)

  # Print hyperparameters
  for key, val in vars(args).items():
    tf.logging.info('{}={}'.format(key, val))

  # Load indices. 
  indices = np.load('../data/indices_untargeted.npy')

  # Main loop
  num_batches = int(math.ceil(args.sample_size/args.batch_size))
  index_to_num_queries = {}
 
  for batch in range(num_batches):
    tf.logging.info('')
    
    bstart = batch*args.batch_size + args.img_index_start
    bend = min(bstart+args.batch_size, args.sample_size+args.img_index_start)  
    
    # Get image and label.
    image_batch = np.zeros([bend-bstart, IMAGENET_SL, IMAGENET_SL, 3], np.float32)
    label_batch = np.zeros([bend-bstart], np.int32)
    for i, idx in enumerate(range(bstart, bend)):
      image, label = get_image(indices[idx], IMAGENET_PATH)
      image_batch[i, ...] = image
      label_batch[i, ...] = label

    tf.logging.info('Untargeted attack on {}th batch starts'.format(batch))
    bandit_attack.perturb(image_batch, label_batch, sess)

