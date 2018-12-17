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
parser.add_argument('--max_queries', default=10000, type=int)
parser.add_argument('--epsilon', default='0.05', type=float)
parser.add_argument('--img_index_start', default=0, type=int)
parser.add_argument('--sample_size', default=1000, type=int)
parser.add_argument('--save_img', dest='save_img', action='store_true')
parser.add_argument('--attack', default='LazyLocalSearchBatchAttack', type=str)
parser.add_argument('--mode', default='val', type=str)

args = parser.parse_args()

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Print hyper-parameters
  for key, val in vars(args).items():
    tf.logging.info('{}={}'.format(key, val))
  
  # Create session
  sess = tf.InteractiveSession()

  # Build graph
  x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
  y_input = tf.placeholder(dtype=tf.int32, shape=[None])
  
  logits, preds = model(sess, x_input)
  labels = tf.one_hot(y_input, NUM_LABELS)
  losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  model = {
    'x_input': x_input,
    'y_input': y_input,
    'logits': logits,
    'preds': preds,
    'losses': losses 
  }

  # Create directory
  tf.gfile.MakeDirs('/data_large/unsynced_store/seungyong/output/adv')
  tf.gfile.MakeDirs('/data_large/unsynced_store/seungyong/output/nat')

  # Create attack class
  attack_class = getattr(sys.modules[__name__], args.attack)
  lazy_local_search_attack = attack_class(model, args.epsilon, args.max_queries)

  count = 0
  index = args.img_index_start
  total_num_corrects = 0
  total_num_queries = 0
  
  if args.mode == 'test':
    indices = np.load('./data/intersection_norm.npy')
  else: 
    indices = np.arange(0, 50000)

  while count < args.sample_size:
    tf.logging.info('')
    initial_img, orig_class = get_image(indices[index], IMAGENET_PATH)
    initial_img = np.expand_dims(initial_img, axis=0)
    orig_class = np.expand_dims(orig_class, axis=0)
    index += 1
    success = False
    
    p = sess.run(preds, feed_dict={x_input: initial_img})
    if p[0] != orig_class:
      tf.logging.info('Misclassified, continue to the next image')
      continue
    
    count += 1
    
    adv_img, num_queries = lazy_local_search_attack.perturb(initial_img, orig_class, sess)
    assert(np.amax(np.abs(adv_img-initial_img)) <= args.epsilon+1e-3)    
    
    if args.save_img:
      adv_image = Image.fromarray(np.ndarray.astype(adv_img[0, :, :, :]*255, np.uint8))
      adv_image.save('/data_large/unsynced_store/seungyong/output/adv/{}_adv.jpg'.format(indices[index-1]))
      image = Image.fromarray(np.ndarray.astype(initial_img[0, :, :, :]*255, np.uint8))
      image.save('/data_large/unsynced_store/seungyong/output/nat/{}_nat.jpg'.format(indices[index-1]))
    
    p = sess.run(preds, feed_dict={x_input: adv_img})

    if p[0] != orig_class:
      total_num_corrects += 1
      total_num_queries += num_queries
      tf.logging.info('Image {} attack success, image index: {}, average queries: {}, success rate: {}'.format(
        count, indices[index-1], total_num_queries/max(1, total_num_corrects), total_num_corrects/count))
    else:
      tf.logging.info('Image {} attack fails, image index: {}, average queries: {}, success rate: {}'.format(
        count, indices[index-1], total_num_queries/max(1, total_num_corrects), total_num_corrects/count))
