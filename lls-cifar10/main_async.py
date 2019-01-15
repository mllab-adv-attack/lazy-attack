import argparse
import inspect
import math
import numpy as np
import os
import sys
import time
import tensorflow as tf

import attacks
import cifar10_input
from model import Model
from utils.format_utils import params2id

""" The collection of all attack classes """
ATTACK_CLASSES = [
  x for x in attacks.__dict__.values()
  if inspect.isclass(x)
]
for attack in ATTACK_CLASSES:
  setattr(sys.modules[__name__], attack.__name__, attack)

""" Arguemnts """
parser = argparse.ArgumentParser()

# Directory
parser.add_argument('--model_dir', default='models/adv_trained', help='Model directory', type=str)
parser.add_argument('--data_dir', default='cifar10_data', help='Data directory', type=str)
parser.add_argument('--save_dir', default='results', help='Save directory', type=str)
parser.add_argument('--save_img', dest='save_img', action='store_true', help='Save adversarial images')

# Experiment Setting
parser.add_argument('--sample_size', default=500, help='Sample size', type=int)
parser.add_argument('--batch_size', default=100, help='Batch size(PGD)', type=int)

# Attack
parser.add_argument('--attack', default='LazyLocalSearchBlockAttack', help='Attack type', type=str)
parser.add_argument('--loss_func', default='xent', help='Loss function', type=str)
parser.add_argument('--epsilon', default=8, help="Epsilon", type=int)

# Block attack
parser.add_argument('--num_steps_outer', default=5, help='The number of steps(outer loop)', type=int)
parser.add_argument('--num_steps_inner', default=2, help='The number of steps(inner loop)', type=int)
args = parser.parse_args()

def get_save_file_path(args):
  file_path = params2id(args.attack, args.epsilon, args.loss_func, args.num_steps_outer, args.num_steps_inner)
  return file_path

""" Main script """
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  #  tf.set_random_seed(0)
  #  np.random.seed(0)
  tf.logging.info('Adversarial attack on CIFAR10')
  
  # Print arguemnts
  for key, val in vars(args).items():
    tf.logging.info('{}={}'.format(key, val))

  # Load pretrained model
  model_file = tf.train.latest_checkpoint(args.model_dir)
  if model_file is None:
    tf.logging.info('No model found')
    sys.exit()
 
  # Construct multiple graphs(models) and sessions 
  models = []
  graphes = []
  sesses = []
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
    
  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    for i in range(4):
      graph = tf.Graph()
      with graph.as_default(): 
        with tf.device('/gpu:'+str(i)):
          model = Model(mode='eval')
          models.append(model)
        sess = tf.Session(config=config)
        sesses.append(sess)
      graphes.append(graph)

  # Main model and session
  model = models[0]
  sess = sesses[0]
 
  # Restore parameters
  for i in range(4):
    with graphes[i].as_default():
      saver = tf.train.Saver(tf.global_variables())
      saver.restore(sesses[i], model_file)
   
  # Create attack
  attack_class = getattr(sys.modules[__name__], args.attack)
  attack = attack_class(models, epsilon=args.epsilon, loss_func=args.loss_func, 
    num_steps_outer=args.num_steps_outer, num_steps_inner=args.num_steps_inner)
  
  # Load dataset
  cifar = cifar10_input.CIFAR10Data(args.data_dir)

  # Create save file path
  save_file_path = os.path.join(args.save_dir, get_save_file_path(args))

  # Iterate over the samples batch-by-batch
  batch_size = 1
  num_batches = int(math.ceil(args.sample_size / batch_size))

  bstart = 0
  x_full_batch = []
  y_full_batch = []
  total_queries = 0
  total_success_queries = 0
  total_corr = 0
  total_runtime = 0
    
  while(True):
    x_candid = cifar.eval_data.xs[bstart:bstart+100]
    x_candid = np.ndarray.astype(x_candid, np.int32)
    y_candid = cifar.eval_data.ys[bstart:bstart+100]
    feed = {
      model.x_input: x_candid,
      model.y_input: y_candid
    }
    mask = sess.run(model.correct_prediction, feed)
    x_masked = x_candid[mask]
    y_masked = y_candid[mask]

    if bstart == 0:
      x_full_batch = x_masked[:min(args.sample_size, len(x_masked))]
      y_full_batch = y_masked[:min(args.sample_size, len(y_masked))]
    else:
      idx = min(args.sample_size-len(x_full_batch), len(x_masked))
      x_full_batch = np.concatenate([x_full_batch, x_masked[:idx]], axis=0)
      y_full_batch = np.concatenate([y_full_batch, y_masked[:idx]], axis=0)

    bstart += 100
    if len(x_full_batch)>= args.sample_size:
      break
  
  # Run batch
  tf.logging.info('Iterating over {} batches'.format(num_batches))

  adv_images = []

  for ibatch in range(num_batches):
    tf.logging.info("")
    bstart = ibatch * batch_size
    bend = min(bstart + batch_size, args.sample_size)
    tf.logging.info('Batch {0}, Batch size: {1}'.format(ibatch, bend - bstart))
    
    x_batch = x_full_batch[bstart:bend, ...]
    y_batch = y_full_batch[bstart:bend]
    
    tf.logging.info('Attack starts')
    start = time.time()
    x_batch_adv, num_queries = attack.perturb(x_batch, y_batch, sesses)
    assert np.amax(np.abs(x_batch_adv-x_batch)) <= args.epsilon
    adv_images.append(x_batch_adv)
    end = time.time()
    tf.logging.info('Attack finishes, Time taken: {0}'.format(end-start))   
  
    feed = {
      model.x_input: x_batch_adv,
      model.y_input: y_batch
    }
    correct_prediction, num_correct = sess.run([model.correct_prediction, model.num_correct], feed)
    total_corr += num_correct
    total_queries += np.sum(num_queries)
    total_success_queries += np.sum((1-correct_prediction)*num_queries)
    total_runtime += np.sum((1-correct_prediction)*(end-start))
     
    tf.logging.info('Num of queries: {0}, Num of correct: {1}/{2}, Net accuracy: {3:2f}%'.format(
      np.sum(num_queries), num_correct, bend-bstart, 100.0*total_corr/(ibatch+1)))
  
  tf.logging.info("")
  average_queries = total_queries / args.sample_size
  average_success_queries = total_success_queries / (args.sample_size - total_corr)
  accuracy = total_corr / args.sample_size
  average_success_runtime = total_runtime / (args.sample_size - total_corr)

  tf.logging.info('Average number of queries: {0:2f}'.format(average_queries))
  tf.logging.info('Average number of queries(only successful attack): {0:2f}'.format(average_success_queries))
  tf.logging.info('Average runtime(only successful attack): {0:2f}s'.format(average_success_runtime))
  tf.logging.info('Net adv accruracy: {0:2f}%'.format(100.0 * accuracy))

  # Save adversarial images
  if args.save_img:
    tf.logging.info('Saving adversarial images')
    adv_images = np.concatenate(adv_images, axis=0)
    np.save(save_file_path, adv_images)
