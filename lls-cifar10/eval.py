import argparse
import math
import numpy as np
import sys
import tensorflow as tf

import cifar10_input
from model import Model

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default='./models/adv_trained', type=str)
parser.add_argument('--data_dir', default='./cifar10_data', type=str)
parser.add_argument('--batch_size', default=100, type=int)

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

  # Load dataset
  cifar = cifar10_input.CIFAR10Data(args.data_dir)

  # Load indices
  num_images = len(cifar.eval_data.xs)
  num_batches = int(math.ceil(num_images//args.batch_size))
  indices = []

  for i in range(num_batches):
    bstart = i*args.batch_size
    bend = min(bstart+args.batch_size, num_images)
    
    image_batch = cifar.eval_data.xs[bstart:bend, ...]
    label_batch = cifar.eval_data.ys[bstart:bend, ...]
    
    preds = sess.run(model.predictions, feed_dict={model.x_input: image_batch, model.y_input: label_batch})
    success_indices, = np.where(preds == label_batch)
    success_indices += bstart
    success_indices = list(success_indices)
    indices += success_indices
  
  np.save('success_indices.npy', indices)
