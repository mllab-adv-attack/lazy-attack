import argparse
import inspect
import numpy as np
import sys
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

# for passing boolean values to *_slurm.py in hyperparameter searching
def str2bool(key):
  return key.lower() in ('yes', 'true', 'y', 't')

""" Arguemnts """
parser = argparse.ArgumentParser()

# Directory
parser.add_argument('--model_dir', default='models/adv_trained', help='Model directory', type=str)
parser.add_argument('--data_dir', default='../cifar10_data', help='Data directory', type=str)

# Experiment Setting
parser.add_argument('--img_index_start', default=0, type=int)
parser.add_argument('--sample_size', default=500, help='Sample size', type=int)
parser.add_argument('--save_img', default='False', help='Save adversarial images', type=str2bool)
parser.add_argument('--save_npy', default='False', help='Save images in numpy', type=str2bool)
parser.add_argument('--save_dir', default='./out/')

# Attack
parser.add_argument('--attack', default='LazyLocalSearchBlockAttack', help='Attack type', type=str)
parser.add_argument('--loss_func', default='xent', help='Loss function', type=str)
parser.add_argument('--epsilon', default=8, help="Epsilon", type=int)
parser.add_argument('--max_queries', default=20000, type=int)
parser.add_argument('--targeted', default='False', type=str2bool)

# Block attack
parser.add_argument('--admm', default='False', help='use admm', type=str2bool)
parser.add_argument('--admm_block_size', default=16, help='block size for admm', type=int)
parser.add_argument('--partition', default='basic', help='block partitioning scheme', type=str)
parser.add_argument('--admm_iter', default=100, help='admm max iteration', type=int)
parser.add_argument('--overlap', default=0, help='overlap size', type=int)
parser.add_argument('--admm_rho', default=1e-12, help='admm rho', type=float)
parser.add_argument('--admm_tau', default=1.5, help='admm tau', type=float)
parser.add_argument('--gpus', default=2, help='number of gpus to use', type=int)
parser.add_argument('--parallel', default=4, help='number of parallel threads to use', type=int)

# Lazy Local Search Batch
parser.add_argument('--lls_block_size', default=4, help='initial block size for lls', type=int)
parser.add_argument('--lls_iter', default=1, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--no_hier', default='False', type=str2bool)
args = parser.parse_args()

""" Main script """
if __name__ == '__main__':
  # Set verbosity
  tf.logging.set_verbosity(tf.logging.INFO)
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

  # Assign 1~2 threads per gpu
  assert (args.parallel >= args.gpus) and (args.parallel % args.gpus == 0) and (args.parallel // args.gpus <= 2)
    
  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    for i in range(args.parallel):
      graph = tf.Graph()
      with graph.as_default(): 
        with tf.device('/gpu:'+str(i%args.gpus)):
          model = Model(mode='eval')
          models.append(model)
        sess = tf.Session(config=config)
        sesses.append(sess)
      graphes.append(graph)

  # Main model and session
  model = models[0]
  sess = sesses[0]
 
  # Restore parameters
  for i in range(args.gpus):
    with graphes[i].as_default():
      saver = tf.train.Saver(tf.global_variables())
      saver.restore(sesses[i], model_file)
   
  # Create attack
  attack_class = getattr(sys.modules[__name__], args.attack)
  attack = attack_class(models, args)
  
  # Load dataset
  cifar = cifar10_input.CIFAR10Data(args.data_dir)

  # Load indices
  indices = np.load('../cifar10_data/indices_untargeted.npy')

  count = 0
  index = args.img_index_start
  total_num_corrects = 0
  total_num_queries = []
  total_parallel_queries = []
  index_to_num_queries = {}
  total_times = []

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

    adv_img, num_queries, parallel_queries, success, time = attack.perturb(initial_img,
                                                                           orig_class,
                                                                           indices[index],
                                                                           sesses)
    assert np.amax(np.abs(adv_img - initial_img)) <= args.epsilon
    assert np.amax(adv_img) <= 255
    assert np.amax(adv_img) >= 0
    p = sess.run(model.predictions, feed_dict={model.x_input: adv_img})

    if args.save_npy and count%10==0:
      np.save('./save/adv_imgs/adv_img_{}'.format(count), (adv_img-initial_img)[0, ...])
    
    if args.save_img:
      nat_image = Image.fromarray(np.ndarray.astype(initial_img[0, ...] * 255, np.uint8))
      nat_image.save(args.save_dir + '/nat/{}_nat.jpg'.format(indices[index]))
      adv_image = Image.fromarray(np.ndarray.astype(adv_img[0, ...] * 255, np.uint8))
      adv_image.save(args.save_dir + '/adv/{}_adv.jpg'.format(indices[index]))

    if success:
      total_num_corrects += 1
      total_num_queries.append(num_queries)
      total_parallel_queries.append(parallel_queries)
      total_times.append(time)
      index_to_num_queries[indices[index]] = num_queries
      average_queries = 0 if len(total_num_queries) == 0 else np.mean(total_num_queries)
      median_queries = 0 if len(total_num_queries) == 0 else np.median(total_num_queries)
      average_parallel_queries = 0 if len(total_parallel_queries) == 0 else np.mean(total_parallel_queries)
      average_time = 0 if len(total_times) == 0 else np.mean(total_times)
      tf.logging.info('Attack success, final class: {}, avg queries: {:.5f}, med queries: {}, '
                      'avg per-gpu queries: {:.0f}, success rate: {:.4f}, avg time: {:.2f}'.format(
        p[0], average_queries, median_queries, average_parallel_queries, total_num_corrects / count, average_time))
    else:
      index_to_num_queries[indices[index]] = -1
      average_queries = 0 if len(total_num_queries) == 0 else np.mean(total_num_queries)
      median_queries = 0 if len(total_num_queries) == 0 else np.median(total_num_queries)
      average_parallel_queries = 0 if len(total_parallel_queries) == 0 else np.mean(total_parallel_queries)
      average_time = 0 if len(total_times) == 0 else np.mean(total_times)
      tf.logging.info('Attack fail, final class: {}, avg queries: {:.5f}, med queries: {}, '
                      'avg per-gpu queries: {:.0f}, success rate: {:.4f}, avg time: {:.2f}'.format(
        p[0], average_queries, median_queries, average_parallel_queries, total_num_corrects / count, average_time))

    index += 1

  admm = 'admm' if args.admm else 'basic'

  filename = args.save_dir + '/lls_untargeted_{}_b{}_o{}_r{}_t{}_i{}.npy'.format(
    admm, args.admm_block_size, args.overlap, args.admm_rho, args.admm_tau, args.img_index_start + args.sample_size)
  np.save(filename, index_to_num_queries)
