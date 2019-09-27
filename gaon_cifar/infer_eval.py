"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math

import tensorflow as tf
import numpy as np

import cifar10_input

from infer_model import generator as Generator
from model import Model as Model

from utils import infer_file_name

import argparse

from pgd_attack import LinfPGDAttack

MODEL_PATH = './models/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # save & load path
    parser.add_argument('--data_path', default='../cifar10_data', type=str)
    parser.add_argument('--model_dir', default='naturally_trained', type=str)
    parser.add_argument('--save_dir', default='safe_net', type=str, help='safe_net saved folder')
    parser.add_argument('--corr_only', action='store_false')
    parser.add_argument('--fail_only', action='store_true')

    # eval parameters
    parser.add_argument('--tf_random_seed', default=451760341, type=int)
    parser.add_argument('--np_random_seed', default=216105420, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--eval_on_cpu', action='store_true')
    parser.add_argument('--delta', default=40, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--sample_size', default=1000, type=int)

    # pgd settings
    parser.add_argument('--eps', default=8.0, type=float)
    parser.add_argument('--num_steps', default=10, type=int)
    parser.add_argument('--step_size', default=2.0, type=float)
    parser.add_argument('--restarts', default=20, type=int)

    args = parser.parse_args()

    for key, val in vars(args).items():
        print('{}={}'.format(key, val))

# seeding randomness
tf.set_random_seed(args.tf_random_seed)
np.random.seed(args.np_random_seed)

# Setting up training parameters
lr = args.lr
data_path = args.data_path
eval_batch_size = args.eval_batch_size


# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.train.get_or_create_global_step()

model = Model('eval')

x_input = tf.placeholder(
    tf.float32,
    shape=[None, 32, 32, 3]
)
y_input = tf.placeholder(
    tf.int64,
    shape=None
)
generator = tf.make_template('generator', Generator, f_dim=64, output_size=32, c_dim=3, is_training=False)

pgd = LinfPGDAttack(model,
                    args.eps,
                    args.num_steps,
                    args.step_size,
                    True,
                    'xent')

# Setting up the optimizer
boundaries = [0, 40000, 60000]
values = [lr, lr/10, lr/100]
boundaries = boundaries[1:]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)

# Setting up the Tensorboard and checkpoint outputs
meta_name = infer_file_name(args)

model_dir = MODEL_PATH + args.save_dir + '/' + meta_name
if not os.path.exists(model_dir):
    print(model_dir)
    print("incorrect path!")
    sys.exit()

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver()
cifar = cifar10_input.CIFAR10Data(data_path)

model_file = tf.train.latest_checkpoint(model_dir)
if model_file is None:
    print('No model found')
    sys.exit()

with tf.Session() as sess:

    # Restore variables if can, set optimizer
    reader = tf.train.NewCheckpointReader(model_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    restore_vars_name_list = []
    for var_name, saved_var_name in var_names:
        curr_var = tf.get_default_graph().get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name] and 'global_step' not in saved_var_name:
            restore_vars.append(curr_var)
            restore_vars_name_list.append(saved_var_name + ':0')

    trainable_variables = tf.trainable_variables()
    variables_to_train = [var for var in trainable_variables if var.name not in restore_vars_name_list]

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        total_loss,
        global_step=global_step,
        var_list=variables_to_train)

    sess.run(tf.global_variables_initializer())
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(sess, model_file)
    print('restore success!')

    num_eval_examples = args.sample_size

    if args.corr_only:
        if args.model_dir == 'naturally_trained':
            indices = np.load('/data/home/gaon/lazy-attack/cifar10_data/nat_indices_untargeted.npy')
        else:
            indices = np.load('/data/home/gaon/lazy-attack/cifar10_data/indices_untargeted.npy')
    elif args.fail_only:
        if args.model_dir == 'naturally_trained':
            indices = np.load('/data/home/gaon/lazy-attack/cifar10_data/fail_nat_indices_untargeted.npy')
        else:
            indices = np.load('/data/home/gaon/lazy-attack/cifar10_data/fail_indices_untargeted.npy')
    else:
        indices = [i for i in range(args.sample_size + args.bstart)]

    # load data
    bstart = args.bstart
    while True:
        x_candid = cifar.eval_data.xs[indices[bstart:bstart + 100]]
        y_candid = cifar.eval_data.ys[indices[bstart:bstart + 100]]
        mask, logits = sess.run([model.correct_prediction, model.pre_softmax],
                                feed_dict={model.x_input: x_candid,
                                           model.y_input: y_candid})
        print(sum(mask))
        if args.corr_only and (np.mean(mask) < 1.0 - 1E-6):
            raise Exception
        if args.fail_only and (np.mean(mask) > 0.0 + 1E-6):
            raise Exception
        if bstart == args.bstart:
            x_full_batch = x_candid[:min(num_eval_examples, len(x_candid))]
            y_full_batch = y_candid[:min(num_eval_examples, len(y_candid))]
            logit_full_batch = logits[:min(num_eval_examples, len(logits))]
        else:
            index = min(num_eval_examples - len(x_full_batch), len(x_candid))
            x_full_batch = np.concatenate((x_full_batch, x_candid[:index]))
            y_full_batch = np.concatenate((y_full_batch, y_candid[:index]))
            logit_full_batch = np.concatenate((logit_full_batch, logits[:index]))
        bstart += 100
        if (len(x_full_batch) >= num_eval_examples) or bstart >= len(indices):
            break

        # Adjust num_eval_examples. Iterate over the samples batch-by-batch
        num_eval_examples = len(x_full_batch)

        if num_eval_examples > args.batch_size :
            eval_batch_size = args.eval_batch_size
        else:
            eval_batch_size = min(args.eval_batch_size, num_eval_examples)

        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        print('Iterating over {} batches'.format(num_batches))

        x_full_batch = x_full_batch.astype(np.float32)

        full_mask = []

        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch {} - {}'.format(bstart, bend-1))

            x_batch = x_full_batch[bstart:bend, :]
            y_batch = y_full_batch[bstart:bend]

            mask = np.array([True for _ in range(len(y_batch))])

            for _ in range(args.restarts):
                
                x_batch_attacked = pgd.perturb(x_batch, y_batch, sess,
                                       proj=True, reverse=False, rand=True)

                correct_prediction = sess.sun(model.correct_prediction,
                                               feed_dict={model.x_input: x_batch_attacked,
                                                          model.y_input: y_batch})

                mask *= correct_prediction

            full_mask.append(mask)

            print('{}/{} safe'.format(np.sum(mask), np.size(mask)))

        full_mask = np.concatenate(full_mask)

        print("evaluation accuracy: {:.2f}".format(np.mean(full_mask)*100))

