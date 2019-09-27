"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import sys
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

import cifar10_input

from infer_model import Model as Safe_model
from infer_target import Model as Target_model

from utils import infer_file_name

import argparse

MODEL_PATH = './models/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # save & load path
    parser.add_argument('--data_path', default='../cifar10_data', type=str)
    parser.add_argument('--model_dir', default='naturally_trained', type=str)
    parser.add_argument('--save_dir', default='safe_net', type=str)
    parser.add_argument('--no_overwrite', action='store_true')

    # training parameters
    parser.add_argument('--tf_random_seed', default=451760341, type=int)
    parser.add_argument('--np_random_seed', default=216105420, type=int)
    parser.add_argument('--max_num_training_steps', default=80000, type=int)
    parser.add_argument('--num_output_steps', default=1, type=int)
    parser.add_argument('--num_summary_steps', default=100, type=int)
    parser.add_argument('--num_checkpoint_steps', default=1000, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--num_eval_examples', default=10000, type=int)
    parser.add_argument('--training_batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--eval_on_cpu', action='store_true')
    parser.add_argument('--delta', default=40, type=int)

    # pgd settings
    parser.add_argument('--eps', default=8.0, type=float)
    parser.add_argument('--num_steps', default=10, type=int)
    parser.add_argument('--step_size', default=2.0, type=float)
    parser.add_argument('--random_start', action='store_false')

    args = parser.parse_args()

    for key, val in vars(args).items():
        print('{}={}'.format(key, val))

# seeding randomness
tf.set_random_seed(args.tf_random_seed)
np.random.seed(args.np_random_seed)

# Setting up training parameters
max_num_training_steps = args.max_num_training_steps
num_output_steps = args.num_output_steps
num_summary_steps = args.num_summary_steps
num_checkpoint_steps = args.num_checkpoint_steps
lr = args.lr
data_path = args.data_path
training_batch_size = args.training_batch_size
eval_batch_size = args.eval_batch_size


# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.train.get_or_create_global_step()

model = Target_model('eval')
full_model = Safe_model('train', model, args)

# Setting up the optimizer
boundaries = [0, 40000, 60000]
values = [lr, lr/10, lr/100]
boundaries = boundaries[1:]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)

# set up metrics
total_loss = full_model.safe_pgd_mean_xent
safe_pgd_acc = full_model.safe_pgd_accuracy
orig_acc = full_model.orig_accuracy
safe_acc = full_model.safe_accuracy
l2_dist = tf.reduce_mean(tf.norm((full_model.x_safe-full_model.x_input)/255, axis=0))

# Setting up the Tensorboard and checkpoint outputs
meta_name = infer_file_name(args)

model_dir = MODEL_PATH + args.model_dir
save_dir = MODEL_PATH + args.save_dir + meta_name
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    if args.no_overwrite:
        print('folder already exists!')
        sys.exit()
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver()
train_summaries = [
    tf.summary.scalar('acc orig', full_model.orig_accuracy),
    tf.summary.scalar('acc safe', full_model.safe_accuracy),
    tf.summary.scalar('acc safe_pgd', full_model.safe_pgd_accuracy),
    tf.summary.scalar('loss', total_loss),
    tf.summary.scalar('l2 dist', l2_dist),
    tf.summary.scalar('image', full_model.x_safe),
]
train_merged_summaries = tf.summary.merge(train_summaries)
eval_summaries = [
    tf.summary.scalar('acc orig (eval)', full_model.orig_accuracy),
    tf.summary.scalar('acc safe (eval)', full_model.safe_accuracy),
    tf.summary.scalar('acc safe_pgd (eval)', full_model.safe_pgd_accuracy),
    tf.summary.scalar('loss (eval)', total_loss),
    tf.summary.scalar('l2 dist (eval)', l2_dist),
    tf.summary.scalar('image (eval)', full_model.x_safe),
]
eval_merged_summaries = tf.summary.merge(eval_summaries)

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

model_file = tf.train.latest_checkpoint(model_dir)
if model_file is None:
    print('No model found')
    sys.exit()

with tf.Session() as sess:

    # initialize data augmentation
    cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

    # set index for evaluation data
    eval_indice = np.array([i for i in range(len(raw_cifar.eval_data.ys))])
    np.randon.shuffle(eval_indice)

    # Initialize the summary writer, global variables, and our time counter.
    summary_writer = tf.summary.FileWriter(save_dir, sess.graph)

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

    training_time = 0.0

    # Main training loop
    for ii in range(max_num_training_steps):
        x_batch, y_batch = cifar.train_data.get_next_batch(training_batch_size,
                                                           multiple_passes=True)

        # Actual training step
        nat_dict = {full_model.x_input: x_batch,
                    full_model.y_input: y_batch}

        assert 0 <= np.amin(x_batch) and np.amax(x_batch) <= 255.0

        start = timer()
        _, total_loss_batch, safe_pgd_acc_batch, orig_acc_batch, safe_acc_batch, \
        x_safe, x_safe_pgd, l2_dist_batch, train_merged_summaries_batch = \
            sess.run([train_step, total_loss, safe_pgd_acc, orig_acc, safe_acc,
                      full_model.x_safe, full_model.x_safe_pgd, l2_dist, train_merged_summaries], feed_dict=nat_dict)
        end = timer()

        assert 0 <= np.amin(x_safe) and np.amax(x_safe) <= 255.0
        assert 0 <= np.amin(x_safe_pgd) and np.amax(x_safe_pgd) <= 255.0

        training_time += end - start

        # Output to stdout
        if ii % num_output_steps == 0:
            print('Step {}:    ({})'.format(ii, datetime.now()))
            print('    orig accuracy {:.4}%'.format(orig_acc_batch * 100))
            print('    safe accuracy {:.4}%'.format(safe_acc_batch * 100))
            print('    safe_pgd accuracy {:.4}%'.format(safe_pgd_acc_batch * 100))
            print('    l2 dist {:.4}'.format(l2_dist_batch))
            print('    total loss {:.6}'.format(total_loss_batch))
            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * training_batch_size / training_time))
                training_time = 0.0

        # Tensorboard summaries
        if ii % num_summary_steps == 0:
            summary_writer.add_summary(train_merged_summaries_batch, global_step.eval(sess))
            # evaluate on test set
            eval_bstart = (ii//num_summary_steps)*eval_batch_size
            eval_bend = (ii//num_summary_steps+1)*eval_batch_size

            eval_x_batch = raw_cifar.eval_data.xs[eval_indice[eval_bstart:eval_bend]]
            eval_y_batch = raw_cifar.eval_data.ys[eval_indice[eval_bstart:eval_bend]]
            eval_dict = {full_model.x_input: eval_x_batch,
                         full_model.y_input: eval_y_batch}
            total_loss_batch, safe_pgd_acc_batch, orig_acc_batch, safe_acc_batch, \
                x_safe, x_safe_pgd, l2_dist_batch, eval_merged_summaries_batch = \
                sess.run([total_loss, safe_pgd_acc, orig_acc, safe_acc,
                          full_model.x_safe, full_model.x_safe_pgd, l2_dist, eval_merged_summaries], feed_dict=eval_dict)
            
            print('    orig accuracy (eval) {:.4}%'.format(orig_acc_batch * 100))
            print('    safe accuracy (eval) {:.4}%'.format(safe_acc_batch * 100))
            print('    safe_pgd accuracy (eval) {:.4}%'.format(safe_pgd_acc_batch * 100))
            print('    l2 dist (eval) {:.4}'.format(l2_dist_batch))
            print('    total loss (eval) {:.6}'.format(total_loss_batch))
            
            summary_writer.add_summary(eval_merged_summaries_batch, global_step.eval(sess))


        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
            saver.save(sess,
                       os.path.join(save_dir, 'checkpoint'),
                       global_step=global_step)
