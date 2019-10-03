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

from utils import infer_file_name, load_imp_data

import argparse

MODEL_PATH = './models/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # save & load path
    parser.add_argument('--data_path', default='../cifar10_data', type=str)
    parser.add_argument('--model_dir', default='naturally_trained', type=str)
    parser.add_argument('--save_dir', default='safe_net', type=str)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true')

    # training parameters
    parser.add_argument('--tf_random_seed', default=451760341, type=int)
    parser.add_argument('--np_random_seed', default=216105420, type=int)
    parser.add_argument('--max_num_training_steps', default=80000, type=int)
    parser.add_argument('--num_output_steps', default=1, type=int)
    parser.add_argument('--num_summary_steps', default=100, type=int)
    parser.add_argument('--num_checkpoint_steps', default=1000, type=int)
    parser.add_argument('--g_lr', default=1e-3, type=float)
    parser.add_argument('--num_eval_examples', default=10000, type=int)
    parser.add_argument('--training_batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--eval_on_cpu', action='store_true')
    parser.add_argument('--delta', default=40, type=int)

    # discriminator settings
    parser.add_argument('--use_d', action='store_true')
    parser.add_argument('--d_lr', default=1e-3, type=float)
    parser.add_argument('--patch', action='store_true', help='use patch discriminator, (2x2)')
    parser.add_argument('--l1_loss', action='store_true', help='use l1 loss on infer(x) and maml(x)')
    parser.add_argument('--g_weight', default=1, type=float, help='loss weight for generator')
    parser.add_argument('--d_weight', default=1, type=float, help='loss weight for discriminator')
    parser.add_argument('--l1_weight', default=1, type=float, help='loss weight for l1')

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
g_lr = args.g_lr
d_lr = args.d_lr
data_path = args.data_path
training_batch_size = args.training_batch_size
eval_batch_size = args.eval_batch_size


# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
imp_cifar = load_imp_data(args)

global_step = tf.train.get_or_create_global_step()

model = Target_model('eval')
full_model = Safe_model('train', model, args)

# Setting up the optimizer
boundaries = [0, 40000, 60000]
boundaries = boundaries[1:]
g_values = [g_lr, g_lr/10, g_lr/100]
g_learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    g_values)
d_values = [d_lr, d_lr/10, d_lr/100]
d_learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    d_values)

# set up metrics
safe_pgd_loss = full_model.safe_pgd_mean_xent
total_loss = safe_pgd_loss
if args.use_d:
    d_loss = args.g_weight * full_model.d_loss
    g_loss = args.d_weight * full_model.g_loss
    l1_loss = args.l1_weight * tf.losses.absolute_difference(full_model.x_input_alg, full_model.x_safe)
    total_loss += d_loss
    total_loss += g_loss
    if args.l1_loss:
        total_loss += l1_loss

safe_pgd_acc = full_model.safe_pgd_accuracy
orig_acc = full_model.orig_accuracy
safe_acc = full_model.safe_accuracy
l2_dist = tf.reduce_mean(tf.norm(tf.reshape((full_model.x_safe-full_model.x_input)/255, shape=[-1, 32*32*3]), axis=1))

# Setting up the Tensorboard and checkpoint outputs
meta_name = infer_file_name(args)

model_dir = MODEL_PATH + args.model_dir
if args.save:
    save_dir = MODEL_PATH + args.save_dir + '/' + meta_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        if not args.overwrite:
            print('folder already exists!')
            sys.exit()
        elif not args.resume:
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
    tf.summary.scalar('safe loss', safe_pgd_loss),
    tf.summary.scalar('l2 dist', l2_dist),
    tf.summary.image('image', full_model.x_safe),
]
if args.use_d:
    train_summaries.append(tf.summary.scalar('d loss', d_loss))
    train_summaries.append(tf.summary.scalar('g loss', g_loss))
    train_summaries.append(tf.summary.scalar('l1 loss', l1_loss))
    train_summaries.append(tf.summary.scalar('total loss', total_loss))
train_merged_summaries = tf.summary.merge(train_summaries)

eval_summaries = [
    tf.summary.scalar('acc orig (eval)', full_model.orig_accuracy),
    tf.summary.scalar('acc safe (eval)', full_model.safe_accuracy),
    tf.summary.scalar('acc safe_pgd (eval)', full_model.safe_pgd_accuracy),
    tf.summary.scalar('safe loss (eval)', safe_pgd_loss),
    tf.summary.scalar('l2 dist (eval)', l2_dist),
    tf.summary.image('image (eval)', full_model.x_safe),
]
eval_merged_summaries = tf.summary.merge(eval_summaries)

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

model_file = tf.train.latest_checkpoint(model_dir)
if model_file is None:
    print('No model found')
    sys.exit()

with tf.Session() as sess:

    # no data augmentation
    # cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)
    cifar = raw_cifar

    # Initialize the summary writer, global variables, and our time counter.
    if args.save:
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
    
    variables_to_train_g = [var for var in trainable_variables if (var.name not in restore_vars_name_list and
                                                                 'generator' in var.name)]
    variables_to_train_d = [var for var in trainable_variables if (var.name not in restore_vars_name_list and
                                                                 'discriminator' in var.name)]

    train_step_g = tf.train.AdamOptimizer(g_learning_rate).minimize(
        total_loss,
        global_step=global_step,
        var_list=variables_to_train_g)

    if args.use_d:
        train_step_d = tf.train.AdamOptimizer(d_learning_rate).minimize(
            total_loss,
            global_step=global_step,
            var_list=variables_to_train_d)

    sess.run(tf.global_variables_initializer())
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(sess, model_file)
    
    if args.resume:
        save_file = tf.train.latest_checkpoint(save_dir)
        full_saver = tf.train.Saver(variables_to_train + [global_step])
        full_saver.restore(sess, save_file)

    print('restore success!')

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    training_time = 0.0

    # Main training loop
    for ii in range(max_num_training_steps):
        print(global_step.eval(sess))

        # Get data
        x_batch, y_batch, indices = cifar.train_data.get_next_batch(training_batch_size,
                                                           multiple_passes=True,
                                                           get_indices=True)

        if args.use_d:
            imp_batch = imp_cifar[indices, ...]

            nat_dict = {full_model.x_input: x_batch,
                        full_model.y_input: y_batch,
                        full_model.x_input_alg: imp_batch}
        else:
            nat_dict = {full_model.x_input: x_batch,
                        full_model.y_input: y_batch}

        # Sanity check
        assert 0 <= np.amin(x_batch) and np.amax(x_batch) <= 255.0
        if args.use_d:
            assert 0 <= np.amin(imp_batch) and np.amax(imp_batch) <= 255.0
            assert np.amax(np.abs(imp_batch-x_batch)) <= 40

        # Train
        start = timer()
        _, _, safe_pgd_loss_batch, safe_pgd_acc_batch, orig_acc_batch, safe_acc_batch, \
            x_safe, x_safe_pgd, l2_dist_batch, train_merged_summaries_batch = \
            sess.run([train_step_g, extra_update_ops, safe_pgd_loss, safe_pgd_acc, orig_acc, safe_acc,
                      full_model.x_safe, full_model.x_safe_pgd, l2_dist, train_merged_summaries],
                     feed_dict=nat_dict)

        if args.use_d:
            _, _, safe_pgd_loss_batch, safe_pgd_acc_batch, orig_acc_batch, safe_acc_batch, \
                x_safe, x_safe_pgd, l2_dist_batch, train_merged_summaries_batch, \
                total_loss_batch, d_loss_batch, g_loss_batch, l1_loss_batch = \
                sess.run([train_step_d, extra_update_ops, safe_pgd_loss, safe_pgd_acc, orig_acc, safe_acc,
                         full_model.x_safe, full_model.x_safe_pgd, l2_dist, train_merged_summaries,
                         total_loss, d_loss, g_loss, l1_loss],
                         feed_dict=nat_dict)
        end = timer()

        assert 0 <= np.amin(x_safe) and np.amax(x_safe) <= 255.0
        assert 0 <= np.amin(x_safe_pgd) and np.amax(x_safe_pgd) <= 255.0

        training_time += end - start

        # Output to stdout
        if ii % num_output_steps == 0:
            print('Step {}:    ({})'.format(ii, datetime.now()))
            print('    orig accuracy {:.4}%'.format(orig_acc_batch * 100))
            print('    safe accuracy {:.4}%'.format(safe_acc_batch * 100))
            print('    safe(pgd) accuracy {:.4}%'.format(safe_pgd_acc_batch * 100))
            print('    l2 dist {:.4}'.format(l2_dist_batch))
            print('    safe(pgd) loss {:.6}'.format(safe_pgd_loss_batch))
            if args.use_d:
                print('    d loss {:.6}'.format(d_loss_batch))
                print('    g loss {:.6}'.format(g_loss_batch))
                print('    l1 loss {:.6}'.format(l1_loss_batch))
                print('    total loss {:.6}'.format(total_loss_batch))
            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * training_batch_size / training_time))
                training_time = 0.0

            #sys.exit()
        # Tensorboard summaries
        if ii % num_summary_steps == 0:
            if args.save:
                summary_writer.add_summary(train_merged_summaries_batch, global_step.eval(sess))
            # evaluate on test set
            #eval_bstart = (ii//num_summary_steps)*eval_batch_size
            #eval_bend = (ii//num_summary_steps+1)*eval_batch_size
            
            eval_x_batch, eval_y_batch = raw_cifar.eval_data.get_next_batch(eval_batch_size, multiple_passes=True)
            eval_dict = {full_model.x_input: eval_x_batch,
                         full_model.y_input: eval_y_batch}
            safe_pgd_loss_batch, safe_pgd_acc_batch, orig_acc_batch, safe_acc_batch, \
                x_safe, x_safe_pgd, l2_dist_batch, eval_merged_summaries_batch = \
                sess.run([safe_pgd_loss, safe_pgd_acc, orig_acc, safe_acc,
                          full_model.x_safe, full_model.x_safe_pgd, l2_dist, eval_merged_summaries], feed_dict=eval_dict)
            
            print('    orig accuracy (eval) {:.4}%'.format(orig_acc_batch * 100))
            print('    safe accuracy (eval) {:.4}%'.format(safe_acc_batch * 100))
            print('    safe(pgd) accuracy (eval) {:.4}%'.format(safe_pgd_acc_batch * 100))
            print('    l2 dist (eval) {:.4}'.format(l2_dist_batch))
            print('    safe(pgd) loss (eval) {:.6}'.format(safe_pgd_loss_batch))
            '''
            if args.use_d:
                print('    d loss (eval) {:.6}'.format(d_loss_batch))
                print('    g loss (eval) {:.6}'.format(g_loss_batch))
                print('    total loss (eval) {:.6}'.format(total_loss_batch))
            '''

            if args.save:
                summary_writer.add_summary(eval_merged_summaries_batch, global_step.eval(sess))


        # Write a checkpoint
        if ii % num_checkpoint_steps == 0 and args.save:
            saver.save(sess,
                       os.path.join(save_dir, 'checkpoint'),
                       global_step=global_step)
