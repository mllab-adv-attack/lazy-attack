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

from tensorflow.examples.tutorials.mnist import input_data

from infer_model import Model as Safe_model
from infer_target import Model as Target_model

from utils import infer_file_name, load_imp_data, CustomDataSet

import argparse

MODEL_PATH = './models/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # save & load path
    parser.add_argument('--model_dir', default='naturally_trained', type=str)
    parser.add_argument('--save_dir', default='safe_net', type=str)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true')

    # training parameters
    parser.add_argument('--tf_random_seed', default=451760341, type=int)
    parser.add_argument('--np_random_seed', default=216105420, type=int)
    parser.add_argument('--max_num_training_steps', default=100000, type=int)
    parser.add_argument('--num_output_steps', default=100, type=int)
    parser.add_argument('--num_summary_steps', default=100, type=int)
    parser.add_argument('--num_checkpoint_steps', default=300, type=int)
    parser.add_argument('--g_lr', default=1e-3, type=float)
    parser.add_argument('--training_batch_size', default=50, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--eval_on_cpu', action='store_true')
    parser.add_argument('--delta', default=0.3, type=float)

    # GAN settings
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--f_dim', default=64, type=int)
    parser.add_argument('--noise_only', action='store_true')
    parser.add_argument('--unet', action='store_true')
    parser.add_argument('--use_d', action='store_true')
    parser.add_argument('--use_advG', action='store_true')
    parser.add_argument('--no_lc', action='store_true')
    parser.add_argument('--advG_lr', default=1e-4, type=float)
    parser.add_argument('--d_lr', default=1e-4, type=float)
    parser.add_argument('--l1_loss', action='store_true', help='use l1 loss on infer(x) and maml(x)')
    parser.add_argument('--l2_loss', action='store_true', help='use l2 loss on infer(x) and maml(x)')
    parser.add_argument('--lp_loss', action='store_true', help='use logit pairing loss on infer(x) and maml(x)')
    parser.add_argument('--g_weight', default=1, type=float, help='loss weight for generator')
    parser.add_argument('--d_weight', default=1, type=float, help='loss weight for discriminator')
    parser.add_argument('--l1_weight', default=1, type=float, help='loss weight for l1')
    parser.add_argument('--l2_weight', default=1, type=float, help='loss weight for l2')
    parser.add_argument('--lp_weight', default=1, type=float, help='loss weight for logit pairing')

    # pgd settings
    parser.add_argument('--eps', default=0.3, type=float)
    parser.add_argument('--num_steps', default=40, type=int)
    parser.add_argument('--step_size', default=0.01, type=float)
    parser.add_argument('--random_start', action='store_true')

    args = parser.parse_args()

    for key, val in vars(args).items():
        print('{}={}'.format(key, val))

USE_ALG = True if (args.use_d or args.l1_loss or args.l2_loss or args.lp_loss) else False

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
advG_lr = args.advG_lr
training_batch_size = args.training_batch_size
eval_batch_size = args.eval_batch_size


# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False, validation_size=0, reshape=False)
if USE_ALG:
    imp_mnist = load_imp_data(args)

mnist_train = CustomDataSet(mnist.train.images, mnist.train.labels)
mnist_test = CustomDataSet(mnist.test.images, mnist.test.labels)

global_step = tf.train.get_or_create_global_step()

model = Target_model()
full_model = Safe_model('train', model, args)

# set up metrics
safe_adv_loss = full_model.safe_adv_mean_xent
safe_pgd_loss = full_model.safe_pgd_mean_xent
safe_loss = full_model.safe_mean_xent
l1_loss = tf.losses.absolute_difference(full_model.x_input_alg, full_model.x_safe)
l2_loss = tf.losses.mean_squared_error(full_model.x_input_alg, full_model.x_safe)
#l2_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((full_model.x_input_alg-full_model.x_safe)), axis=[1, 2, 3])))
lp_loss = tf.losses.mean_squared_error(full_model.safe_pre_softmax, full_model.alg_pre_softmax)


if args.no_lc:
    total_loss = 0
else:
    total_loss = safe_adv_loss

if args.l1_loss:
    total_loss += args.l1_weight * l1_loss
if args.l2_loss:
    total_loss += args.l2_weight * l2_loss
if args.lp_loss:
    total_loss += args.lp_weight * lp_loss
if args.use_advG:
    total_loss += safe_loss
if args.use_d:
    d_loss = full_model.d_loss
    g_loss = full_model.g_loss
    total_d_loss = total_loss + args.d_weight * d_loss
    total_g_loss = total_loss + args.g_weight * g_loss

safe_adv_acc = full_model.safe_adv_accuracy
safe_pgd_acc = full_model.safe_pgd_accuracy
orig_acc = full_model.orig_accuracy
safe_acc = full_model.safe_accuracy
l2_dist = tf.reduce_mean(tf.norm(tf.reshape(full_model.x_safe-full_model.x_input, shape=[-1, 28*28*1]), axis=1))

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
            shutil.rmtree(save_dir, ignore_errors=True)
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
    tf.summary.scalar('acc safe_adv', full_model.safe_adv_accuracy),
    tf.summary.scalar('acc safe_pgd', full_model.safe_pgd_accuracy),
    tf.summary.scalar('safe adv loss', safe_adv_loss),
    tf.summary.scalar('safe pgd loss', safe_pgd_loss),
    tf.summary.scalar('l2 dist', l2_dist),
    tf.summary.image('safe image', full_model.x_safe),
    tf.summary.image('orig image', full_model.x_input),
]
if args.use_d:
    train_summaries.append(tf.summary.scalar('d loss', d_loss))
    train_summaries.append(tf.summary.scalar('g loss', g_loss))
if USE_ALG:
    tf.summary.image('alg image', full_model.x_input_alg),
    train_summaries.append(tf.summary.scalar('l1 loss', l1_loss))
    train_summaries.append(tf.summary.scalar('l2 loss', l2_loss))
    train_summaries.append(tf.summary.scalar('lp loss', lp_loss))
    train_summaries.append(tf.summary.scalar('total loss', total_loss))

train_merged_summaries = tf.summary.merge(train_summaries)

eval_summaries = [
    tf.summary.scalar('acc orig (eval)', full_model.orig_accuracy),
    tf.summary.scalar('acc safe (eval)', full_model.safe_accuracy),
    tf.summary.scalar('acc safe_adv (eval)', full_model.safe_adv_accuracy),
    tf.summary.scalar('acc safe_pgd (eval)', full_model.safe_pgd_accuracy),
    tf.summary.scalar('safe adv loss (eval)', safe_adv_loss),
    tf.summary.scalar('safe pgd loss (eval)', safe_pgd_loss),
    tf.summary.scalar('l2 dist (eval)', l2_dist),
    tf.summary.image('safe image (eval)', full_model.x_safe),
    tf.summary.image('orig image (eval)', full_model.x_input),
]
eval_merged_summaries = tf.summary.merge(eval_summaries)

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

model_file = tf.train.latest_checkpoint(model_dir)
if model_file is None:
    print('No model found')
    sys.exit()

with tf.Session() as sess:

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
    variables_to_train_advG = [var for var in trainable_variables if (var.name not in restore_vars_name_list and
                                                                   'adv_generator' in var.name)]

    print(np.sum([np.prod(v.get_shape().as_list()) for v in restore_vars]))
    print(np.sum([np.prod(v.get_shape().as_list()) for v in variables_to_train]))
    print(np.sum([np.prod(v.get_shape().as_list()) for v in variables_to_train_g]))
    print(np.sum([np.prod(v.get_shape().as_list()) for v in variables_to_train_d]))
    print(np.sum([np.prod(v.get_shape().as_list()) for v in variables_to_train_advG]))

    train_step_g = tf.train.AdamOptimizer(args.g_lr).minimize(
        total_loss if not args.use_d else total_g_loss,
        global_step=global_step,
        var_list=variables_to_train_g)

    if args.use_d:
        train_step_d = tf.train.AdamOptimizer(args.d_lr).minimize(
            total_d_loss,
            var_list=variables_to_train_d)

    if args.use_advG:
        train_step_advG = tf.train.AdamOptimizer(args.advG_lr).minimize(
            -safe_adv_loss,
            var_list=variables_to_train_advG)

    sess.run(tf.global_variables_initializer())
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(sess, model_file)
    
    if args.resume:
        save_file = tf.train.latest_checkpoint(save_dir)
        full_saver = tf.train.Saver(variables_to_train + [global_step])
        full_saver.restore(sess, save_file)

    print('restore success!')

    '''
    tvar = tf.trainable_variables()
    for var in tvar:
        print(var.name)
    sys.exit()
    '''

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    training_time = 0.0

    # Main training loop
    for ii in range(max_num_training_steps):

        # Get data
        x_batch, y_batch, indices = mnist_train.get_next_batch(training_batch_size,
                                                               multiple_passes=True,
                                                               get_indices=True)

        if USE_ALG:
            imp_batch = imp_mnist[indices, ...]

            nat_dict = {full_model.x_input: x_batch,
                        full_model.y_input: y_batch,
                        full_model.x_input_alg: imp_batch}
        else:
            nat_dict = {full_model.x_input: x_batch,
                        full_model.y_input: y_batch}

        # Sanity check
        assert 0 <= np.amin(x_batch) and np.amax(x_batch) <= 1.0
        if USE_ALG:
            assert 0 <= np.amin(imp_batch) and np.amax(imp_batch) <= 1.0
            assert np.amax(np.abs(imp_batch-x_batch)) <= args.delta + 1e-6

        # Train
        start = timer()

        if args.no_lc:
            if USE_ALG:
                _, _, orig_acc_batch, safe_acc_batch, \
                x_safe, l2_dist_batch, l1_loss_batch, l2_loss_batch, lp_loss_batch = \
                    sess.run([train_step_g, extra_update_ops, orig_acc, safe_acc,
                              full_model.x_safe, l2_dist, l1_loss, l2_loss, lp_loss],
                             feed_dict=nat_dict)
            else:
                _, _, orig_acc_batch, safe_acc_batch, \
                x_safe, l2_dist_batch = \
                    sess.run([train_step_g, extra_update_ops, orig_acc, safe_acc,
                              full_model.x_safe, l2_dist],
                             feed_dict=nat_dict)
        else:
            if USE_ALG:
                _, _, safe_adv_loss_batch, safe_adv_acc_batch, orig_acc_batch, safe_acc_batch, \
                x_safe, x_safe_adv, l2_dist_batch, l1_loss_batch, l2_loss_batch, lp_loss_batch = \
                    sess.run([train_step_g, extra_update_ops, safe_adv_loss, safe_adv_acc, orig_acc, safe_acc,
                              full_model.x_safe, full_model.x_safe_adv, l2_dist, l1_loss, l2_loss, lp_loss],
                             feed_dict=nat_dict)
            else:
                _, _, safe_adv_loss_batch, safe_adv_acc_batch, orig_acc_batch, safe_acc_batch, \
                x_safe, x_safe_adv, l2_dist_batch = \
                    sess.run([train_step_g, extra_update_ops, safe_adv_loss, safe_adv_acc, orig_acc, safe_acc,
                              full_model.x_safe, full_model.x_safe_adv, l2_dist],
                             feed_dict=nat_dict)

        if args.use_advG:
            _, safe_adv_loss_batch, safe_adv_acc_batch, orig_acc_batch, safe_acc_batch, \
            x_safe, x_safe_adv, l2_dist_batch = \
                sess.run([train_step_advG, safe_adv_loss, safe_adv_acc, orig_acc, safe_acc,
                          full_model.x_safe, full_model.x_safe_adv, l2_dist],
                         feed_dict=nat_dict)

        if args.use_d:
            _, d_loss_batch, g_loss_batch, d_alg_out_batch, d_safe_out_batch = \
                sess.run([train_step_d, d_loss, g_loss,
                          full_model.d_alg_out, full_model.d_safe_out],
                         feed_dict=nat_dict)

        end = timer()

        assert 0 <= np.amin(x_safe) and np.amax(x_safe) <= 1.0
        if not args.no_lc:
            assert 0 <= np.amin(x_safe_adv) and np.amax(x_safe_adv) <= 1.0
        assert np.amax(np.abs(x_safe-x_batch)) <= args.delta + 1e-6

        training_time += end - start

        # Output to stdout
        if ii % num_output_steps == 0:
            print('Step {}:    ({})'.format(ii, datetime.now()))
            print('    orig accuracy {:.4}%'.format(orig_acc_batch * 100))
            print('    safe accuracy {:.4}%'.format(safe_acc_batch * 100))
            print('    l2 dist {:.4}'.format(l2_dist_batch))
            if not args.no_lc:
                print('    safe(adv) accuracy {:.4}%'.format(safe_adv_acc_batch * 100))
                print('    safe(adv) loss {:.6}'.format(safe_adv_loss_batch))
            if args.use_d:
                print('    d loss {:.6}'.format(d_loss_batch))
                print('    g loss {:.6}'.format(g_loss_batch))
                print('    d alg out {:.4}'.format(d_alg_out_batch.mean()))
                print('    d safe out {:.4}'.format(d_safe_out_batch.mean()))
                print('    d alg acc {:.4}%'.format((d_alg_out_batch>0.5).mean()*100))
                print('    d safe acc {:.4}%'.format((d_safe_out_batch<=0.5).mean()*100))
            if USE_ALG:
                print('    l1 loss {:.6}'.format(l1_loss_batch))
                print('    l2 loss {:.6}'.format(l2_loss_batch))
                print('    lp loss {:.6}'.format(lp_loss_batch))
            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * training_batch_size / training_time))
                training_time = 0.0

            #sys.exit()
        # Tensorboard summaries
        if ii % num_summary_steps == 0:
            safe_adv_acc_batch, safe_adv_loss_batch, \
                safe_pgd_acc_batch, safe_pgd_loss_batch, train_merged_summaries_batch \
                = sess.run([safe_adv_acc, safe_adv_loss,
                            safe_pgd_acc, safe_pgd_loss, train_merged_summaries], feed_dict=nat_dict)
            print('    safe(adv) accuracy {:.4}%'.format(safe_adv_acc_batch * 100))
            print('    safe(pgd) accuracy {:.4}%'.format(safe_pgd_acc_batch * 100))
            print('    safe(adv) loss {:.6}'.format(safe_adv_loss_batch))
            print('    safe(pgd) loss {:.6}'.format(safe_pgd_loss_batch))
            if args.save:
                summary_writer.add_summary(train_merged_summaries_batch, global_step.eval(sess))
            # evaluate on test set
            #eval_bstart = (ii//num_summary_steps)*eval_batch_size
            #eval_bend = (ii//num_summary_steps+1)*eval_batch_size
            
            eval_x_batch, eval_y_batch, _ = mnist_test.get_next_batch(eval_batch_size, multiple_passes=True,
                                                                   get_indices=True)
            eval_dict = {full_model.x_input: eval_x_batch,
                         full_model.y_input: eval_y_batch}
            orig_acc_batch, safe_acc_batch, safe_adv_acc_batch, safe_adv_loss_batch, \
                safe_pgd_acc_batch, safe_pgd_loss_batch, \
                x_safe, l2_dist_batch, eval_merged_summaries_batch = \
                sess.run([orig_acc, safe_acc, safe_adv_acc, safe_adv_loss, safe_pgd_acc, safe_pgd_loss,
                          full_model.x_safe, l2_dist, eval_merged_summaries], feed_dict=eval_dict)

            print('    orig accuracy (eval) {:.4}%'.format(orig_acc_batch * 100))
            print('    safe accuracy (eval) {:.4}%'.format(safe_acc_batch * 100))
            print('    l2 dist (eval) {:.4}'.format(l2_dist_batch))
            print('    safe(adv) accuracy (eval) {:.4}%'.format(safe_adv_acc_batch * 100))
            print('    safe(pgd) accuracy (eval) {:.4}%'.format(safe_pgd_acc_batch * 100))
            print('    safe(adv) loss (eval) {:.6}'.format(safe_adv_loss_batch))
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