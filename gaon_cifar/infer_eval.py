"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import sys
import math
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

import cifar10_input

from infer_model import Model as Safe_model
from infer_target import Model as Target_model

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

    # pgd settings
    parser.add_argument('--eps', default=8.0, type=float)
    parser.add_argument('--num_steps', default=20, type=int)
    parser.add_argument('--step_size', default=2.0, type=float)
    parser.add_argument('--restarts', default=20, type=int)

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
full_model = Safe_model('eval', model, args)

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

model_dir = MODEL_PATH + args.save_dir + meta_name
if not os.path.exists(model_dir):
    print("incorrect path!")
    sys.exit()

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.saver()
cifar = cifar10_input.CIFAR10Data(data_path)

model_file = tf.train.latest_checkpoint(model_dir)
if model_file is None:
    print('No model found')
    sys.exit()

with tf.Session() as sess:

    saver.restore(sess, model_file)
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

        if args.val_step_per > 0:
            eval_batch_size = 100
        else:
            eval_batch_size = min(100, num_eval_examples)

        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        print('Iterating over {} batches'.format(num_batches))

        x_full_batch = x_full_batch.astype(np.float32)

        x_imp = []  # imp accumulator
        step_imp = []  # num steps accumulator
        l2_li = [] # l2 distance accumulator

        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}'.format(bend - bstart))

            x_batch = x_full_batch[bstart:bend, :]
            y_batch = y_full_batch[bstart:bend]

            mask = np.array([True for _ in range(len(y_batch))])

            for _ in range(args.restarts):
                
                cur_corr = sess.sun(


            # run our algorithm
            print('fortifying image ', bstart)
            x_batch_imp, step_batch_imp = impenet.fortify(x_batch, y_batch, sess, y_batch_gt)
            print()

            # evaluation
            # y_batch_hard = np.argmax(y_batch, axis=1)
            # y_batch_hard = np.eye(10)[y_batch_hard.reshape(-1)]

            # suc_flag = impenet.validation(x_batch_imp, y_batch, y_batch_hard)

            x_imp.append(x_batch_imp)
            step_imp.append(np.array([step_batch_imp]))
            l2_li.append(np.linalg.norm((x_batch_imp - x_batch)/255))
            print('l2 distance (curr):', np.linalg.norm(x_batch_imp - x_batch)/255)
            print('l2 distance (total):', np.mean(l2_li))
            print()
            print('------------------------------------------------------------------------------')
            print('------------------------------------------------------------------------------')

        x_imp = np.concatenate(x_imp)
        step_imp = np.concatenate(step_imp)

        # save image
        folder_name = './arr' + '_main' + '/'
        batch_name = '_' + str(args.bstart) + '_' + str(args.sample_size)
        common_name = folder_name + meta_name + batch_name
        
        np.save(common_name + '_x_org', x_full_batch)
        np.save(common_name + '_x_imp', x_imp)
        np.save(common_name + '_step_imp', step_imp)
        np.save(common_name + '_y', y_full_batch)

        # sanity check
        if np.amax(x_imp) > 255.0001 or \
            np.amin(x_imp) < -0.0001 or \
            np.isnan(np.amax(x_imp)):
            print('Invalid pixel range in x_imp. Expected [0,255], fount[{},{}]'.format(np.amin(x_imp),
                                                                                        np.amax(x_imp)))
        else:
            result(x_imp, model, sess, x_full_batch, y_full_batch)



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
            if not args.no_save:
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
            
            if not args.no_save:
                summary_writer.add_summary(eval_merged_summaries_batch, global_step.eval(sess))


        # Write a checkpoint
        if ii % num_checkpoint_steps == 0 and not args.no_save:
            saver.save(sess,
                       os.path.join(save_dir, 'checkpoint'),
                       global_step=global_step)
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
            if not args.no_save:
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
            
            if not args.no_save:
                summary_writer.add_summary(eval_merged_summaries_batch, global_step.eval(sess))


        # Write a checkpoint
        if ii % num_checkpoint_steps == 0 and not args.no_save:
            saver.save(sess,
                       os.path.join(save_dir, 'checkpoint'),
                       global_step=global_step)
