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

from tensorflow.examples.tutorials.mnist import input_data

from generator import generator_v2 as Generator
from discriminator import Discriminator
from model import Model as Model

from utils import infer_file_name, load_imp_data

import argparse

from pgd_attack import LinfPGDAttack

MODEL_PATH = './models/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # save & load path
    parser.add_argument('--model_dir', default='naturally_trained', type=str)
    parser.add_argument('--save_dir', default='safe_net', type=str, help='safe_net saved folder')
    parser.add_argument('--corr_only', action='store_true')
    parser.add_argument('--fail_only', action='store_true')
    parser.add_argument('--eval', action='store_true', help='use test set data. else use train set data.')
    parser.add_argument('--force_path', default='', type=str, help='if you want to manually select the folder')

    # eval parameters
    parser.add_argument('--tf_random_seed', default=451760341, type=int)
    parser.add_argument('--np_random_seed', default=216105420, type=int)
    parser.add_argument('--g_lr', default=1e-3, type=float)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--eval_on_cpu', action='store_true')
    parser.add_argument('--delta', default=0.3, type=float)
    parser.add_argument('--sample_size', default=1000, type=int)
    parser.add_argument('--bstart', default=0, type=int)
    parser.add_argument('--train_mode', action='store_true', help='use train mode neural nets')
    parser.add_argument('--comp_imp', action='store_true', help='compare distance with S_maml results')
    parser.add_argument('--eval_imp', action='store_true', help='also evaluate S_maml results')

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

    # pgd (filename) settings
    parser.add_argument('--eps', default=0.3, type=float)
    parser.add_argument('--num_steps', default=40, type=int)
    parser.add_argument('--step_size', default=0.01, type=float)

    # pgd (eval) settings
    parser.add_argument('--val_eps', default=0.3, type=float)
    parser.add_argument('--val_num_steps', default=100, type=int)
    parser.add_argument('--val_step_size', default=0.01, type=float)
    parser.add_argument('--val_restarts', default=20, type=int)
    parser.add_argument('--val_rand', action='store_true', help='PGD random start. Set to True if val_restarts > 1')

    args = parser.parse_args()

    for key, val in vars(args).items():
        print('{}={}'.format(key, val))

# seeding randomness
tf.set_random_seed(args.tf_random_seed)
np.random.seed(args.np_random_seed)

# Setting up training parameters
eval_batch_size = args.eval_batch_size

# Setting up the data and the model
global_step = tf.train.get_or_create_global_step()

model = Model()

x_input = tf.placeholder(
    tf.float32,
    shape=[None, 28, 28, 1]
)
y_input = tf.placeholder(
    tf.int64,
    shape=None
)

generator = tf.make_template('generator', Generator, f_dim=args.f_dim, c_dim=3,
                             is_training=args.train_mode)

if args.use_d:
    discriminator = Discriminator()
    d_out = discriminator(x_input)

noise = generator(x_input)
x_safe = x_input + args.delta * noise
x_safe_clipped = tf.clip_by_value(x_safe, 0, 1)

pgd = LinfPGDAttack(model,
                    args.val_eps,
                    args.val_num_steps,
                    args.val_step_size,
                    random_start=(True if args.val_restarts > 1 else args.val_rand),
                    loss_func='xent')

# Setting up the Tensorboard and checkpoint outputs
if args.force_path:
    meta_name = args.force_path
else:
    meta_name = infer_file_name(args)
print(meta_name)

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
if args.eval:
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False, validation_size=0, reshape=False).test
else:
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False, validation_size=0, reshape=False).train

if args.comp_imp or args.eval_imp:
    imp_mnist = load_imp_data(args, args.eval)

model_file = tf.train.latest_checkpoint(model_dir)
if model_file is None:
    print('No model found')
    sys.exit()

with tf.Session() as sess:

    # Restore variables if can, set optimizer
    reader = tf.train.NewCheckpointReader(model_file)
    saved_shapes = reader.get_variable_to_shape_map()
    
    '''
    tvar = tf.trainable_variables()
    for var in tvar:
        print(var.name)
    sys.exit()
    '''

    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    restore_vars_name_list = []
    for var_name, saved_var_name in var_names:
        curr_var = tf.get_default_graph().get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
            restore_vars_name_list.append(saved_var_name + ':0')

    trainable_variables = tf.trainable_variables()
    variables_to_train = [var for var in trainable_variables if var.name not in restore_vars_name_list]

    sess.run(tf.global_variables_initializer())
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(sess, model_file)
    print('restore success!')

    num_eval_examples = args.sample_size

    if args.corr_only:
        if args.model_dir == 'naturally_trained':
            indices = np.load('/data/home/gaon/lazy-attack/mnist/mnist_data/nat_success_indices.npy')
        else:
            indices = np.load('/data/home/gaon/lazy-attack/mnist/mnist_data/adv_success_indices.npy')
    elif args.fail_only:
        if args.model_dir == 'naturally_trained':
            indices = np.load('/data/home/gaon/lazy-attack/mnist/mnist_data/nat_fail_indices.npy')
        else:
            indices = np.load('/data/home/gaon/lazy-attack/mnist/mnist_data/adv_fail_indices.npy')
    else:
        indices = [i for i in range(args.sample_size + args.bstart)]

    # load data
    bstart = args.bstart
    while True:
        x_candid = mnist.images[indices[bstart:bstart + 100]]
        y_candid = mnist.labels[indices[bstart:bstart + 100]]

        if args.comp_imp or args.eval_imp:
            imp_candid = imp_mnist[indices[bstart:bstart + 100]]

            if np.amax(np.abs(x_candid-imp_candid)) > args.delta + 1e-3:
                raise Exception

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
            if args.comp_imp or args.eval_imp:
                imp_full_batch = imp_candid[:min(num_eval_examples, len(imp_candid))]
        else:
            index = min(num_eval_examples - len(x_full_batch), len(x_candid))
            x_full_batch = np.concatenate((x_full_batch, x_candid[:index]))
            y_full_batch = np.concatenate((y_full_batch, y_candid[:index]))
            logit_full_batch = np.concatenate((logit_full_batch, logits[:index]))
            if args.comp_imp or args.eval_imp:
                imp_full_batch = np.concatenate((imp_full_batch, imp_candid[:index]))
        bstart += 100
        if (len(x_full_batch) >= num_eval_examples) or bstart >= len(indices):
            break

    # Adjust num_eval_examples. Iterate over the samples batch-by-batch
    num_eval_examples = len(x_full_batch)

    if num_eval_examples > args.eval_batch_size:
        eval_batch_size = args.eval_batch_size
    else:
        eval_batch_size = min(args.eval_batch_size, num_eval_examples)

    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    print('Iterating over {} batches'.format(num_batches))

    x_full_batch = x_full_batch.astype(np.float32)

    full_mask = []
    orig_correct_num = 0
    safe_correct_num = 0
    l2_dist = []

    if args.comp_imp:
        l1_loss = []
        l2_loss = []
    if args.eval_imp:
        imp_correct_num = 0
        imp_full_mask = []

    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)
        print('batch {}-{}'.format(bstart, bend-1))

        x_batch = x_full_batch[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]

        if args.comp_imp or args.eval_imp:
            imp_batch = imp_full_batch[bstart: bend, :]

        correct_prediction = sess.run(model.correct_prediction,
                                       feed_dict={model.x_input: x_batch,
                                                  model.y_input: y_batch})

        print('original acc: {}'.format(np.sum(correct_prediction)))
        orig_correct_num += np.sum(correct_prediction)
        
        [x_batch_safe, noise_batch] = sess.run([x_safe_clipped, noise],
                                feed_dict={x_input: x_batch})
        
        correct_prediction = sess.run(model.correct_prediction,
                                       feed_dict={model.x_input: x_batch_safe,
                                                  model.y_input: y_batch})

        print('safe acc: {}'.format(np.sum(correct_prediction)))
        safe_correct_num += np.sum(correct_prediction)

        if args.eval_imp:
            correct_prediction = sess.run(model.correct_prediction,
                                          feed_dict={model.x_input: imp_batch,
                                                     model.y_input: y_batch})

            print('imp acc: {}'.format(np.sum(correct_prediction)))
            imp_correct_num += np.sum(correct_prediction)

        if args.use_d:
            disc_out = sess.run(d_out,
                                feed_dict={x_input: x_batch_safe})
            #print('full disc:')
            #print(np.mean(disc_out.reshape(disc_out.shape[0], -1), axis=1))
            print('disc value: {}'.format(np.mean(disc_out)))

        assert np.amin(x_batch_safe) >= (0-1e-3) and np.amax(x_batch_safe) <= (1.0+1e-3)
        assert np.amax(np.abs(x_batch_safe-x_batch)) <= args.delta+1e-3

        l2_dist_batch = np.mean(np.linalg.norm((x_batch_safe-x_batch).reshape(x_batch.shape[0], -1), axis=1))
        print('l2 dist: {:.4f}'.format(l2_dist_batch))

        mask = np.array([True for _ in range(len(y_batch))])

        if args.comp_imp:
            raw_dist = (imp_batch - x_batch_safe)/255
            l1_loss_batch = np.mean(np.abs(raw_dist))
            l2_loss_batch = np.mean(np.linalg.norm(raw_dist.reshape(raw_dist.shape[0], -1), axis=1))
            print('l1 loss: {:.5f}'.format(l1_loss_batch))
            print('l2 loss: {:.5f}'.format(l2_loss_batch))
            
            l1_loss.append(l1_loss_batch)
            l2_loss.append(l2_loss_batch)

        for _ in range(args.val_restarts):
            
            x_batch_attacked, _ = pgd.perturb(x_batch_safe, y_batch, sess)

            correct_prediction = sess.run(model.correct_prediction,
                                           feed_dict={model.x_input: x_batch_attacked,
                                                      model.y_input: y_batch})

            mask *= correct_prediction

            if np.sum(mask)==0:
                break

        l2_dist.append(l2_dist_batch)
        full_mask.append(mask)

        print('safe(PGD) acc: {}'.format(np.sum(mask)/np.size(mask)))

        if args.eval_imp:

            mask = np.array([True for _ in range(len(y_batch))])
            for _ in range(args.val_restarts):

                imp_batch_attacked, _ = pgd.perturb(imp_batch.astype(np.float32), y_batch, sess)

                correct_prediction = sess.run(model.correct_prediction,
                                              feed_dict={model.x_input: imp_batch_attacked,
                                                         model.y_input: y_batch})

                mask *= correct_prediction

                if np.sum(mask)==0:
                    break

            imp_full_mask.append(mask)

            print('imp(PGD) acc: {}'.format(np.sum(mask)/np.size(mask)))

        print()

    full_mask = np.concatenate(full_mask)

    if args.eval_imp:
        imp_full_mask = np.concatenate(imp_full_mask)

    print("orig accuracy: {:.2f}".format(orig_correct_num/np.size(full_mask)*100))
    print("safe accuracy: {:.2f}".format(safe_correct_num/np.size(full_mask)*100))
    if args.eval_imp:
        print("imp accuracy: {:.2f}".format(imp_correct_num/np.size(full_mask)*100))
    print("safe(PGD) accuracy: {:.2f}".format(np.mean(full_mask)*100))
    if args.eval_imp:
        print("imp(PGD) accuracy: {:.2f}".format(np.mean(imp_full_mask)*100))
    print("l2 dist: {:.4f}".format(np.mean(l2_dist)))
    if args.comp_imp:
        print("l1 loss: {:.5f}".format(np.mean(l1_loss)))
        print("l2 loss: {:.5f}".format(np.mean(l2_loss)))

