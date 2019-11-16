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

from disc_model import Model as Safe_model
from infer_target import Model as Model

from utils import disc_file_name, load_imp_data

import argparse

MODEL_PATH = './models/'
NUM_CLASSES = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # save & load path
    parser.add_argument('--model_dir', default='adv_trained', type=str)
    parser.add_argument('--save_dir', default='safe_net_2', type=str, help='safe_net saved folder')
    parser.add_argument('--corr_only', action='store_true')
    parser.add_argument('--fail_only', action='store_true')
    parser.add_argument('--eval', action='store_true', help='use test set data. else use train set data.')
    parser.add_argument('--force_path', default='', type=str, help='if you want to manually select the folder')
    parser.add_argument('--multi_class', action='store_true')

    # eval parameters
    parser.add_argument('--tf_random_seed', default=451760341, type=int)
    parser.add_argument('--np_random_seed', default=216105420, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--eval_on_cpu', action='store_true')
    parser.add_argument('--delta', default=0.3, type=float)
    parser.add_argument('--sample_size', default=10000, type=int)
    parser.add_argument('--bstart', default=0, type=int)
    parser.add_argument('--train_mode', action='store_true', help='use train mode neural nets')

    # GAN settings
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--f_dim', default=64, type=int)
    parser.add_argument('--d_lr', default=1e-3, type=float)
    parser.add_argument('--c_loss', action='store_true')
    parser.add_argument('--patch', action='store_true')
    parser.add_argument('--logits', action='store_true')

    # pgd (filename) settings
    parser.add_argument('--eps', default=0.3, type=float)
    parser.add_argument('--num_steps', default=100, type=int)
    parser.add_argument('--step_size', default=0.01, type=float)
    parser.add_argument('--random_start', action='store_true')

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
full_model = Safe_model('train' if args.train_mode else 'eval', model, args)

# set up metrics
if args.c_loss:
    total_loss = full_model.c_loss
    accuracy_train = full_model.c_accuracy
    accuracy_train_real = full_model.c_accuracy_real
    accuracy_train_fake = full_model.c_accuracy_fake
else:
    total_loss = full_model.d_loss
    accuracy_train = full_model.d_accuracy
    accuracy_train_real = full_model.d_accuracy_real
    accuracy_train_fake = full_model.d_accuracy_fake

orig_model_acc = full_model.orig_accuracy

# Setting up the Tensorboard and checkpoint outputs
if args.force_path:
    meta_name = args.force_path
else:
    meta_name = disc_file_name(args, multi_class=args.multi_class)
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
    imp_mnist_li = [load_imp_data(args, eval_flag=True, target=i) for i in range(NUM_CLASSES)]
else:
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False, validation_size=0, reshape=False).train
    imp_mnist_li = [load_imp_data(args, eval_flag=False, target=i) for i in range(NUM_CLASSES)]

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

        imp_candid_li = [imp_mnist[indices[bstart:bstart + 100]] for imp_mnist in imp_mnist_li]

        for imp_candid in imp_candid_li:
            assert 0 <= np.amin(imp_candid) and np.amax(imp_candid) <= 1.0
            assert np.amax(np.abs(imp_candid-x_candid)) <= args.delta + 1e-6

        mask = sess.run(full_model.orig_correct_prediction,
                        feed_dict={full_model.x_input: x_candid,
                                   full_model.y_input: y_candid})
        print(sum(mask))
        if args.corr_only and (np.mean(mask) < 1.0 - 1E-6):
            raise Exception
        if args.fail_only and (np.mean(mask) > 0.0 + 1E-6):
            raise Exception
        if bstart == args.bstart:
            x_full_batch = x_candid[:min(num_eval_examples, len(x_candid))]
            y_full_batch = y_candid[:min(num_eval_examples, len(y_candid))]
            imp_full_batch_li = [imp_candid[:min(num_eval_examples, len(imp_candid))] for imp_candid in imp_candid_li]
        else:
            index = min(num_eval_examples - len(x_full_batch), len(x_candid))
            x_full_batch = np.concatenate((x_full_batch, x_candid[:index]))
            y_full_batch = np.concatenate((y_full_batch, y_candid[:index]))
            imp_full_batch_li = [np.concatenate((imp_full_batch, imp_candid[:index]))
                                 for imp_full_batch, imp_candid in zip(imp_full_batch_li, imp_candid_li)]
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
    safe_full_mask = []
    orig_correct_num = 0
    safe_correct_num = 0
    acc_infer = []
    acc_train = []
    acc_train_real = []
    acc_train_fake = []
    l2_dist = []

    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)
        print('batch {}-{}'.format(bstart, bend-1))

        x_batch = x_full_batch[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]

        imp_batch_li = [imp_full_batch[bstart: bend, :] for imp_full_batch in imp_full_batch_li]

        correct_prediction = sess.run(full_model.orig_correct_prediction,
                                      feed_dict={full_model.x_input: x_batch,
                                                 full_model.y_input: y_batch})

        print('original acc: {}'.format(np.sum(correct_prediction)))
        orig_correct_num += np.sum(correct_prediction)

        mask = np.array([True for _ in range(len(y_batch))])
        for _ in range(args.val_restarts):

            correct_prediction = sess.run(full_model.orig_pgd_correct_prediction,
                                          feed_dict={full_model.x_input: x_batch,
                                                     full_model.y_input: y_batch})

            mask *= correct_prediction

            if np.sum(mask) == 0:
                break

        full_mask.append(mask)

        print('orig(PGD) acc: {:.2f}'.format(np.sum(mask)/np.size(mask)*100))

        # eval detections
        y_fake_batch, mask_batch, x_input_alg_fake_batch = full_model.generate_fakes(y_batch, imp_batch_li)

        nat_dict = {full_model.x_input: x_batch,
                    full_model.x_input_alg: x_input_alg_fake_batch,
                    full_model.y_input: y_batch,
                    full_model.mask_input: mask_batch}

        accuracy_batch, accuracy_real_batch, accuracy_fake_batch, \
            total_loss_batch, orig_model_acc_batch = \
            sess.run([accuracy_train, accuracy_train_real, accuracy_train_fake, total_loss,
                      orig_model_acc],
                     feed_dict=nat_dict)

        print("train acc: {:.2f}".format(accuracy_batch*100))
        print("train acc - real: {:.2f}".format(accuracy_real_batch*100))
        print("train acc - fake: {:.2f}".format(accuracy_fake_batch*100))

        acc_train.append(accuracy_batch)
        acc_train_real.append(accuracy_real_batch)
        acc_train_fake.append(accuracy_fake_batch)

        y_pred, x_batch_safe = full_model.infer(sess, x_batch, imp_batch_li, return_images=True)
        accuracy_infer_batch = np.mean(y_pred == y_batch)
        print('infer acc: {:.2f}'.format(accuracy_infer_batch*100))

        acc_infer.append(accuracy_infer_batch)
        
        safe_dict = {full_model.x_input: x_batch,
                    full_model.x_input_alg: x_batch_safe,
                    full_model.y_input: y_batch}

        correct_prediction = sess.run(full_model.alg_correct_prediction,
                                      feed_dict=safe_dict)

        print('safe acc: {}'.format(np.sum(correct_prediction)))
        safe_correct_num += np.sum(correct_prediction)

        mask = np.array([True for _ in range(len(y_batch))])
        for _ in range(args.val_restarts):

            correct_prediction = sess.run(full_model.alg_pgd_correct_prediction,
                                          feed_dict=safe_dict)

            mask *= correct_prediction

            if np.sum(mask) == 0:
                break

        safe_full_mask.append(mask)

        print('safe(PGD) acc: {:.2f}'.format(np.sum(mask)/np.size(mask)*100))

        print()

    full_mask = np.concatenate(full_mask)
    safe_full_mask = np.concatenate(safe_full_mask)

    print("orig accuracy: {:.2f}".format(orig_correct_num/np.size(full_mask)*100))
    print("orig(PGD) accuracy: {:.2f}".format(np.mean(full_mask)*100))
    print("train accuracy: {:.2f}".format(np.mean(acc_train)*100))
    print("train accuracy - real: {:.2f}".format(np.mean(acc_train_real)*100))
    print("train accuracy - fake: {:.2f}".format(np.mean(acc_train_fake)*100))
    print("infer accuracy: {:.2f}".format(np.mean(acc_infer)*100))
    print("safe accuracy: {:.2f}".format(safe_correct_num/np.size(safe_full_mask)*100))
    print("safe(PGD) accuracy: {:.2f}".format(np.mean(safe_full_mask)*100))
