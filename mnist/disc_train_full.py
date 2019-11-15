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

from disc_model import ModelMultiClass as Safe_model
from infer_target import Model as Target_model

from utils import disc_file_name, load_imp_data, CustomDataSet

import argparse

MODEL_PATH = './models/'
NUM_CLASSES = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # save & load path
    parser.add_argument('--model_dir', default='adv_trained', type=str)
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
    parser.add_argument('--training_batch_size', default=50, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--eval_on_cpu', action='store_true')
    parser.add_argument('--delta', default=0.3, type=float)

    # GAN settings
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--f_dim', default=64, type=int)
    parser.add_argument('--d_lr', default=1e-3, type=float)
    parser.add_argument('--multi_pass', action='store_true')
    parser.add_argument('--patch', action='store_true')

    # pgd settings
    parser.add_argument('--eps', default=0.3, type=float)
    parser.add_argument('--num_steps', default=100, type=int)
    parser.add_argument('--step_size', default=0.01, type=float)
    parser.add_argument('--random_start', action='store_true')

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
d_lr = args.d_lr
training_batch_size = args.training_batch_size
eval_batch_size = args.eval_batch_size


# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False, validation_size=0, reshape=False)

mnist_train = CustomDataSet(mnist.train.images, mnist.train.labels)
mnist_test = CustomDataSet(mnist.test.images, mnist.test.labels)

imp_mnist_train_li = [load_imp_data(args, eval_flag=False, target=i) for i in range(NUM_CLASSES)]
imp_mnist_eval_li = [load_imp_data(args, eval_flag=True, target=i) for i in range(NUM_CLASSES)]

imp_mnist_train_gt = load_imp_data(args, eval_flag=False, target=-1)
imp_mnist_eval_gt = load_imp_data(args, eval_flag=True, target=-1)

global_step = tf.train.get_or_create_global_step()


model = Target_model()
full_model = Safe_model('train', model, args)

# set up metrics
total_loss = full_model.xent
accuracy_train = full_model.accuracy

orig_model_acc = full_model.orig_accuracy



# Setting up the Tensorboard and checkpoint outputs
meta_name = disc_file_name(args, multi_class=True)

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
    tf.summary.scalar('acc (train)', accuracy_train),
    tf.summary.scalar('total loss', total_loss),
]

train_merged_summaries = tf.summary.merge(train_summaries)

eval_summaries = [
    tf.summary.scalar('eval acc (train)', accuracy_train),
    tf.summary.scalar('eval total loss', total_loss),
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

    variables_to_train_d = [var for var in trainable_variables if (var.name not in restore_vars_name_list and
                                                                   'discriminator' in var.name)]

    print(np.sum([np.prod(v.get_shape().as_list()) for v in restore_vars]))
    print(np.sum([np.prod(v.get_shape().as_list()) for v in variables_to_train]))
    print(np.sum([np.prod(v.get_shape().as_list()) for v in variables_to_train_d]))

    train_step_d = tf.train.AdamOptimizer(args.d_lr).minimize(
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

        # Get data
        x_batch, y_batch, indices = mnist_train.get_next_batch(training_batch_size,
                                                               multiple_passes=True,
                                                               get_indices=True)

        imp_batch_li = [imp_mnist[indices, ...] for imp_mnist in imp_mnist_train_li]

        # Sanity check
        #assert np.amax(np.abs((imp_batch_gt - x_input_alg_fake_batch) * mask_batch.reshape(-1, 1, 1, 1))) <= 1e-6
        #assert np.amax(np.abs((imp_batch_gt - x_input_alg_fake_batch) * (1-mask_batch.reshape(-1, 1, 1, 1)))) > 1e-6
        assert 0 <= np.amin(x_batch) and np.amax(x_batch) <= 1.0
        for imp_batch in imp_batch_li:
            assert 0 <= np.amin(imp_batch) and np.amax(imp_batch) <= 1.0
            assert np.amax(np.abs(imp_batch-x_batch)) <= args.delta + 1e-6

        imp_batch_full = np.concatenate(imp_batch_li, axis=-1)

        nat_dict = {full_model.x_input: x_batch,
                    full_model.x_input_alg: imp_batch_full,
                    full_model.y_input: y_batch}

        # Train
        start = timer()

        _, _, accuracy_train_batch, \
            total_loss_batch, \
            train_merged_summaries_batch, orig_model_acc_batch = \
            sess.run([train_step_d, extra_update_ops, accuracy_train,
                      total_loss,
                      train_merged_summaries, orig_model_acc],
                     feed_dict=nat_dict)

        end = timer()
        training_time += end - start

        # Output to stdout
        if ii % num_output_steps == 0:
            print('Step {}:    ({})'.format(ii, datetime.now()))
            #print('    acc orig {:.4}%'.format(orig_model_acc_batch * 100))
            print('    acc train {:.4}%'.format(accuracy_train_batch * 100))
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

            eval_x_batch, eval_y_batch, indices = mnist_test.get_next_batch(eval_batch_size, multiple_passes=True,
                                                                            get_indices=True)
            imp_batch_li = [imp_mnist[indices, ...] for imp_mnist in imp_mnist_eval_li]

            imp_batch_full = np.concatenate(imp_batch_li, axis=-1)
            eval_dict = {full_model.x_input: eval_x_batch,
                         full_model.x_input_alg: imp_batch_full,
                         full_model.y_input: eval_y_batch}

            # Sanity check
            #assert np.amax(np.abs((imp_batch_gt - x_input_alg_fake_batch) * mask_batch)) <= 1e-6
            #assert np.amax(np.abs((imp_batch_gt - x_input_alg_fake_batch) * (1-mask_batch))) > 1e-6
            assert 0 <= np.amin(eval_x_batch) and np.amax(eval_x_batch) <= 1.0
            for imp_batch in imp_batch_li:
                assert 0 <= np.amin(imp_batch) and np.amax(imp_batch) <= 1.0
                assert np.amax(np.abs(imp_batch-eval_x_batch)) <= args.delta + 1e-6

            _, _, accuracy_train_batch, \
                total_loss_batch, \
                eval_merged_summaries_batch, orig_model_acc_batch = \
                sess.run([train_step_d, extra_update_ops, accuracy_train,
                          total_loss,
                          eval_merged_summaries, orig_model_acc],
                         feed_dict=eval_dict)

            accuracy_infer_batch = np.mean(full_model.infer(sess, eval_x_batch, imp_batch_li) == eval_y_batch)

            #print('    acc orig (eval) {:.4}%'.format(orig_model_acc_batch * 100))
            print('    acc train (eval) {:.4}%'.format(accuracy_train_batch * 100))
            print('    total loss (eval) {:.6}'.format(total_loss_batch))

            if args.save:
                summary_writer.add_summary(eval_merged_summaries_batch, global_step.eval(sess))

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0 and args.save:
            saver.save(sess,
                       os.path.join(save_dir, 'checkpoint'),
                       global_step=global_step)
