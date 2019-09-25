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

#from cleverhans.model_zoo.madry_lab_challenges.cifar10_model import make_wresnet
import cifar10_input

from infer_model import generator, PGD
from infer_target import Model as Target_model

import argparse

MODEL_PATH = './models/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # save & load path
    parser.add_argument('--data_path', default='../cifar10_data', type=str)
    parser.add_argument('--model_dir', default='naturally_trained', type=str)
    parser.add_argument('--save_dir', default='safe_net', type=str)

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

# Sanity checker
attack_params = {
            'eps': args.eps,
            'step_size': args.step_size,
            'num_steps': args.num_steps,
            'random_start': args.random_start,
            'bounds': (0, 255),
        }

x_input = tf.placeholder(
    tf.float32,
    shape=[None, 32, 32, 3])
y_input = tf.placeholder(tf.int64, shape=None)

generator = tf.make_template('generator', generator, f_dim=64, output_size=32, c_dim=3, is_training=True)

x_safe = generator(x_input) * args.delta + x_input
x_safe = tf.clip_by_value(x_safe, 0, 255)

pgd_x_input = tf.placeholder(
    tf.float32,
    shape=[None, 32, 32, 3])
pgd_y_input = tf.placeholder(tf.int64, shape=None)
x_safe_pgd = PGD(pgd_x_input, pgd_y_input, model.fprop, attack_params)
diff = x_safe_pgd - x_safe
x_safe_pgd_fo = x_safe + tf.stop_gradient(diff)

eval_x_input = tf.placeholder(
    tf.float32,
    shape=[None, 32, 32, 3])
eval_y_input = tf.placeholder(
    tf.int64,
    shape=None
)

pre_softmax = model.fprop(eval_x_input)
predictions = tf.argmax(pre_softmax, 1)
correct_prediction = tf.equal(predictions, eval_y_input)
accuracy = tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32)
)
y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=pre_softmax, labels=eval_y_input
)
mean_xent = tf.reduce_mean(y_xent)

# Setting up the optimizer
boundaries = [0, 40000, 60000]
values = [lr, lr/10, lr/100]
boundaries = boundaries[1:]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
#total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
#total_loss = full_model.safe_pgd_mean_xent + full_model.safe_mean_xent

# Setting up the Tensorboard and checkpoint outputs
model_dir = MODEL_PATH + args.model_dir
save_dir = MODEL_PATH + args.save_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver()
#tf.summary.scalar('accuracy adv train', model.accuracy)
#tf.summary.scalar('accuracy adv', model.accuracy)
#tf.summary.scalar('xent adv train', model.xent / batch_size)
#tf.summary.scalar('xent adv', model.xent / batch_size)
#tf.summary.image('images adv train', model.x_input)
#merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

model_file = tf.train.latest_checkpoint(model_dir)
if model_file is None:
    print('No model found')
    sys.exit()

with tf.Session() as sess:

    # initialize data augmentation
    cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

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
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
            restore_vars_name_list.append(saved_var_name + ':0')
    #print(restore_vars_name_list)
    #print(len(restore_vars))
    
    trainable_variables = tf.trainable_variables()
    #print(trainable_variables)
    #print(len(trainable_variables))
    variables_to_train = [var for var in trainable_variables if (var.name) not in restore_vars_name_list]
    #print(variables_to_train)
    #print(len(variables_to_train))
    '''
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        total_loss,
        global_step=global_step,
        var_list=variables_to_train)
    '''
    
    sess.run(tf.global_variables_initializer())
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(sess, model_file)
    print('restore success!')
    
    gen_training_time = 0.0
    pgd_training_time = 0.0

    # Main training loop
    for ii in range(max_num_training_steps):
        x_batch, y_batch = cifar.train_data.get_next_batch(training_batch_size,
                                                           multiple_passes=True)

        # Actual training step
        nat_dict = {x_input: x_batch,
                    y_input: y_batch}

        assert 0 <= np.amin(x_batch) and np.amax(x_batch) <= 255.0

        start = timer()
        x_safe_batch = \
            sess.run(x_safe, feed_dict=nat_dict)
        end = timer()

        gen_training_time += end-start

        safe_dict = {pgd_x_input: x_safe_batch,
                     pgd_y_input: y_input}

        start = timer()
        x_safe_pgd_batch = \
            sess.run(x_safe_pgd, feed_dict=safe_dict)
        end = timer()
        assert 0 <= np.amin(x_safe_batch) and np.amax(x_safe_batch) <= 255.0
        assert 0 <= np.amin(x_safe_pgd_batch) and np.amax(x_safe_pgd_batch) <= 255.0

        l2_dist = np.mean(np.linalg.norm((x_batch-x_safe_batch).reshape(training_batch_size, -1)/255, axis=1))

        pgd_training_time += end - start

        # Output to stdout
        if ii % num_output_steps == 0:
            print('Step {}:    ({})'.format(ii, datetime.now()))
            #print('    orig accuracy {:.4}%'.format(orig_acc_batch * 100))
            #print('    safe accuracy {:.4}%'.format(safe_acc_batch * 100))
            #print('    safe_pgd accuracy {:.4}%'.format(safe_pgd_acc_batch * 100))
            #print('    l2 dist {:.4}'.format(l2_dist))
            #print('    total loss {:.6}'.format(loss))
            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * training_batch_size / training_time))
                training_time = 0.0

        # Tensorboard summaries
        '''
        if ii % num_summary_steps == 0:
            summary = sess.run(merged_summaries, feed_dict=nat_dict)
            summary_writer.add_summary(summary, global_step.eval(sess))
        '''

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
            saver.save(sess,
                       os.path.join(save_dir, 'checkpoint'),
                       global_step=global_step)
