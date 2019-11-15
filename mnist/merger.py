"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np
import sys

from tensorflow.examples.tutorials.mnist import input_data
from utils import load_imp_data

from pgd_attack import LinfPGDAttack

FORCE_DIR = './arr_main/'
SAVE_DIR = './arr_save/'

def merge(args):

    if args.force_path:
        x_org = []
        x_imp = []
        mask = []
        y = []

        batch_size = args.batch_size
        sample_size = args.sample_size
        num_batches = sample_size//batch_size
        bstart = args.bstart
        

        for i in range(num_batches):
            full_name = FORCE_DIR + args.force_path + '_' + str(bstart) + '_' + str(batch_size)
            org_full_name = full_name + '_x_org.npy'
            imp_full_name = full_name + '_x_imp.npy'
            mask_full_name = full_name + '_mask_imp.npy'
            y_full_name = full_name + '_y.npy'

            x_org_batch = np.load(org_full_name)
            x_imp_batch = np.load(imp_full_name)
            mask_batch = np.load(mask_full_name)
            y_batch = np.load(y_full_name)

            x_org.append(x_org_batch)
            x_imp.append(x_imp_batch)
            mask.append(mask_batch)
            y.append(y_batch)

            bstart += batch_size

        x_org = np.concatenate(x_org)
        x_imp = np.concatenate(x_imp)
        mask = np.concatenate(mask)
        y = np.concatenate(y)

        final_name = SAVE_DIR + args.force_path + '_' + str(args.bstart) + '_' + str(sample_size)
        
        np.save(final_name + '_x_org.npy', x_org)
        np.save(final_name + '_x_imp.npy', x_imp)
        np.save(final_name + '_mask.npy', mask)
        np.save(final_name + '_y.npy', y)

        print("saved at:", final_name)
    
    else:

        x_imp = load_imp_data(args, args.eval, args.target)

        mnist = input_data.read_data_sets('MNIST_data', one_hot=False, validation_size=0, reshape=False)
        mnist = mnist.test if args.eval else mnist.train
        x_org = mnist.images
        y = mnist.labels

        x_org = x_org[args.bstart: args.sample_size, ...]
        x_imp = x_imp[args.bstart: args.sample_size, ...]
        y = y[args.bstart: args.sample_size, ...]


    return x_org, x_imp, y


def safe_validation(x, y, model, sess, start_eps=0.3, end_eps=0.3, step_size=0.1, num_steps=100, val_num=20):

    cur_eps = start_eps

    true_mask = [True for _ in range(len(x))]
    mean_loss = []
    rand = True if val_num > 1 else False

    while cur_eps <= end_eps:
        pgd = LinfPGDAttack(model, cur_eps, num_steps=num_steps, step_size=step_size, random_start=rand, loss_func='xent')

        for i in range(val_num):
            x_adv, _ = pgd.perturb(x, y, sess)

            corr_mask, loss = sess.run([model.correct_prediction, model.y_xent],
                                feed_dict={model.x_input: x_adv,
                                           model.y_input: y})

            true_mask *= corr_mask
            mean_loss.append(np.array(loss).reshape(-1, 1))
            #if sum(true_mask)==0:
            #    break

        #if sum(true_mask)==0:
        #    break

        cur_eps += 1
        mean_loss = np.concatenate(mean_loss, axis=1)
        mean_loss = np.mean(mean_loss, axis=1)

    return sum(true_mask), mean_loss, true_mask


def result(x_imp, model, sess, x_full_batch, y_full_batch, args):
    num_eval_examples = x_imp.shape[0]
    eval_batch_size = min(num_eval_examples, 100)
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    imp_loss_li = []
    #imp_clean_mask = []

    #y_pred = []
    total_corr = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_full_batch[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_batch}
        cur_corr, y_pred_batch = sess.run([model.num_correct, model.predictions],
                                          feed_dict=dict_adv)
        total_corr += cur_corr
    accuracy = total_corr / num_eval_examples

    print('nat Accuracy: {:.2f}%'.format(100.0 * accuracy))

    total_corr = 0
    ''' 
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_full_batch[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        cur_corr, cur_loss, _ = safe_validation(x_batch, y_batch, model, sess) 
        total_corr += cur_corr

    accuracy = total_corr / num_eval_examples
    '''
    accuracy = 0
    print('nat(PGD) Accuracy: {:.2f}%'.format(100.0 * accuracy))

    total_corr = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_imp[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_batch}
        cur_corr, y_pred_batch, correct_prediction, losses = \
            sess.run([model.num_correct, model.predictions, model.correct_prediction, model.y_xent],
                     feed_dict=dict_adv)
        total_corr += cur_corr
        accuracy = total_corr / num_eval_examples
        #y_pred.append(y_pred_batch)
        #imp_clean_mask.append(correct_prediction)

    print('imp Accuracy: {:.2f}%'.format(100.0 * accuracy))
    #y_pred = np.concatenate(y_pred)
    #np.save('gt.npy', y_full_batch)
    #np.save('preds.npy', y_pred)

    #imp_clean_mask = np.concatenate(imp_clean_mask)
    #np.save('nat_label_infer.npy', imp_clean_mask)
    #sys.exit()

    total_corr = 0
    imp_mask = []
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_imp[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        cur_corr, cur_loss, cur_mask = safe_validation(x_batch, y_batch, model, sess)
        total_corr += cur_corr
        imp_loss_li.append(cur_loss)
        imp_mask.append(cur_mask)

    accuracy = total_corr / num_eval_examples

    print('imp(PGD) Accuracy: {:.2f}%'.format(100.0 * accuracy))

    l2_dist = np.linalg.norm((x_imp-x_full_batch).reshape(len(x_imp), -1)/255.0, axis=1).mean()
    linf_dist = np.amax(np.abs((x_imp-x_full_batch)/255.0))
    print('l2_dist:', l2_dist)
    print('linf_dist:', linf_dist)

    #imp_loss = np.concatenate(imp_loss_li)
    #if args.target_y >= 0:
    #    np.save(final_name + '_loss' + str(args.target_y) + '.npy', imp_loss)
    #np.save(final_name + '_loss.npy', imp_loss)

    #imp_mask = np.concatenate(imp_mask)
    #np.save(final_name + '_mask.npy', imp_mask)

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='adv_trained', type=str)
    parser.add_argument('--bstart', default=0, type=int)
    parser.add_argument('--sample_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--target', default=-1, type=int)
    parser.add_argument('--force_path', default='', type=str)
    parser.add_argument('--load_arr', default='./mnist_data/', type=str)
    parser.add_argument('--delta', default=0.3, type=float)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    for key, val in vars(args).items():
        print('{}={}'.format(key, val))

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint('models/' + args.model_dir)
    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model()
    saver = tf.train.Saver()

    x_org, x_imp, y = merge(args)

    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    with tf.Session(config=configs) as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        if np.amax(x_imp) > 255.0001 or \
            np.amin(x_imp) < -0.0001 or \
            np.isnan(np.amax(x_imp)):
            print('Invalid pixel range in x_imp. Expected [0,255], fount[{},{}]'.format(np.amin(x_imp),
                                                                                        np.amax(x_imp)))
        else:
            num_tests = 1
            for _ in range(num_tests):
                result(x_imp, model, sess, x_org, y, args)
                print()


