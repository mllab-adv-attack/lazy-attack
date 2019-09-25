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

from pgd_attack import LinfPGDAttack


def merge(params):
    load_arr = params.load_arr
    save_arr = params.save_arr
    file_name = params.file_name
    bstart = params.bstart
    batch_size = params.batch_size
    sample_size = params.sample_size

    x_org = []
    x_imp = []
    step = []
    y = []

    num_batches = sample_size//batch_size

    for i in range(num_batches):
        full_name = load_arr + file_name + '_' + str(bstart) + '_' + str(batch_size)
        org_full_name = full_name + '_x_org.npy'
        imp_full_name = full_name + '_x_imp.npy'
        step_full_name = full_name + '_step_imp.npy'
        y_full_name = full_name + '_y.npy'

        x_org_batch = np.load(org_full_name)
        x_imp_batch = np.load(imp_full_name)
        step_batch = np.load(step_full_name)
        y_batch = np.load(y_full_name)
        #y_batch = np.argmax(y_batch, axis=1)

        x_org.append(x_org_batch)
        x_imp.append(x_imp_batch)
        step.append(step_batch)
        y.append(y_batch)

        bstart += batch_size

    x_org = np.concatenate(x_org)
    x_imp = np.concatenate(x_imp)
    step = np.concatenate(step)
    y = np.concatenate(y)

    final_name = save_arr + file_name + '_' + str(params.bstart) + '_' + str(sample_size)

    np.save(final_name + '_x_org.npy', x_org)
    np.save(final_name + '_x_imp.npy', x_imp)
    np.save(final_name + '_step_imp.npy', step)
    np.save(final_name + '_y.npy', y)

    print("saved at:", final_name)

    return x_org, x_imp, y, final_name

def safe_validation(x, y, model, sess, start_eps=8, end_eps=8, val_num=20):

    cur_eps = start_eps

    true_mask = [True for _ in range(len(x))]
    mean_loss = []

    while cur_eps <= end_eps:
        pgd = LinfPGDAttack(model, cur_eps, num_steps=20, step_size=cur_eps/4, random_start=True, loss_func='xent')

        for i in range(val_num):
            x_adv, _ = pgd.perturb(x, y, sess, rand=True)

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

    return sum(true_mask), mean_loss


def result(x_imp, model, sess, x_full_batch, y_full_batch, final_name, params):
    num_eval_examples = x_imp.shape[0]
    eval_batch_size = min(num_eval_examples, 100)
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    imp_loss_li = []

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
        cur_corr, cur_loss = safe_validation(x_batch, y_batch, model, sess) 
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

    print('imp Accuracy: {:.2f}%'.format(100.0 * accuracy))

    total_corr = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_imp[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        cur_corr, cur_loss = safe_validation(x_batch, y_batch, model, sess)
        total_corr += cur_corr
        imp_loss_li.append(cur_loss)

    accuracy = total_corr / num_eval_examples

    print('imp(PGD) Accuracy: {:.2f}%'.format(100.0 * accuracy))

    l2_dist = np.linalg.norm((x_imp-x_full_batch).reshape(len(x_imp), -1)/255.0, axis=1).mean()
    linf_dist = np.amax(np.abs((x_imp-x_full_batch)/255.0))
    print('l2_dist:', l2_dist)
    print('linf_dist:', linf_dist)

    imp_loss = np.concatenate(imp_loss_li)
    #if params.target_y >= 0:
    #    np.save(final_name + '_loss' + str(params.target_y) + '.npy', imp_loss)
    np.save(final_name + '_loss.npy', imp_loss)

if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--bstart', default=0, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--sample_size', default=1000, type=int)
    parser.add_argument('--target_y', default=-1, type=int)
    parser.add_argument('--load_arr', default='./arr_main/', type=str)
    parser.add_argument('--save_arr', default='./arr_new/', type=str)
    parser.add_argument('--file_name', default='nat_pgd_8_20_2_20_imp_0_1000_5.0_corr_val_10_8_20_20_0', type=str)
    params = parser.parse_args()
    for key, val in vars(params).items():
        print('{}={}'.format(key, val))

    assert params.sample_size % params.batch_size == 0

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)


    if params.file_name[:3]=='nat':
        model_dir = 'naturally_trained'
    elif params.file_name[:3]=='adv':
        model_dir = 'adv_trained'
    else:
        raise Exception

    model_file = tf.train.latest_checkpoint('models/' + model_dir)
    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model(mode='eval')
    saver = tf.train.Saver()

    x_org, x_imp, y, final_name = merge(params)

    if params.target_y >= 0:
        y.fill(params.target_y)

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
                result(x_imp, model, sess, x_org, y, final_name, params)
                print()


