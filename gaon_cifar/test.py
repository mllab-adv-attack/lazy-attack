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


def load_files(args):
    load_arr = args.load_arr
    save_arr = args.save_arr
    file_name = args.file_name
    bstart = args.bstart
    batch_size = args.batch_size
    sample_size = args.sample_size

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

    final_name = save_arr + file_name + '_' + str(args.bstart) + '_' + str(sample_size)

    np.save(final_name + '_x_org.npy', x_org)
    np.save(final_name + '_x_imp.npy', x_imp)
    np.save(final_name + '_step_imp.npy', step)
    np.save(final_name + '_y.npy', y)

    print("saved at:", final_name)

    return x_org, x_imp, y


def safe_validation(x, y, model, sess, start_eps=8, end_eps=8, val_num=1):

    cur_eps = start_eps

    true_mask = [True for _ in range(len(x))]
    mean_loss = []

    while cur_eps <= end_eps:
        pgd = LinfPGDAttack(model, cur_eps, num_steps=20, step_size=cur_eps/4,
                            random_start=False if val_num <=1 else True, loss_func='xent')

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

    assert num_eval_examples%10 == 0
    target_batch_size = num_eval_examples//10
    y_target = np.copy(y_full_batch)
    for  i in range(10):
        y_target[100*i: 100*(i+1)] = i

    total_loss = np.zeros(eval_batch_size)
    total_corr = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_full_batch[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_batch}
        cur_corr, cur_loss, y_pred_batch = sess.run([model.num_correct, model.y_xent, model.predictions],
                                          feed_dict=dict_adv)
        total_corr += cur_corr
        total_loss += cur_loss
    accuracy = total_corr / num_eval_examples
    nat_avg_loss = total_loss / num_batches

    print('nat Accuracy: {:.2f}%'.format(100.0 * accuracy))
    print('nat Loss: {:.6f}'.format(nat_avg_loss.mean()))

    # get full loss batch
    loss_li = []
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_full_batch[bstart:bend, :]
        y_source_batch = y_full_batch[bstart:bend]
        y_target_batch = y_target[bstart:bend]
        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_target_batch}
        cur_corr, cur_loss, y_pred_batch = sess.run([model.num_correct, model.y_xent, model.predictions],
                                                    feed_dict=dict_adv)
        total_corr += cur_corr
        total_loss += cur_loss
        loss_li.append(cur_loss)
    loss_li = np.concatenate(loss_li)
    loss_li = loss_li.reshape((-1, 10)).T

    # detection loss
    success_rate = np.argmax(loss_li, axis=1) == y_full_batch[0]
    print(success_rate.mean())
    return

    total_loss = np.zeros(eval_batch_size)
    total_corr = 0
    '''
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_full_batch[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        cur_corr, cur_loss, _ = safe_validation(x_batch, y_batch, model, sess, val_num=args.val_restarts)
        total_corr += cur_corr
        total_loss += cur_loss

    '''
    accuracy = total_corr / num_eval_examples
    nat_pgd_avg_loss = total_loss / num_batches
    print('nat(PGD) Accuracy: {:.2f}%'.format(100.0 * accuracy))
    print('nat(PGD) Loss: {:.6f}'.format(nat_pgd_avg_loss.mean()))

    total_loss = np.zeros(eval_batch_size)
    total_corr = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_imp[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_batch}
        cur_corr, y_pred_batch, correct_prediction, cur_loss = \
            sess.run([model.num_correct, model.predictions, model.correct_prediction, model.y_xent],
                     feed_dict=dict_adv)
        total_corr += cur_corr
        total_loss += cur_loss

    accuracy = total_corr / num_eval_examples
    imp_avg_loss = total_loss / num_batches

    print('imp Accuracy: {:.2f}%'.format(100.0 * accuracy))
    print('imp Loss: {:.6f}'.format(imp_avg_loss.mean()))

    total_loss = np.zeros(eval_batch_size)
    total_corr = 0
    imp_loss_li = []
    imp_mask = []
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_imp[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        cur_corr, cur_loss, cur_mask = safe_validation(x_batch, y_batch, model, sess, val_num=args.val_restarts)
        total_corr += cur_corr
        total_loss += cur_loss
        imp_loss_li.append(cur_loss)
        imp_mask.append(cur_mask)

    accuracy = total_corr / num_eval_examples
    imp_pgd_avg_loss = total_loss / num_batches

    print('imp(PGD) Accuracy: {:.2f}%'.format(100.0 * accuracy))
    print('imp(PGD) Loss: {:.6f}'.format(imp_pgd_avg_loss.mean()))

    l2_dist = np.linalg.norm((x_imp-x_full_batch).reshape(len(x_imp), -1)/255.0, axis=1).mean()
    linf_dist = np.amax(np.abs((x_imp-x_full_batch)/255.0))
    print('l2_dist:', l2_dist)
    print('linf_dist:', linf_dist)

    imp_loss = np.concatenate(imp_loss_li)
    np.save( + '_loss' + str(args.target_y) + '.npy', imp_loss)
    np.save(final_name + '_loss.npy', imp_loss)

    imp_mask = np.concatenate(imp_mask)
    np.save(final_name + '_mask.npy', imp_mask)


if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--bstart', default=0, type=int)
    parser.add_argument('--model_dir', default='adv_trained', type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--sample_size', default=100, type=int)
    parser.add_argument('--val_restarts', default=20, type=int)
    parser.add_argument('--target_y', default=-1, type=int)
    parser.add_argument('--load_arr', default='./arr_main/', type=str)
    parser.add_argument('--save_arr', default='./arr_new/', type=str)
    args = parser.parse_args()
    for key, val in vars(args).items():
        print('{}={}'.format(key, val))

    assert args.sample_size % args.batch_size == 0

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_dir = args.model_dir
    load_arr = args.load_arr
    save_arr = args.save_arr

    model_file = tf.train.latest_checkpoint('models/' + model_dir)
    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model(mode='eval')
    saver = tf.train.Saver()

    file_name_pre = 'adv_pgd_8_20_2_rand_1_imp_8.0_100_2.0_0_0_0_'
    file_name_pos = '_rms_val_0_8_20_20_0_100'
    file_name_li = [(load_arr + file_name_pre + 's{}_t{}'.format(i, j) + file_name_pos) for
                    j in range(10) for i in range(10)]

    org_li = []
    imp_li = []
    y_li = []
    for file_name in file_name_li:
        org_file_name = file_name + '_x_org.npy'
        imp_file_name = file_name + '_x_imp.npy'
        step_file_name = file_name + '_step_imp.npy'
        y_file_name = file_name + '_y.npy'

        x_org_batch = np.load(org_file_name)
        x_imp_batch = np.load(imp_file_name)
        y_batch = np.load(y_file_name)
        #y_batch = np.argmax(y_batch, axis=1)

        org_li.append(x_org_batch)
        imp_li.append(x_imp_batch)
        y_li.append(y_batch)

    org_full = np.concatenate(org_li)
    imp_full = np.concatenate(imp_li)
    y_full = np.concatenate(y_li)

    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    with tf.Session(config=configs) as sess:

        # Restore the checkpoint
        saver.restore(sess, model_file)
        for source in range(10):
            x_org = org_full[1000*source: 1000*(source+1)]
            x_imp = imp_full[1000*source: 1000*(source+1)]
            y = y_full[1000*source: 1000*(source+1)]

            assert np.amax(y) <= source + 1e-5 and np.amin(y) >= source - 1e-5

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


