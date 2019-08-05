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
import cifar10_input

def merge():
    pass

def result(x_imp, x_adv, model, sess, x_full_batch, y_full_batch):
    num_eval_examples = x_imp.shape[0]
    eval_batch_size = min(num_eval_examples, 100)
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

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

        x_batch = x_adv[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]
        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_batch}
        cur_corr, y_pred_batch, correct_prediction, losses = \
            sess.run([model.num_correct, model.predictions, model.correct_prediction, model.y_xent],
                     feed_dict=dict_adv)
        total_corr += cur_corr
        accuracy = total_corr / num_eval_examples

    print('adv Accuracy: {:.2f}%'.format(100.0 * accuracy))


if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', default=1000, help='sample size', type=int)
    parser.add_argument('--bstart', default=0, type=int)
    parser.add_argument('--model_dir', default='adv_trained', type=str)
    parser.add_argument('--save_dir_num', default=10, type=int)
    parser.add_argument('--loss_func', default='xent', type=str)
    # PGD
    parser.add_argument('--eps', default=8, help='Attack eps', type=int)
    parser.add_argument('--pgd_random_start', action='store_true')
    parser.add_argument('--pgd_num_steps', default=20, type=int)
    parser.add_argument('--pgd_step_size', default=2, type=float)
    # impenetrable
    parser.add_argument('--imp_random_start', action='store_true')
    parser.add_argument('--imp_gray_start', action='store_true')
    parser.add_argument('--imp_num_steps', default=1000, help='0 for until convergence', type=int)
    parser.add_argument('--res_num_steps', default=1, type=int)
    parser.add_argument('--res_step_size', default=1, type=int)
    # evaluation
    parser.add_argument('--val_step', default=100, help="validation per val_step iterations. =< 0 means no evaluation", type=int)
    parser.add_argument('--val_num', default=100, help="validation PGD numbers per eps", type=int)
    params = parser.parse_args()
    for key, val in vars(params).items():
        print('{}={}'.format(key, val))

    assert not (params.imp_random_start and params.imp_gray_start)

    meta_name = 'nat' if params.model_dir=='naturally_trained' else 'adv'
    meta_name += '_pgd' + '_' + str(params.eps) + '_' + str(params.pgd_num_steps) + '_' + str(params.pgd_step_size) + ('_rand' if params.pgd_random_start else '')
    meta_name += '_imp' + '_' + str(params.imp_num_steps) + ('_rand' if params.imp_random_start else '') + ('_gray' if params.imp_gray_start else '')
    meta_name += '_res' + '_' + str(params.res_num_steps) + '_' + str(params.res_step_size)

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint('models/' + params.model_dir)
    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model(mode='eval')
    impenet = Impenetrable(model,
                           params)
    saver = tf.train.Saver()

    data_path = config['data_path']
    cifar = cifar10_input.CIFAR10Data(data_path)

    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    with tf.Session(config=configs) as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        # Iterate over the samples batch-by-batch
        num_eval_examples = params.sample_size + params.bstart
        if params.val_step > 0:
            eval_batch_size = 1
        else:
            eval_batch_size = min(config['eval_batch_size'], num_eval_examples)

        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        x_org = []  # org accumulator
        x_imp = []  # imp accumulator
        x_adv = []  # adv accumulator
        y_imp = []  # y accumulator

        if params.model_dir=='naturally_trained':
            indices = np.load('/data/home/gaon/lazy-attack/cifar10_data/nat_indices_untargeted.npy')
        else:
            indices = np.load('/data/home/gaon/lazy-attack/cifar10_data/indices_untargeted.npy')

        bstart = 0
        while (True):
            x_candid = cifar.eval_data.xs[indices[bstart:bstart + 100]]
            y_candid = cifar.eval_data.ys[indices[bstart:bstart + 100]]
            mask = sess.run(model.correct_prediction, feed_dict={model.x_input: x_candid,
                                                                 model.y_input: y_candid})
            x_masked = x_candid[mask]
            y_masked = y_candid[mask]
            print(len(x_masked))
            if bstart == 0:
                x_full_batch = x_masked[:min(num_eval_examples, len(x_masked))]
                y_full_batch = y_masked[:min(num_eval_examples, len(y_masked))]
            else:
                index = min(num_eval_examples - len(x_full_batch), len(x_masked))
                x_full_batch = np.concatenate((x_full_batch, x_masked[:index]))
                y_full_batch = np.concatenate((y_full_batch, y_masked[:index]))
            bstart += 100
            if len(x_full_batch) >= (num_eval_examples) or bstart >= 10000:
                break

        print('Iterating over {} batches'.format(num_batches))

        x_full_batch = x_full_batch.astype(np.float32)

        assert params.bstart % eval_batch_size == 0

        for ibatch in range(num_batches):
            if ibatch * eval_batch_size >= params.bstart:
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)
                print('batch size: {}'.format(bend - bstart))

                x_batch = x_full_batch[bstart:bend, :]
                y_batch = y_full_batch[bstart:bend]

                print('fortifying image ', bstart)
                x_batch_imp = impenet.fortify(x_batch, y_batch, ibatch, meta_name, sess)

                # evaluation
                x_batch_adv = np.copy(x_batch_imp)

                val_eps = 20
                obj_eps = 20
                num_images = len(x_batch_imp)

                impenet.pgd.num_steps = 20

                success_mask = [True for _ in range(num_images)]

                while val_eps <= obj_eps:
                    val_iter = impenet.val_num
                    val_total_corr = 0

                    impenet.pgd.epsilon = val_eps
                    impenet.pgd.step_size = impenet.pgd.epsilon//4

                    for i in range(val_iter):

                        x_batch_adv = impenet.pgd.perturb(x_batch_imp, y_batch, sess,
                                                          proj=True, reverse=False, rand=True)

                        corr_mask = sess.run(impenet.model.correct_prediction,
                                                       feed_dict={impenet.model.x_input: x_batch_adv,
                                                                  impenet.model.y_input: y_batch})

                        success_mask *= corr_mask
                        num_survived = np.sum(success_mask) / len(success_mask)

                    print("{} validation accuracy: {:.2f}%".format(val_eps, num_survived * 100))

                    # goal achievement check
                    if num_survived >= 1:
                        print("reached performance goal for", val_eps)

                        if val_eps == obj_eps:
                            print("reached final objective!")
                            break
                    else:
                        print("failed for", val_eps)

                    val_eps += 1
                
                impenet.pgd.epsilon = impenet.eps
                impenet.pgd.step_size = impenet.pgd_step_size
                impenet.pgd.num_steps = impenet.pgd_num_steps
                

                x_org.append(x_batch)
                x_imp.append(x_batch_imp)
                x_adv.append(x_batch_adv)
                y_imp.append(y_batch)

        x_org = np.concatenate(x_org)
        x_imp = np.concatenate(x_imp)
        x_adv = np.concatenate(x_adv)
        y_imp = np.concatenate(y_imp)

        folder_name = './arr' + '_main' + '/'
        batch_name = '_' + str(params.bstart) + '_' + str(params.sample_size)
        common_name = folder_name + meta_name + batch_name
        
        np.save(common_name + '_x_org', x_org)
        np.save(common_name + '_x_imp', x_imp)
        np.save(common_name + '_y', y_imp)

        if np.amax(x_imp) > 255.0001 or \
            np.amin(x_imp) < -0.0001 or \
            np.isnan(np.amax(x_imp)):
            print('Invalid pixel range in x_imp. Expected [0,255], fount[{},{}]'.format(np.amin(x_imp),
                                                                                        np.amax(x_imp)))
        elif np.amax(x_adv) > 255.0001 or \
            np.amin(x_adv) < -0.0001 or \
            np.isnan(np.amax(x_adv)):
            print('Invalid pixel range in x_adv. Expected [0,255], fount[{},{}]'.format(np.amin(x_adv),
                                                                                        np.amax(x_adv)))
        else:
            result(x_imp, x_adv, model, sess, x_org, y_imp)


