"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from PIL import Image

import math
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# import os

from impenetrable_batch import Impenetrable
from safe_maml import result
from model import Model

if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', default=10000, help='sample size', type=int)
    parser.add_argument('--bstart', default=0, type=int)
    parser.add_argument('--model_dir', default='naturally_trained', type=str)
    parser.add_argument('--corr_only', action='store_true')
    parser.add_argument('--fail_only', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--loss_func', default='xent', type=str)
    # PGD (training)
    parser.add_argument('--pgd_eps', default=0.3, help='Attack eps', type=float)
    parser.add_argument('--pgd_num_steps', default=100, type=int)
    parser.add_argument('--pgd_step_size', default=0.01, type=float)
    parser.add_argument('--pgd_random_start', action='store_true')
    parser.add_argument('--pgd_restarts', default=20, help="training PGD restart numbers per eps", type=int)
    # impenetrable
    parser.add_argument('--imp_delta', default=0, help='<= 0 for no imp eps', type=float)
    parser.add_argument('--imp_random_start', default=0, help='eps for random start of image', type=float)
    parser.add_argument('--imp_random_seed', default=0, help='random seed for random start of image', type=int)
    parser.add_argument('--imp_gray_start', action='store_true')
    parser.add_argument('--imp_num_steps', default=100, help='0 for until convergence', type=int)
    parser.add_argument('--imp_step_size', default=0.01, type=float)
    parser.add_argument('--imp_rep', action='store_true', help='use reptile instead of MAML')
    parser.add_argument('--imp_pp', default=0, help='step intervals to sum PGD gradients. <= 0 for pure MAML', type=int)
    parser.add_argument('--imp_adam', action='store_true')
    parser.add_argument('--imp_rms', action='store_true')
    parser.add_argument('--imp_adagrad', action='store_true')
    parser.add_argument('--imp_no_sign', action='store_true')
    parser.add_argument('--label_infer', action='store_true')
    parser.add_argument('--source', default=-1, type=int, help='set source label if >= 0')
    parser.add_argument('--target', default=-1, type=int, help='set target label if >= 0')
    # PGD (evaluation)
    parser.add_argument('--val_step_per', default=0, help="validation per val_step. =< 0 means no eval", type=int)
    parser.add_argument('--val_eps', default=0.3, help='Evaluation eps', type=float)
    parser.add_argument('--val_num_steps', default=100, help="validation PGD number of steps per PGD", type=int)
    parser.add_argument('--val_restarts', default=0.01, help="validation PGD restart numbers per eps", type=int)
    args = parser.parse_args()
    for key, val in vars(args).items():
        print('{}={}'.format(key, val))

    assert not (args.corr_only and args.fail_only)
    assert not (args.imp_rep and args.imp_pp)

    # numpy options
    np.set_printoptions(precision=6, suppress=True)
    
    if args.imp_random_start > 0:
        np.random.seed(args.imp_random_seed)
        print('random seed set to:', args.imp_random_seed)

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint('models/' + args.model_dir)
    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model()
    impenet = Impenetrable(model,
                           args)
    saver = tf.train.Saver()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=False, validation_size=0, reshape=False)

    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    with tf.Session(config=configs) as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        # Set number of examples to evaluate
        num_eval_examples = args.sample_size

        indices = [i for i in range(max(args.sample_size + args.bstart, 10000))]

        # load data
        bstart = args.bstart
        while True:
            '''
            x_candid = cifar.eval_data.xs[indices[bstart:bstart + 100]]
            y_candid = cifar.eval_data.ys[indices[bstart:bstart + 100]]
            mask, logits = sess.run([model.correct_prediction, model.pre_softmax],
                                    feed_dict={model.x_input: x_candid,
                                               model.y_input: y_candid})
            x_masked = x_candid[mask]
            y_masked = y_candid[mask]
            logit_masked = logits[mask]
            print(len(x_masked))
            if bstart == args.bstart:
                x_full_batch = x_masked[:min(num_eval_examples, len(x_masked))]
                y_full_batch = y_masked[:min(num_eval_examples, len(y_masked))]
                logit_full_batch = logit_masked[:min(num_eval_examples, len(logit_masked))]
            else:
                index = min(num_eval_examples - len(x_full_batch), len(x_masked))
                x_full_batch = np.concatenate((x_full_batch, x_masked[:index]))
                y_full_batch = np.concatenate((y_full_batch, y_masked[:index]))
                logit_full_batch = np.concatenate((logit_full_batch, logit_masked[:index]))
            bstart += 100
            if (len(x_full_batch) >= num_eval_examples) or bstart >= 10000:
                break
            '''
            if args.eval:
                x_candid = mnist.test.images[indices[bstart:bstart + 100]]
                y_candid = mnist.test.labels[indices[bstart:bstart + 100]]
            else:
                x_candid = mnist.train.images[indices[bstart:bstart + 100]]
                y_candid = mnist.train.labels[indices[bstart:bstart + 100]]
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

            if args.source >= 0:
                mask = y_full_batch == args.source
                x_full_batch = x_full_batch[mask]
                y_full_batch = y_full_batch[mask]
            bstart += 100
            if (len(x_full_batch) >= num_eval_examples) or bstart >= len(indices):
                break

        # Adjust num_eval_examples. Iterate over the samples batch-by-batch
        num_eval_examples = len(x_full_batch)

        eval_batch_size = min(100, num_eval_examples)
        assert num_eval_examples < 100 or num_eval_examples%100==0

        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        print('Iterating over {} batches'.format(num_batches))

        x_full_batch = x_full_batch.astype(np.float32)

        x_imp = []  # imp accumulator
        mask_imp = []  # num steps accumulator
        l2_li = [] # l2 distance accumulator

        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}'.format(bend - bstart))

            x_batch = x_full_batch[bstart:bend, :]
            y_batch = y_full_batch[bstart:bend]

            # run our algorithm
            print('fortifying image {} - {}'.format(bstart, bend-1))
            x_batch_imp, mask_batch_imp = impenet.fortify(x_batch, y_batch if args.target < 0 else
                                                          np.ones_like(y_batch).astype(int) * args.target, sess)
            print()

            # evaluation
            # y_batch_hard = np.argmax(y_batch, axis=1)
            # y_batch_hard = np.eye(10)[y_batch_hard.reshape(-1)]

            # suc_flag = impenet.validation(x_batch_imp, y_batch, y_batch_hard)

            x_imp.append(x_batch_imp)
            mask_imp.append(mask_batch_imp)
            l2_li.append(np.linalg.norm(x_batch_imp - x_batch))
            print('l2 distance (curr):', np.linalg.norm(x_batch_imp - x_batch))
            print('l2 distance (total):', np.mean(l2_li))
            print()
            print('------------------------------------------------------------------------------')
            print('------------------------------------------------------------------------------')

        x_imp = np.concatenate(x_imp)
        mask_imp = np.concatenate(mask_imp)

        # save image
        folder_name = './mnist_data/imp_adv_fixed/' if args.model_dir == 'adv_trained' \
            else './mnist_data/imp_nat_fixed/'
        file_name = 'imp_' + ('eval' if args.eval else 'train') + '_fixed' + '_' + str(args.imp_delta)
        batch_name = '_' + str(args.bstart) + '_' + str(args.sample_size)
        target_name = '' if args.target < 0 else '_' + str(args.target)
        common_name = folder_name + file_name + batch_name + target_name
        
        np.save(common_name, x_imp)

        print('total success rate: {:.2f}% ({}/{})'.format(np.mean(mask_imp)*100,
                                                           np.sum(mask_imp), np.size(mask_imp)))

        # sanity check
        if np.amax(x_imp) > 1.0001 or \
            np.amin(x_imp) < -0.0001 or \
            np.isnan(np.amax(x_imp)):
            print('Invalid pixel range in x_imp. Expected [0,255], found[{},{}]'.format(np.amin(x_imp),
                                                                                        np.amax(x_imp)))
        else:
            result(x_imp, model, sess, x_full_batch, y_full_batch)


