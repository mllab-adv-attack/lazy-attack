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
import time

#from PIL import Image

import matplotlib.pyplot as plt
plt.switch_backend('agg')

#from tools.logging_utils import log_output, render_frame
from tools.inception_v3_imagenet import model
#from tools.imagenet_labels import label_to_name
from tools.utils import get_image

IMAGENET_PATH = './../data/'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    #basic
    parser.add_argument('--eps', default=0.05, help='Attack eps', type=float)
    parser.add_argument('--sample_size', default=1000, help='sample size', type=int)
    parser.add_argument('--num_steps', default=20, help='pgd steps', type=int)
    parser.add_argument('--step_size', default=0.002, help='pgd step size', type=float)
    parser.add_argument('--model_dir', default='nat', help='model name', type=str)
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--rand', action='store_true')
    args = parser.parse_args()
    for key, val in vars(args).items():
        print('{}={}'.format(key,val))

class PGDAttack:
    def __init__(self, sess, model, epsilon):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
             point."""
        
        self.model = model
        self.epsilon = epsilon
        
        self.x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))
        self.y_input = tf.placeholder(tf.int32, None)
        
        self.logits, self.predictions = model(sess, self.x_input, args.model_dir)
        self.predictions = tf.cast(self.predictions, tf.int32)
        self.correct_prediction = tf.equal(self.predictions, self.y_input)
        self.num_correct = tf.reduce_sum(
            tf.cast(self.correct_prediction, tf.int32))
        
        self.probs = tf.nn.softmax(self.logits)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.y_input)

        self.grad = tf.gradients(self.loss, self.x_input)[0]

    def perturb(self, x_nat, y, sess):
        
        if args.rand:
            x = x_nat + np.random_uniform(-args.eps, args.eps, x_nat.shape)
        else:
            x = np.copy(x_nat)

        for i in range(args.num_steps):
            grad = sess.run(self.grad, feed_dict={self.x_input: x,
                                                  self.y_input: y})
            
            x = np.add(x, args.step_size * np.sign(grad), out=x, casting='unsafe')

            x = np.clip(x, x_nat - args.eps, x_nat + args.eps)
            x = np.clip(x, 0, 1)
        
        return x
    
    
# evaluate perturbed images
def run_attack(x_adv, sess, attack, x_full_batch, y_full_batch):

    num_eval_samples = x_adv.shape[0]
    eval_batch_size = min(num_eval_samples, 64)

    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_corr = 0

    x_nat = x_full_batch
    l_inf = np.amax(np.abs(x_nat-x_adv))

    # error checking
    if l_inf > args.eps + 0.0001:
        print('breached maximum perturbation')
        print('l_inf value:{}'.format(l_inf))
        return

    y_pred = []
    success = []
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_adv[bstart:bend,:]
        y_batch = y_full_batch[bstart:bend]

        dict_adv = {attack.x_input: x_batch,
                    attack.y_input: y_batch}
        cur_corr, y_pred_batch, correct_prediction = \
            sess.run([attack.num_correct, attack.predictions, attack.correct_prediction],
                     feed_dict=dict_adv)
        total_corr += cur_corr
        y_pred.append(y_pred_batch)
        success.append(np.array(np.nonzero(np.invert(correct_prediction)))+ibatch*eval_batch_size)

    success = np.concatenate(success, axis=1)
    #np.save('out/'+args.attack_type+'_success.npy', success)
    accuracy = total_corr / num_eval_examples
    print('adv Accuracy: {:.2f}%'.format(100.0 * accuracy))
    #with open('out/result.txt', 'a') as f:
    #    f.write('''Resnet, {}, eps:{},
    #                    sample_size:{}, loss_func:{}
    #                    => acc:{}, percentage:{}\n'''.format(args.eps,
    #                                                        args.attack_type,
    #                                                         args.sample_size,
    #                                                         args.loss_func,
    #                                                         accuracy,
    #                                                         percentage_mean))
    

    total_corr = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_full_batch[bstart:bend,:]
        y_batch = y_full_batch[bstart:bend]

        dict_adv = {attack.x_input: x_batch,
                    attack.y_input: y_batch}
        cur_corr, y_pred_batch = sess.run([attack.num_correct, attack.predictions],
                                          feed_dict=dict_adv)
        total_corr += cur_corr

    accuracy = total_corr / num_eval_examples
    print('nat Accuracy: {:.2f}%'.format(100.0 * accuracy))
    y_pred = np.concatenate(y_pred, axis=0)
    #np.save('pred.npy', y_pred)
    #print('Output saved at pred.npy')

if __name__ == '__main__':
    #import sys

    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    with tf.Session(config=configs) as sess:
        # set attack
        attack = PGDAttack(sess, model,
                                  args.eps)
        # Iterate over the samples batch-by-batch
        num_eval_examples = args.sample_size
        eval_batch_size = 100
        if args.test:
            target_indices = np.load('./../data/intersection_norm.npy')
        else:
            target_indices = np.array([i for i in range(50000)])
        if args.shuffle:
            np.random.shuffle(target_indices)

        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
        x_adv = [] # adv accumulator
        
        attack_set = []

        bstart = 0
        x_full_batch = []
        y_full_batch = []
        print('loading image data')
        while(True):
            x_candid = []
            y_candid = []
            for i in range(100):
                img_batch, y_batch = get_image(target_indices[bstart+i], IMAGENET_PATH)
                #img_batch, y_batch = get_image(bstart+i, IMAGENET_PATH)
                img_batch = np.reshape(img_batch, (-1, *img_batch.shape))
                x_candid.append(img_batch)
                y_candid.append(y_batch)
            x_candid = np.concatenate(x_candid, axis=0)
            y_candid = np.array(y_candid)
            logits, preds = sess.run([attack.logits, attack.predictions],
                                     feed_dict={attack.x_input: x_candid,
                                                attack.y_input: y_candid})
            idx = np.where(preds == y_candid)
            for i in idx[0]:
                attack_set.append(bstart+i)
            x_masked = x_candid[idx]
            y_masked = y_candid[idx]
            if bstart == 0:
                x_full_batch = x_masked[:min(num_eval_examples, len(x_masked))]
                y_full_batch = y_masked[:min(num_eval_examples, len(y_masked))]
            else:
                index = min(num_eval_examples-len(x_full_batch), len(x_masked))
                x_full_batch = np.concatenate((x_full_batch, x_masked[:index]))
                y_full_batch = np.concatenate((y_full_batch, y_masked[:index]))
            bstart += 100
            print(len(x_full_batch))
            if len(x_full_batch) >= num_eval_examples or (bstart == 50000):
                break
        #np.save('./imagenet_out/tensorflow_{}'.format(args.sample_size), attack_set)

        print('Iterating over {} batches\n'.format(num_batches))
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('attacking {}-{}th image...'.format(bstart, bend))
            x_batch = x_full_batch[bstart:bend, :]
            y_batch = y_full_batch[bstart:bend]

            start = time.time()

            '''fig = np.reshape(x_batch, (299, 299, 3)).astype(np.uint8).squeeze()
            pic = Image.fromarray(fig)
            pic.save("out/images/"+str(ibatch)+"nat.png")'''
            
            x_batch_adv = attack.perturb(x_batch, y_batch, sess)
            
            '''fig = np.reshape(x_batch_adv, (299, 299, 3)).astype(np.uint8).squeeze()
            pic = Image.fromarray(fig)
            pic.save("out/images/"+str(ibatch)+"adv.png")'''
            
            x_adv.append(x_batch_adv)
            end = time.time()

        x_adv = np.concatenate(x_adv, axis=0)
        
        for key, val in vars(args).items():
            print('{}={}'.format(key,val))

        # error checking
        if np.amax(x_adv) > 1.0001 or \
                np.amin(x_adv) < -0.0001 or \
                np.isnan(np.amax(x_adv)):
            print('Invalid pixel range. Expected [0,1], fount [{},{}]'.format(np.amin(x_adv),
                                                                              np.amax(x_adv)))
        else:
            run_attack(x_adv, sess, attack, x_full_batch, y_full_batch)
