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

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from tools.inception_v3_imagenet import model
from tools.utils import get_image

IMAGENET_PATH = './data/'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--eps', default='0.05', help='Attack eps', type=float)
    parser.add_argument('--sample_size', default=1000, help='sample size', type=int)
    parser.add_argument('--samples_per_batch', default=200, help='samples per batch', type=int)
    parser.add_argument('--loss_func', default='xent', help='loss func', type=str)
    parser.add_argument('--model_dir', default='nat', help='model name', type=str)
    parser.add_argument('--log', default = 'n', help='use logged value', type=str)
    params = parser.parse_args()
    for key, val in vars(params).items():
        print('{}={}'.format(key,val))

class Submodular:
    def __init__(self, sess, model, epsilon, loss_func):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
             point."""
        self.model = model
        self.epsilon = epsilon
        self.loss_func = loss_func

        self.model_x = tf.placeholder(tf.float32, (None, 299, 299, 3))
        self.model_y = tf.placeholder(tf.int64, None)
        
        self.logits, self.predictions = model(sess, self.model_x, params.model_dir)
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels = self.model_y)
        self.correct_prediction = tf.equal(self.predictions, self.model_y)
        self.num_correct = tf.reduce_sum(
                tf.cast(self.correct_prediction, tf.int32))
        if params.log == 'y':
            print('using log softmax')
            self.loss = tf.math.log(y_xent)
        else:
            print('using softmax')
            self.loss = y_xent

    def test(self, x_nat, y, sess, ibatch):
        _, xt, yt, zt = x_nat.shape
        batch_size = 50
        assert params.samples_per_batch % batch_size == 0
        num_batches = int(math.ceil(params.samples_per_batch / batch_size))
        samples = [(xi, yi, zi) for xi in range(xt) for yi in range(yt) for zi in range(zt)]
        
        count = 0
        for i in range(num_batches):
            img_batch = np.tile(x_nat, (4*batch_size, 1, 1, 1))
            label_batch = np.tile(y, (4*batch_size))
            
            for j in range(batch_size):
                A_size = np.random.randint(len(samples))
                shuffled = np.random.permutation(samples)
                A = shuffled[:A_size]
                
                B_size = np.random.randint(len(samples))
                shuffled = np.random.permutation(samples)
                B = shuffled[:B_size]

                A = set([tuple(x) for x in A])
                B = set([tuple(x) for x in B])
                intersect = A & B
                union = A | B
                
                A_noise = np.ones_like(np.reshape(x_nat, (xt, yt, zt))) * (- params.eps)
                for tup in A:
                    xi, yi, zi = tup
                    A_noise[xi, yi, zi] *= -1
                img_batch[j] = np.clip(x_nat + A_noise, 0, 1)
               
                B_noise = np.ones_like(np.reshape(x_nat, (xt, yt, zt))) * (- params.eps)
                for tup in B:
                    xi, yi, zi = tup
                    B_noise[xi, yi, zi] *= -1
                img_batch[batch_size+j] = np.clip(x_nat + B_noise, 0, 1)

                intersect_noise = np.ones_like(np.reshape(x_nat, (xt, yt, zt))) * (- params.eps)
                for tup in intersect:
                    xi, yi, zi = tup
                    intersect_noise[xi, yi, zi] *= -1
                img_batch[batch_size*2+j] = np.clip(x_nat + intersect_noise, 0, 1)
                
                union_noise = np.ones_like(np.reshape(x_nat, (xt, yt, zt))) * (- params.eps)
                for tup in union:
                    xi, yi, zi = tup
                    union_noise[xi, yi, zi] *= -1
                img_batch[batch_size*3+j] = np.clip(x_nat + union_noise, 0, 1)
           
            losses = sess.run(self.loss, feed_dict={self.model_x: img_batch,
                                                    self.model_y: label_batch})
            for j in range(batch_size):
                if losses[j] + losses[batch_size+j] >= losses[batch_size*2+j] + losses[batch_size*3+j]:
                    count += 1
        
        return count / params.samples_per_batch

if __name__ == '__main__':
    import json

    with open('config.json') as config_file:
        config = json.load(config_file)

    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    with tf.Session(config=configs) as sess:

        # set attack
        tester = Submodular(sess, model,
                            params.eps,
                            params.loss_func)
        # Iterate over the samples batch-by-batch
        num_eval_examples = params.sample_size
        eval_batch_size = 1
        #num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
        x_adv = [] # adv accumulator

        bstart = 0
        x_full_batch = []
        y_full_batch = []
        print('loading image data')
        while(True):
            x_candid = []
            y_candid = []
            for i in range(100):
                img_batch, y_batch = get_image(bstart+i, IMAGENET_PATH)
                img_batch = np.reshape(img_batch, (-1, *img_batch.shape))
                if i == 0:
                    x_candid = img_batch
                    y_candid = np.array([y_batch])
                else:
                    x_candid = np.concatenate([x_candid, img_batch])
                    y_candid = np.concatenate([y_candid, [y_batch]])
            logits, preds = sess.run([tester.logits, tester.predictions],
                                     feed_dict={tester.model_x: x_candid,
                                                tester.model_y: y_candid})
            idx = np.where(preds == y_candid)
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
            if len(x_full_batch) >= num_eval_examples:
                    break
        percentages = []
        
        print('Iterating over {} batches\n'.format(num_batches))
        for ibatch in range(num_batches):
            print('attacking {}th image...'.format(ibatch))
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            x_batch = x_full_batch[bstart:bend, :]
            y_batch = y_full_batch[bstart:bend]

            start = time.time()

            percentage = tester.test(x_batch, y_batch, sess, ibatch)
            percentages.append(percentage)
            percentage_mean = sum(percentages) / len(percentages)
            print('submodularity percentage:{:.2f}% over {} samples'.format(percentage*100, params.samples_per_batch))
            print('percentage mean:{:.2f}%'.format(percentage_mean*100))
            end = time.time()
            print('time taken:{}'.format(end - start))
            print()
        
        print('eps:{}, loss function:{}'.format(params.eps, params.loss_func))
        print('percentage mean:{:.2f}%'.format(percentage_mean*100))
