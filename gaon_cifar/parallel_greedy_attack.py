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
import threading
from helper import Greedy, PeekablePriorityQueue

from PIL import Image

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import cifar10_input

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    #basic
    parser.add_argument('--eps', default='8', help='Attack eps', type=int)
    parser.add_argument('--sample_size', default=500, help='sample size', type=int)
    parser.add_argument('--loss_func', default='xent', help='loss func', type=str)
    parser.add_argument('--test', default='y', help='include run attack', type=str)
    parser.add_argument('--model_dir', default='adv_trained', help='model name', type=str)
    parser.add_argument('--early_stop', default='y', help='attack all pixels', type=str)
    parser.add_argument('--attack_type', default='ldg_block', help='attack type', type=str)
    parser.add_argument('--plot_f', default='n', help='plot F(s)', type=str)
    parser.add_argument('--plot_interval', default=1000, help ='plot interval', type=int)
    parser.add_argument('--resize', default=1, help = 'resize ratio', type=int)
    # block partition
    parser.add_argument('--block_size', default=16, help = 'block partition size', type=int)
    parser.add_argument('--partition', default='basic', help = 'block partitioning', type=str)
    parser.add_argument('--admm_iter', default=10, help = 'admm max iteration', type=int)
    parser.add_argument('--block_scheme', default='admm', help = 'convergence scheme', type=str)
    parser.add_argument('--overlap', default=1, help = 'overlap size', type=int)
    parser.add_argument('--admm_rho', default=1e-7, help = 'admm rho', type=float)
    parser.add_argument('--admm_tau', default=4, help ='admm tau', type=float)
    # mask out
    parser.add_argument('--top_k', default = 5, help = 'mask-out top k', type=int)
    parser.add_argument('--max_boxes', default=100, help = 'max boxes', type=int)
    parser.add_argument('--min_scale', default = 0.5, help = 'min ratio of box', type=float)
    # parallel
    parser.add_argument('--gpus', default=4, help = 'num of gpus to use', type=int)
    params = parser.parse_args()
    for key, val in vars(params).items():
        print('{}={}'.format(key,val))

def plot(li, num):
    if num % params.plot_interval == 0:
        x = [(i+1) for i in range(len(li))]
        if params.plot_f == 'y':
            ylabel = 'F(S)'
        else:
            ylabel = 'loss gain'
        plt.scatter(x, li, 10)
        plt.legend([ylabel])
        plt.xlabel('# perturbs')
        plt.ylabel(ylabel)
        plt.title('{}, eps: {}, loss: {}, #{}'.format(params.attack_type, params.eps, params.loss_func, num))
        plt.savefig('out/resnet_'+ylabel+'_graph_{}_{}_{}_{}.png'.format(params.attack_type,
                                                                         params.eps, params.loss_func,
                                                                         num))
        plt.close()


class LazyGreedyAttack:
    def __init__(self, models, epsilon, loss_func):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
             point."""
        
        self.models = models
        self.epsilon = epsilon
        self.loss_func = loss_func
        self.queries = []
        self.block_queries = [0 for i in range(params.gpus)]
        self.block_insert_count = [0 for i in range(params.gpus)]
        self.block_put_count = [0 for i in range(params.gpus)]
        self.query = [0, 0] # [sequential, parallel]
        self.insert_count = 0
        self.put_count = 0
        self.ratios = []
        self.block_success_stat = {}
        self.admm_converge_stat = {}
        self.success = False
        for i in range(params.admm_iter):
            self.block_success_stat[i] = 0
            self.admm_converge_stat[i] = 0

        self.model = self.models[0]
        

    # block partition algorithm
    def block_partition(self, shape):

        block_size = params.block_size
        overlap = params.overlap
        xt, yt, zt = shape
        blocks = []
        # grid
        if params.partition == 'basic':
            num_block_rows = xt//block_size
            num_block_cols = yt//block_size

            assert(xt%block_size==0 and yt%block_size==0)

            for block_xi in range(num_block_rows):
                for block_yi in range(num_block_cols):
                    block = [(block_size*block_xi+xi, block_size*block_yi+yi, zi) \
                             for zi in range(zt) \
                             for yi in range(-overlap, block_size+overlap) \
                             for xi in range(-overlap, block_size+overlap)]
                    block = [index for index in block if (max(index) <= 31 and min(index) >= 0)]
                    blocks.append(block)
        # centered
        elif params.partition == 'centered':
            
            center_start = (xt//4, yt//4, zt)
            center_end = (xt*3//4, yt*3//4, zt)
            
            center_block = [(xi, yi, zi) \
                            for zi in range(zt) \
                            for yi in range(center_start[1]-overlap, center_end[1]+overlap) \
                            for xi in range(center_start[0]-overlap, center_end[0]+overlap)]
            center_block = [index for index in center_block if (max(index) <= 31 and min(index) >= 0)]
            blocks.append(center_block)
            
            upper_block = [(xi, yi, zi) \
                           for zi in range(zt) \
                           for yi in range(yt) \
                           for xi in range(center_start[1]+overlap)]
            upper_block = [index for index in upper_block if (max(index) <= 31 and min(index) >= 0)]
            blocks.append(upper_block)

            left_block1 = [(xi, yi, zi) \
                           for zi in range(zt) \
                           for yi in range(center_start[1]+overlap) \
                           for xi in range(center_start[0]-overlap, center_end[1]+overlap)]
            left_block2 = [(xi, yi, zi) \
                           for zi in range(zt) \
                           for yi in range(yt//2+overlap) \
                           for xi in range(center_end[1]-overlap, xt)]
            left_block = left_block1 + left_block2
            left_block = list(set(left_block))
            left_block = [index for index in left_block if (max(index) <= 31 and min(index) >= 0)]
            blocks.append(left_block)

            right_block1 = [(xi, yi, zi) \
                           for zi in range(zt) \
                           for yi in range(center_end[1]-overlap, yt) \
                           for xi in range(center_start[0]-overlap, center_end[1]+overlap)]
            right_block2 = [(xi, yi, zi) \
                           for zi in range(zt) \
                           for yi in range(yt//2-overlap, yt) \
                           for xi in range(center_end[1]-overlap, xt)]
            right_block = right_block1 + right_block2
            right_block = list(set(right_block))
            right_block = [index for index in right_block if (max(index) <= 31 and min(index) >= 0)]
            blocks.append(right_block)
        else:
            print("unimplemented partition method!")
            raise Exception

        return blocks

    # compute xi - Si z from DOPE
    def admm_loss(self, block, x, z, yk, rho):
        block_dist = []
        x = np.reshape(x, z.shape)
        for index in block:
            xi, yi, zi = index
            block_dist.append(x[0, xi, yi, zi] - z[0, xi, yi, zi])
        
        block_dist = np.array(block_dist)
        
        return np.dot(yk, block_dist) + (rho / 2) * np.dot(block_dist, block_dist)

    # helper function for lazy double greedy ({-1, 1}) w.r.t. given block
    def ldg_block_seg(self, block_results, x_adv, y, sess, model, ith_thread, block, x_m, x_p, yk=0, rho=0, Resize = params.resize):
        insert_count = 0
        put_count = 0
        queue = PeekablePriorityQueue()
        _, xt, yt, zt = x_adv.shape
        resize = Resize
        block = sorted(block)

        label_mask = tf.one_hot(model.y_input,
                                10,
                                on_value=1.0,
                                off_value=0.0,
                                dtype=tf.float32)
        if self.loss_func == 'xent':
            loss = -model.y_xent
        elif self.loss_func == 'cw':
            correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax, axis=1)
            loss = tf.nn.relu(correct_logit - wrong_logit + 50)
        elif self.loss_func == 'gt':
            loss = -tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
        elif self.loss_func == 'xent_v2':
            softmax = tf.nn.softmax(model.pre_softmax, axis=1)
            correct_predict = tf.reduce_sum(label_mask * softmax, axis=1)
            wrong_predict = tf.reduce_max((1-label_mask) * softmax, axis=1)
            loss = tf.nn.relu(correct_predict - wrong_predict + 50)
        else:
            print('not a supported loss function. switching to xent')
            loss = -model.y_xent

        anchor_block = []
        selected = set()
        for index in block:
            if index not in selected:
                anchor_block.append(index)
                xi, yi, zi = index
                for xxi in range(resize):
                    for yyi in range(resize):
                        selected.add((xi+xxi, yi+yyi, zi))
        anchor_block = sorted(anchor_block)
        block = set(block)

        block_x_m = np.copy(x_adv)
        block_x_p = np.copy(x_adv)
        
        num_pixels = len(anchor_block)
        for index in block:
            xi, yi, zi = index
            block_x_m[0, xi, yi, zi] = x_m[0, xi, yi, zi]
            block_x_p[0, xi, yi, zi] = x_p[0, xi, yi, zi]
        cur_m = sess.run(loss, feed_dict={model.x_input:block_x_m,
                                               model.y_input:y})
        
        cur_p = sess.run(loss, feed_dict={model.x_input:block_x_p,
                                               model.y_input:y})

        if params.block_scheme == 'admm':
            cur_m += self.admm_loss(block, block_x_m, x_adv, yk, rho)
            cur_p += self.admm_loss(block, block_x_p, x_adv, yk, rho)

        #print('first pass', ith_thread)
        # first pass
        #start = time.time()
        batch_size = min(100, num_pixels)
        num_batches = num_pixels//batch_size
        block_index1, block_index2 = 0, 0
        for ith_batch in range(num_batches+1):
            if ith_batch == num_batches:
                if num_pixels%batch_size == 0:
                    break
                else:
                    batch_size = num_pixels%batch_size
            img_batch_m = np.tile(block_x_m, (batch_size, 1, 1, 1))
            img_batch_p = np.tile(block_x_p, (batch_size, 1, 1, 1))
            img_batch = np.concatenate([img_batch_m, img_batch_p])
            label_batch = np.tile(y, (2*batch_size))
            for j in range(batch_size):
                xb, yb, zb = anchor_block[block_index1]
                block_index1 += 1
                for xxi in range(resize):
                    for yyi in range(resize):
                        if (xb+xxi, yb+yyi, zi) in block:
                            img_batch[j, xb+xxi, yb+yyi, zb] = block_x_p[0, xb+xxi, yb+yyi, zb]
                            img_batch[batch_size + j, xb+xxi, yb+yyi, zb] = block_x_m[0, xb+xxi, yb+yyi, zb]
            feed_dict = {
                model.x_input: img_batch,
                model.y_input: label_batch}
            losses = sess.run(loss,
                              feed_dict=feed_dict)
            for pos in range(losses.size//2):
                xb, yb, zb = anchor_block[block_index2]
                block_index2 += 1
                
                if params.block_scheme == 'admm':
                    losses[pos] += self.admm_loss(block, img_batch[pos], x_adv, yk, rho)
                    losses[batch_size+pos] += self.admm_loss(block, img_batch[batch_size+pos], x_adv, yk, rho)
                
                pi = losses[pos] - cur_m
                mi = losses[batch_size+pos] - cur_p

                queue.put(Greedy([xb, yb, zb], pi, mi, False))
        num_queries = 2 * num_pixels

            #end = time.time()
            #print('first pass time:', end-start)

        #print('second pass', ith_thread)
        # second pass
        while not queue.empty():
            candid = queue.get()
            second = None

            if not queue.empty():
                second = queue.peek()

            xi, yi, zi = candid.loc
            img_batch = np.concatenate([block_x_m, block_x_p])
            for xxi in range(resize):
                for yyi in range(resize):
                    if (xi+xxi, yi+yyi, zi) in block:
                        img_batch[0, xi+xxi, yi+yyi, zi] = block_x_p[0, xi+xxi, yi+yyi, zi]
                        img_batch[1, xi+xxi, yi+yyi, zi] = block_x_m[0, xi+xxi, yi+yyi, zi]
            y_batch = np.tile(y, 2)

            losses, correct_prediction = sess.run([loss, model.correct_prediction],
                                                  feed_dict={model.x_input: img_batch,
                                                             model.y_input: y_batch})

            num_queries += 2
            if params.early_stop == 'y':
                success = np.array([0, 1])[np.invert(correct_prediction)]
                for i in success:
                    self.block_insert_count[ith_thread] = insert_count
                    self.block_put_count[ith_thread] = put_count
                    self.block_queries[ith_thread] = num_queries
                    self.success = True
                    block_results[ith_thread] = np.reshape(img_batch[i], (1, *img_batch[i].shape))
                    return

            if params.block_scheme == 'admm':
                losses[0] += self.admm_loss(block, img_batch[0], x_adv, yk, rho)
                losses[1] += self.admm_loss(block, img_batch[1], x_adv, yk, rho)

            candid.update(losses[0]-cur_m, losses[1]-cur_p)
            
            if not second or candid <= second:
                put_count += 1
                if candid.getDir():
                    for xxi in range(resize):
                        for yyi in range(resize):
                            if (xi+xxi, yi+yyi, zi) in block:
                                block_x_m[0, xi+xxi, yi+yyi, zi] = block_x_p[0, xi+xxi, yi+yyi, zi]
                    cur_m = losses[0]
                else:
                    for xxi in range(resize):
                        for yyi in range(resize):
                            if (xi+xxi, yi+yyi, zi) in block:
                                block_x_p[0, xi+xxi, yi+yyi, zi] = block_x_m[0, xi+xxi, yi+yyi, zi]
                    cur_p = losses[1]
                
            else:
                insert_count+=1
                queue.put(candid)
            
            # parallel early stopping
            if self.success:
                break

        #print("num of re-inserted pixels:", insert_count)
        #print("num of pertubed pixels:", put_count)
        #print("num of queries:", num_queries)
        '''self.insert_count += insert_count
        self.put_count += put_count
        self.query += num_queries'''
        self.block_insert_count[ith_thread] = insert_count
        self.block_put_count[ith_thread] = put_count
        self.block_queries[ith_thread] = num_queries
        block_results[ith_thread] = block_x_m
        return

    def perturb(self, x_nat, y, sesses, ibatch):

        sess = sesses[0]

        self.query = [0, 0]
        self.insert_count = 0
        self.put_count = 0
        self.success = False
        x_adv = np.copy(x_nat)
        x_m = np.clip(x_nat - params.eps, 0, 255)
        x_p = np.clip(x_nat + params.eps, 0, 255)

        block_size = params.block_size
        _, xt, yt, zt = x_nat.shape

        assert (xt%block_size==0 and yt%block_size==0)

        blocks = self.block_partition((xt, yt, zt))

        assert (len(blocks)%params.gpus == 0)

        # admm variables
        yk_li = []
        for block in blocks:
            yk_li.append(np.zeros(len(block)))
        rho = params.admm_rho
        tau = params.admm_tau

        img_indices = [(xi,yi,zi) for xi in range(xt) for yi in range(yt) for zi in range(zt)]

        iter_round = 0
        while(iter_round < params.admm_iter):
            print('{}th round...'.format(iter_round))

            indices_count = dict()
            for index in img_indices:
                indices_count[index] = 0

            x_adv_new = np.copy(x_adv)

            # parallel update
            block_results = [0 for i in range(len(blocks))]
            self.block_insert_count = [0 for i in range(len(blocks))]
            self.block_put_count = [0 for i in range(len(blocks))]
            self.block_queries = [0 for i in range(len(blocks))]

            threads = [threading.Thread(target=self.ldg_block_seg,
                                        args=(block_results, x_adv, y,
                                              sesses[i%params.gpus], self.models[i%params.gpus], i, blocks[i],
                                              x_m, x_p, yk_li[i], rho)) for i in range(len(blocks))]
            
            num_running = 0
            for i in range(len(blocks)):
                threads[i].start()
                num_running += 1
                if num_running == params.gpus:
                    for j in range(i-params.gpus+1, i+1):
                        threads[j].join()
                    if self.success:
                        img_batch = np.concatenate(block_results[i-params.gpus+1:i+1], axis=0)
                        y_batch = np.tile(y, len(img_batch))
                        
                        self.query[0] += params.gpus
                        self.query[1] += params.gpus
                        correct_prediction = sess.run(self.model.correct_prediction,
                                                              feed_dict={self.model.x_input: img_batch,
                                                                         self.model.y_input: y_batch})
                        success = np.array([idx for idx in range(len(correct_prediction))])[np.invert(correct_prediction)]
                        for k in success:
                            print("attack success!")
                            self.insert_count += sum(self.block_insert_count)
                            self.put_count += sum(self.block_put_count)
                            self.query[0] += sum(self.block_queries)
                            self.query[1] += max(self.block_queries)
                            print("num of re-inserted pixels:", self.insert_count)
                            print("num of perturbed pixels:", self.put_count)
                            print("num of queries:", self.query)

                            self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
                            self.queries.append(self.query)
                            self.block_success_stat[iter_round] += 1
                            return np.reshape(img_batch[k], (1, *img_batch[k].shape))
                    
                    num_running = 0

            # update x_adv_new
            for i in range(len(blocks)):
                block = blocks[i]
                block_result = block_results[i]
                for index in block:
                    xi, yi, zi = index
                    val = block_result[0, xi, yi, zi]
                    x_adv_new[0, xi, yi, zi] = (x_adv_new[0, xi, yi, zi] * indices_count[index] + val) \
                        / (indices_count[index]+1)
                    indices_count[index] += 1
            
            pixel_change = np.count_nonzero(x_adv_new - x_adv)
            #unique, counts = np.unique(np.abs(x_adv-x_adv_new), return_counts=True)
            #unique_counts = np.asarray((unique, counts)).T
            #unique, counts = np.unique(np.abs(x_nat-x_adv_new), return_counts=True)
            #unique_counts_nat = np.asarray((unique, counts)).T
            print("changed pixels:", pixel_change)
            #print("unique count:", unique_counts)
            #print("unique count from nat:", unique_counts_nat)
            self.insert_count += sum(self.block_insert_count)
            self.put_count += sum(self.block_put_count)
            self.query[0] += sum(self.block_queries)
            round_query = 0
            for i in range(len(blocks)//params.gpus):
                round_query += np.mean(self.block_queries[i*params.gpus:(i+1)*params.gpus])
            self.query[1] += round_query
            #self.query[1] += max(self.block_queries)
            print("round re-inserts:", sum(self.block_insert_count))
            print("round perturbs:", sum(self.block_put_count))
            print("round queries:", sum(self.block_queries))
            print("round queries(parallel):", round_query)

            #global variable(z) update
            x_adv = np.copy(x_adv_new)

            #admm update(yk, rho)
            if params.block_scheme == 'admm':

                for i in range(len(yk_li)):
                    block = blocks[i]
                    block_result = block_results[i]
                    block_dist = []
                    
                    for index in block:
                        xi, yi, zi = index
                        block_dist.append(block_result[0, xi, yi, zi] - x_adv[0, xi, yi, zi])
                    
                    block_dist = np.array(block_dist)

                    yk_li[i] += rho * (block_dist)

                rho *= tau


            if params.early_stop == 'y':
            
                num_correct = sess.run(self.model.num_correct,
                                              feed_dict={self.model.x_input: x_adv,
                                                         self.model.y_input: y})

                self.query[0] += 1
                self.query[1] += 1

                assert np.amax(np.abs(x_adv-x_nat)) < params.eps+0.0001
                if num_correct == 0:
                    print("attack success!")
                    print("num of re-inserted pixels:", self.insert_count)
                    print("num of perturbed pixels:", self.put_count)
                    print("num of queries:", self.query)
                    self.success = True

                    self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
                    self.queries.append(self.query)
                    self.block_success_stat[iter_round] += 1
                    return x_adv

            #termination condition
            if pixel_change == 0:
                self.admm_converge_stat[iter_round] += 1
                break
            
            # stop if # of blocks == 1
            if len(blocks) == 1:
                break
            
            iter_round += 1


        
        print("attack failed")
        print("num of re-inserted pixels:", self.insert_count)
        print("num of perturbed pixels:", self.put_count)
        print("num of queries:", self.query)
        self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
        return x_adv
    
# evaluate perturbed images
def run_attack(x_adv, model, sess, x_full_batch, y_full_batch, percentage_mean):

    num_eval_samples = x_adv.shape[0]
    eval_batch_size = min(num_eval_samples, 64)

    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_corr = 0

    x_nat = x_full_batch
    l_inf = np.amax(np.abs(x_nat-x_adv))

    # error checking
    if l_inf > params.eps + 0.0001:
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

        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_batch}
        cur_corr, y_pred_batch, correct_prediction = \
            sess.run([model.num_correct, model.predictions, model.correct_prediction],
                                          feed_dict=dict_adv)
        total_corr += cur_corr
        y_pred.append(y_pred_batch)
        success.append(np.array(np.nonzero(np.invert(correct_prediction)))+ibatch*eval_batch_size)

    success = np.concatenate(success, axis=1)
    np.save('out/parallel_admm_success.npy', success)

    accuracy = total_corr / num_eval_examples
    print('adv Accuracy: {:.2f}%'.format(100.0 * accuracy))
    with open('out/result.txt', 'a') as f:
        f.write('''Resnet, {}, eps:{},
                        sample_size:{}, loss_func:{}
                        => acc:{}, percentage:{}\n'''.format(params.eps,
                                                             params.attack_type,
                                                             params.sample_size,
                                                             params.loss_func,
                                                             accuracy,
                                                             percentage_mean))


    total_corr = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_full_batch[bstart:bend,:]
        y_batch = y_full_batch[bstart:bend]

        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_batch}
        cur_corr, y_pred_batch = sess.run([model.num_correct, model.predictions],
                                          feed_dict=dict_adv)
        total_corr += cur_corr

    accuracy = total_corr / num_eval_examples
    print('nat Accuracy: {:.2f}%'.format(100.0 * accuracy))
    y_pred = np.concatenate(y_pred, axis=0)
    np.save('pred.npy', y_pred)
    print('Output saved at pred.npy')

if __name__ == '__main__':
    import json
    import sys

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint('models/'+params.model_dir)
    if model_file is None:
        print('No model found')
        sys.exit()

    models = []
    graphes = []
    sesses = []
    Config = tf.ConfigProto()
    Config.gpu_options.allow_growth = True

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for i in range(params.gpus):
            graph = tf.Graph()
            with graph.as_default():
                with tf.device('/gpu:'+str(i)):
                    model = Model(mode='eval')
                    models.append(model)
                sess = tf.Session(config=Config)
                sesses.append(sess)
            graphes.append(graph)

    # main model and session
    model = models[0]
    sess = sesses[0]

    for i in range(params.gpus):
        with graphes[i].as_default():
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sesses[i], model_file)

    attack = LazyGreedyAttack(models,
                              params.eps,
                              params.loss_func)

    data_path = config['data_path']
    cifar = cifar10_input.CIFAR10Data(data_path)

    # Iterate over the samples batch-by-batch
    num_eval_examples = params.sample_size
    eval_batch_size = 1
    #num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    x_adv = [] # adv accumulator

    bstart = 0
    while(True):
        x_candid = cifar.eval_data.xs[bstart:bstart+100]
        y_candid = cifar.eval_data.ys[bstart:bstart+100]
        mask = sess.run(model.correct_prediction, feed_dict = {model.x_input: x_candid,
                                                               model.y_input: y_candid})
        x_masked = x_candid[mask]
        y_masked = y_candid[mask]
        if bstart == 0:
            x_full_batch = x_masked[:min(num_eval_examples, len(x_masked))]
            y_full_batch = y_masked[:min(num_eval_examples, len(y_masked))]
        else:
            index = min(num_eval_examples-len(x_full_batch), len(x_masked))
            x_full_batch = np.concatenate((x_full_batch, x_masked[:index]))
            y_full_batch = np.concatenate((y_full_batch, y_masked[:index]))
        bstart += 100
        if len(x_full_batch) >= num_eval_examples:
            break

    percentages = []
    times = []
    total_times = []
    print('Iterating over {} batches\n'.format(num_batches))
    for ibatch in range(num_batches):
        print('attacking {}th image...'.format(ibatch))
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)
        x_batch = x_full_batch[bstart:bend, :]
        y_batch = y_full_batch[bstart:bend]

        start = time.time()

        fig = np.reshape(x_batch, (32, 32, 3)).astype(np.uint8).squeeze()
        pic = Image.fromarray(fig)
        pic.save("out/images/"+str(ibatch)+"nat.png")
        
        x_batch_adv = attack.perturb(x_batch, y_batch, sesses, ibatch)
        
        fig = np.reshape(x_batch_adv, (32, 32, 3)).astype(np.uint8).squeeze()
        pic = Image.fromarray(fig)
        pic.save("out/images/"+str(ibatch)+"adv.png")
        
        x_adv.append(x_batch_adv)
        percentage_mean = np.mean(attack.ratios)
        print('perturb / (perturb + re-insert):{}'.format(percentage_mean))
        end = time.time()
        print('attack time taken:{}'.format(end - start))
        total_times.append(end-start)
        if attack.success:
            times.append(end-start)
        print()

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))
    print('Average queries:', np.mean(attack.queries, axis=0))
    print('average time taken:', np.array([np.mean(total_times), np.mean(times)]))

    if not all(x == 0 for x in attack.block_success_stat.values()):
        print('success round count:', attack.block_success_stat)
    for key, val in vars(params).items():
        print('{}={}'.format(key,val))

    if params.test == 'y':
        # error checking
        if np.amax(x_adv) > 255.0001 or \
                np.amin(x_adv) < -0.0001 or \
                np.isnan(np.amax(x_adv)):
            print('Invalid pixel range. Expected [0,1], fount [{},{}]'.format(np.amin(x_adv),
                                                                              np.amax(x_adv)))
        else:
            run_attack(x_adv, model, sess, x_full_batch, y_full_batch, percentage_mean)
