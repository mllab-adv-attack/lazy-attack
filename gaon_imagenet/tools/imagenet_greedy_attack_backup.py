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
import random
import cv2 as cv

#from PIL import Image

from helper import Greedy, ExpandedGreedy, PeekablePriorityQueue

import matplotlib.pyplot as plt
plt.switch_backend('agg')

#from tools.logging_utils import log_output, render_frame
from tools.inception_v3_imagenet import model
#from tools.imagenet_labels import label_to_name
from tools.utils import get_image

IMAGENET_PATH = './data/'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    #basic
    parser.add_argument('--eps', default='0.05', help='Attack eps', type=float)
    parser.add_argument('--sample_size', default=1000, help='sample size', type=int)
    parser.add_argument('--batch_size', default=1, help='first pass batch size', type=int)
    parser.add_argument('--loss_func', default='xent', help='loss func', type=str)
    parser.add_argument('--test', default='y', help='include run attack', type=str)
    parser.add_argument('--model_dir', default='nat', help='model name', type=str)
    parser.add_argument('--early_stop', default='y', help='attack all pixels', type=str)
    parser.add_argument('--attack_type', default='ldg_dec_v2', help='attack type', type=str)
    parser.add_argument('--plot_f', default='n', help='plot F(s)', type=str)
    parser.add_argument('--plot_interval', default=1000, help ='plot interval', type=int)
    parser.add_argument('--max_q', default = 10000, help = 'max queries', type=int)
    # NES grad estimation
    parser.add_argument('--grad_est', default=0, help ='NES grad estimation sampling size', type=int)
    # resizing
    parser.add_argument('--resize', default=64, help = 'resize ratio', type=int)
    parser.add_argument('--interpolation', default='cv.INTER_NEAREST', type=str)
    # block partition
    parser.add_argument('--block_size', default=16, help = 'block partition size', type=int)
    parser.add_argument('--admm_iter', default=20, help = 'admm max iteration', type=int)
    parser.add_argument('--block_scheme', default='admm', help = 'block partition scheme', type=str)
    parser.add_argument('--overlap', default=1, help = 'overlap size', type=int)
    parser.add_argument('--admm_rho', default=1e-7, help = 'admm rho', type=float)
    parser.add_argument('--admm_tau', default=8, help ='admm tau', type=float)
    # mask out
    parser.add_argument('--top_k', default = 5, help = 'mask-out top k', type=int)
    parser.add_argument('--max_boxes', default=100, help = 'max boxes', type=int)
    parser.add_argument('--min_size', default = 8, help = 'min size of box', type=float)
    # dec method
    parser.add_argument('--dec_scale', default=0.5, help = 'ratio of decreasing size in dec', type=float)
    parser.add_argument('--dec_keep', default=0.5, help = 'ratio of selection in dec', type=float)
    parser.add_argument('--dec_select', default='margin', help = 'block selection scheme', type=str)
    parser.add_argument('--dec_v2_reset', default='y', help = 'reset x_m and x_p', type=str)
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

def pixel_checker(x_nat, x_adv):
    
    bw_combs = {}
    rgb_combs = {}
    
    for bw in [1, -1]:
        bw_combs[(bw)] = 0

    for r in [1, -1]:
        for g in [1, -1]:
            for b in [1, -1]:
                rgb_combs[(r, g, b)] = 0

    n, h, w, c = x_adv.shape
    image = x_adv - x_nat
    for ni in range(n):
        for hi in range(h):
            for wi in range(c):
                r, g, b = np.sign(image[ni, hi, wi, :]).astype(np.int32)
                
                if r==0:
                    if x_adv[ni, hi, wi, 0]==0:
                        r = -1
                    else:
                        r = 1
                if g==0:
                    if x_adv[ni, hi, wi, 1]==0:
                        g = -1
                    else:
                        g = 1
                if b==0:
                    if x_adv[ni, hi, wi, 2]==0:
                        b = -1
                    else:
                        b = 1

                rgb_combs[(r, g, b)] += 1
                bw_combs[(r)] += 1
                bw_combs[(g)] += 1
                bw_combs[(b)] += 1
    
    print('rgb combinations:', rgb_combs)
    print('bw combinations:', bw_combs)


class LazyGreedyAttack:
    def __init__(self, sess, model, epsilon, loss_func):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
             point."""
        
        self.model = model
        self.epsilon = epsilon
        self.loss_func = loss_func
        self.queries = []
        self.query = 0
        self.insert_count = 0
        self.put_count = 0
        self.ratios = []
        self.block_success_stat = {}
        self.admm_converge_stat = {}
        self.dec_block = []
        self.success = False
        self.query_exceed = False
        # mask
        self.avg_block_size = []
        for i in range(params.admm_iter):
            self.block_success_stat[i] = 0
            self.admm_converge_stat[i] = 0
        
        self.model_x = tf.placeholder(tf.float32, (None, 299, 299, 3))
        self.model_y = tf.placeholder(tf.int64, None)
        
        self.logits, self.predictions = model(sess, self.model_x, params.model_dir)
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.logits, labels = self.model_y)
        self.correct_prediction = tf.equal(self.predictions, self.model_y)
        self.num_correct = tf.reduce_sum(
            tf.cast(self.correct_prediction, tf.int32))
        self.loss = - y_xent

    # update loss gain for candid in lazy greedy attack
    def update(self, greedy, adv_image, y, sess):
        xi, yi, zi = greedy.loc
        img_batch = np.tile(adv_image, (3, 1, 1, 1))
        label_batch = np.tile(y, 3)
        img_batch[1,xi,yi,zi] += params.eps
        img_batch[2,xi,yi,zi] -= params.eps
        
        feed_dict = {
            self.model_x:np.clip(img_batch,0, 1),
            self.model_y: label_batch}
        loss, num_correct = sess.run([self.loss,
                                      self.num_correct],
                                     feed_dict=feed_dict)
        greedy.update(loss[1]-loss[0], loss[2]-loss[0])
        if num_correct == 0:
            print('attack success!')
            if params.plot_f == 'y':
                return None, loss[0]
            else:
                return None
        if params.plot_f == 'y':
            return greedy, loss[0]
        else:
            return greedy

    # NES gradient estimate
    def grad_est(self, x_adv, y, sess):
        sigma = 0.25
        _, xt, yt, zt = x_adv.shape
        noise_pos = np.random.normal(size=(params.grad_est//2, xt, yt, zt))
        noise = np.concatenate([noise_pos, -noise_pos], axis=0)
        eval_points = x_adv + sigma * noise
        labels = np.tile(y, (len(eval_points)))
        losses = sess.run([self.loss],
                          feed_dict={self.model_x: eval_points,
                                     self.model_y: labels})
        losses_tiled = np.tile(np.reshape(losses, (-1, 1, 1, 1)), x_adv.shape)
        grad_estimates = np.mean(losses_tiled * noise, axis=0)/sigma
        return 2 * params.eps * grad_estimates

    # block partition algorithm
    def block_partition(self, shape):
        
        block_size = params.block_size
        overlap = params.overlap
        xt, yt, zt = shape
        num_block_rows = xt//block_size
        num_block_cols = yt//block_size

        assert(xt%block_size==0 and yt%block_size==0)

        blocks = []
        for block_xi in range(num_block_rows):
            for block_yi in range(num_block_cols):
                block = [(block_size*block_xi+xi, block_size*block_yi+yi, zi) \
                         for zi in range(zt) \
                         for yi in range(-overlap, block_size+overlap) \
                         for xi in range(-overlap, block_size+overlap)]
                block = [index for index in block if (max(index) <= 31 and min(index) >= 0)]
                blocks.append(block)

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
    def ldg_block_seg(self, x_adv, y, sess, ibatch, block, x_m, x_p, yk=0, rho=0, Resize = params.resize):
        insert_count = 0
        put_count = 0
        queue = PeekablePriorityQueue()
        _, xt, yt, zt = x_adv.shape
        resize = Resize
        block = sorted(block)

        #for ldg_dec
        self.dec_block = []
        new_block_rank = []
        margin_queue = PeekablePriorityQueue()

        # select anchor pixels
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

        cur_m = sess.run(self.loss, feed_dict={self.model_x:block_x_m,
                                               self.model_y:y})
        
        cur_p = sess.run(self.loss, feed_dict={self.model_x:block_x_p,
                                               self.model_y:y})

        if params.attack_type == 'ldg_block' and params.block_scheme == 'admm':
            cur_m += self.admm_loss(block, block_x_m, x_adv, yk, rho)
            cur_p += self.admm_loss(block, block_x_p, x_adv, yk, rho)
        
        if params.grad_est > 0:
            #start = time.time()
            print('admm + NES not implemented yet!!!!!!!')
            m_est = self.grad_est(block_x_m, y, sess)
            p_est = self.grad_est(block_x_p, y, sess)
            for xi in range(xt):
                for yi in range(yt):
                    for zi in range(zt):
                        if (xi, yi, zi) in block:
                            queue.put(Greedy([xi, yi, zi], m_est[xi, yi, zi], -p_est[xi, yi, zi], False))
            num_queries = 2 * params.grad_est
            #end = time.time()
            #print('first pass time:', end-start)
        else:
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
                    self.model_x: img_batch,
                    self.model_y: label_batch}
                losses = sess.run(self.loss,
                                  feed_dict=feed_dict)
                for pos in range(losses.size//2):
                    xb, yb, zb = anchor_block[block_index2]
                    block_index2 += 1
                    
                    if params.attack_type == 'ldg_block' and params.block_scheme == 'admm':
                        losses[pos] += self.admm_loss(block, img_batch[pos], x_adv, yk, rho)
                        losses[batch_size+pos] += self.admm_loss(block, img_batch[batch_size+pos], x_adv, yk, rho)
                    
                    pi = losses[pos] - cur_m
                    mi = losses[batch_size+pos] - cur_p

                    queue.put(Greedy((xb,yb,zb), pi, mi, False))
            num_queries = 2 * num_pixels

            #end = time.time()
            #print('first pass time:', end-start)

        # second pass
        while not queue.empty():
            candid = queue.get()
            second = None

            if not queue.empty():
                second = queue.peek()

            # update candid
            xi, yi, zi = candid.loc
            img_batch = np.concatenate([block_x_m, block_x_p])
            for xxi in range(resize):
                for yyi in range(resize):
                    if (xi+xxi, yi+yyi, zi) in block:
                        img_batch[0, xi+xxi, yi+yyi, zi] = block_x_p[0, xi+xxi, yi+yyi, zi]
                        img_batch[1, xi+xxi, yi+yyi, zi] = block_x_m[0, xi+xxi, yi+yyi, zi]
            y_batch = np.tile(y, 2)
            
            losses, correct_prediction = sess.run([self.loss, self.correct_prediction],
                                                  feed_dict={self.model_x: img_batch,
                                                             self.model_y: y_batch})
            num_queries += 2
            success = np.array([0, 1])[np.invert(correct_prediction)]
            for i in success:
                self.insert_count += insert_count
                self.put_count += put_count
                self.query += num_queries
                self.success = True
                return np.reshape(img_batch[i], (1, *img_batch[i].shape))

            if params.attack_type=='ldg_dec' and params.max_q and (num_queries + self.query) >= params.max_q:
                self.insert_count += insert_count
                self.put_count += put_count
                self.query += num_queries
                self.query_exceed = True
                return block_x_m


            if params.attack_type == 'ldg_block' and params.block_scheme == 'admm':
                losses[0] += self.admm_loss(block, img_batch[0], x_adv, yk, rho)
                losses[1] += self.admm_loss(block, img_batch[1], x_adv, yk, rho)

            candid.update(losses[0]-cur_m, losses[1]-cur_p)

            if not second or candid <= second:
                put_count += 1
                xi, yi, zi = candid.loc
                margin_queue.put(candid)
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
                #dec
                if params.dec_select == 'rank':
                    if put_count <= num_pixels*params.dec_keep:
                        new_block_rank.append((xi, yi, zi))
                elif params.dec_select == 'reverse':
                    if put_count >= num_pixels*params.dec_keep:
                        new_block_rank.append((xi, yi, zi))

                
            else:
                insert_count+=1
                queue.put(candid)
        #print("num of re-inserted pixels:", insert_count)
        #print("num of pertubed pixels:", put_count)
        #print("num of queries:", num_queries)
        self.insert_count += insert_count
        self.put_count += put_count
        self.query += num_queries

        #print(losses[0])

        # dec - block selection
        if params.attack_type == 'ldg_dec':
            
            if params.dec_select == 'rank' or params.dec_select == 'reverse':
                new_dec_block = new_block_rank
            
            elif params.dec_select == 'margin':
                new_dec_block = []
                for i in range(int(num_pixels * params.dec_keep)):
                    new_dec_block.append(margin_queue.get().loc)
            
            else:
                print('block selection method not implemented')
                raise Exception

            for index in new_dec_block:
                xi, yi, zi = index
                for xxi in range(resize):
                    for yyi in range(resize):
                            if (xi+xxi, yi+yyi, zi) in block:
                                self.dec_block.append((xi+xxi, yi+yyi, zi))
            
        assert np.amax(np.abs(block_x_p-block_x_m)) < 0.0001
        assert np.amax(np.abs(block_x_m-x_adv)) < 2 * params.eps + 0.0001
        return block_x_m
    
    # choose attack type
    def perturb(self, x_nat, y, sess, ibatch):

        if params.attack_type == 'lg':
            print('performing vanilla lazy greedy attack')
            return self.perturb_lg(x_nat, y, sess, ibatch)
        elif params.attack_type == 'ldg':
            print('performing lazy double greedy attack w/ resizing')
            return self.perturb_ldg(x_nat, y, sess, ibatch)
        elif params.attack_type == 'ldg_block':
            print('performing lazy double greedy attack w/ block partition')
            return self.perturb_ldg_block(x_nat, y, sess, ibatch)
        elif params.attack_type == 'one':
            print('performing lazy double greedy attack only-first-pass version')
            return self.perturb_one(x_nat, y, sess, ibatch)
        elif params.attack_type == 'ldg_v2':
            print('performing lazy double greedy attack w/ 0 term')
            return self.perturb_ldg_v2(x_nat, y, sess, ibatch)
        elif params.attack_type == 'lrg' :
            print('performing lazy random attack')
            return self.perturb_lrg(x_nat, y, sess, ibatch)
        elif params.attack_type == 'ldg_mask':
            print('performing lazy double greedy attck w/ mask-out')
            return self.perturb_ldg_mask(x_nat, y, sess, ibatch)
        elif params.attack_type == 'ldg_dec':
            print('performing lazy double greedy attack w/ decreasing size')
            return self.perturb_ldg_dec(x_nat, y, sess, ibatch)
        elif params.attack_type == 'ldg_dec_v2':
            print('performing lazy double greedy attack w/ decreasing size - ver.2')
            return self.perturb_ldg_dec_v2(x_nat, y, sess, ibatch)
        else:
            raise Exception("not valid attack_type")


    # first-pass-only version
    def perturb_one(self, x_nat, y, sess, ibatch):

        loss_gains = list()
        _, xt, yt, zt = x_nat.shape
        cur = sess.run(self.loss, feed_dict={self.model_x:x_nat,
                                             self.model_y:y})
        x_adv = np.copy(x_nat)
        for xi in range(xt):
            img_batch = np.tile(x_nat, (yt*zt*2, 1, 1, 1))
            label_batch = np.tile(y, (yt*zt*2))
            for yi in range(yt):
                for zi in range(zt):
                    img_batch[2*zt*yi+2*zi, xi, yi, zi] -= params.eps
                    img_batch[2*zt*yi+2*zi+1, xi, yi, zi] += params.eps
            feed_dict = {
                self.model_x: np.clip(img_batch, 0, 1),
                self.model_y: label_batch}
            xents = sess.run(self.loss,
                             feed_dict=feed_dict)
            for pos in range(xents.size//2):
                h = pos//zt
                c = pos%zt
                mi = xents[2*pos] - cur
                pi = xents[2*pos+1] - cur
                if pi <= mi:
                    x_adv[0,xi,h,c] += params.eps
                else:
                    x_adv[0,xi,h,c] -= params.eps
                    x_adv = np.clip(x_adv, 0, 1)
                loss_gains.append(max(pi, mi))
        plot(loss_gains, ibatch)
        self.ratios.append(0)
        return np.clip(x_adv, 0, 1)

    # basic lazy greedy
    def perturb_lg(self, x_nat, y, sess, ibatch):

        queue = PeekablePriorityQueue()
        _, xt, yt, zt = x_nat.shape
        cur = sess.run(self.loss, feed_dict={self.model_x:x_nat,
                                             self.model_y:y})

        loss_gains = list()
        for xi in range(xt):
            img_batch = np.tile(x_nat, (yt*zt*2, 1, 1, 1))
            label_batch = np.tile(y, (yt*zt*2))
            for yi in range(yt):
                for zi in range(zt):
                    img_batch[2*zt*yi+2*zi, xi, yi, zi] -= params.eps
                    img_batch[2*zt*yi+2*zi+1, xi, yi, zi] += params.eps
            feed_dict = {
                self.model_x: np.clip(img_batch, 0, 1),
                self.model_y: label_batch}
            xents = sess.run(self.loss,
                             feed_dict=feed_dict)
            for pos in range(xents.size//2):
                h = pos//zt
                c = pos%zt
                mi = xents[2*pos] - cur
                pi = xents[2*pos+1] - cur
                queue.put(Greedy([xi,h,c], pi, mi, False))

        adv_image = np.copy(x_nat)
        insert_count = 0
        put_count = 0
        while not queue.empty():
            candid = queue.get()
            second = None

            if not queue.empty():
                second = queue.peek()

            if params.plot_f == 'y':
                candid, loss = self.update(candid, adv_image, y, sess)
            else:
                candid = self.update(candid, adv_image, y, sess)
            if not candid:
                break 
            if not second or candid <= second:
                if params.early_stop == 'y' or candid.getVal() >= 0:
                    put_count += 1
                    noise = np.zeros_like(adv_image)
                    xi, yi, zi = candid.loc
                    noise[0][xi][yi][zi] += self.epsilon
                    if candid.getDir():
                        adv_image += noise
                        adv_image = np.clip(adv_image, 0, 1)
                    else:
                        adv_image -= noise
                        adv_image = np.clip(adv_image, 0, 1)
                    if params.plot_f == 'y':
                        loss_gains.append(loss)
                    else:
                        loss_gains.append(candid.getVal())
                else:
                    break
            else:
                insert_count+=1
                queue.put(candid)
        plot(loss_gains, ibatch)
        self.ratios.append((put_count + 0.0001) / (insert_count + put_count+0.0001))
        print("num of re-inserted pixels:", insert_count)
        print("num of pertubed pixels:", put_count)
        return adv_image

    # lazy double greedy ({-1, 1})
    def perturb_ldg(self, x_nat, y, sess, ibatch):
        self.query = 0
        self.insert_count = 0
        self.put_count = 0
        self.success = False
        _, xt, yt, zt = x_nat.shape

        x_m = np.clip(x_nat - params.eps, 0, 1)
        x_p = np.clip(x_nat + params.eps, 0, 1)

        whole_block = [(xi, yi, zi) for xi in range(xt) for yi in range(yt) for zi in range(zt)]
        
        x_adv = self.ldg_block_seg(x_nat, y, sess, ibatch, whole_block, x_m, x_p)

        if self.success:
            print("attack success!")
            self.queries.append(self.query)
        else:
            print("attack failed")
        assert np.amax(np.abs(x_adv-x_nat)) < params.eps + 0.0001
        print("num of re-inserted pixels:", self.insert_count)
        print("num of perturbed pixels:", self.put_count)
        print("num of queries:", self.query)
        self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
        return x_adv

    # lazy double greedy with mask-out
    def perturb_ldg_mask(self, x_nat, y, sess, ibatch):
        self.query = 0
        self.insert_count = 0
        self.put_count = 0
        _, xt, yt, zt = x_nat.shape
        
        # mask-out
        edge_model = "model.yml.gz"
        edge_detection = cv.ximgproc.createStructuredEdgeDetection(edge_model)
        img = np.float32(np.reshape(x_nat,(xt, yt, zt)))/1.0
        edges = edge_detection.detectEdges(img)

        orimap = edge_detection.computeOrientation(edges)
        edges = edge_detection.edgesNms(edges, orimap)

        maxboxes = params.max_boxes
        min_size = int(params.min_size * params.min_size)
        minscore = 0.01
        for i in range(100):
            edge_boxes = cv.ximgproc.createEdgeBoxes(minBoxArea=min_size, minScore=minscore)
            edge_boxes.setMaxBoxes(maxboxes)
            boxes = edge_boxes.getBoundingBoxes(edges, orimap)
            if len(boxes) >= maxboxes :
                break
            minscore -= 0.01
        
        if len(boxes) == 0:
            boxes = np.array([[0, 0, xt, yt]])

        print('num of mask:',len(boxes))
        x_adv = np.copy(x_nat)
        batch_size = len(boxes)
        img_batch = np.tile(x_adv, (batch_size, 1, 1, 1))
        label_batch = np.tile(y, (batch_size))
        for i in range(batch_size):
            xi, yi, w, h = boxes[i]
            img_batch[i, xi:xi+w, yi:yi+h, :] = np.ones_like(img_batch[i, xi:xi+w, yi:yi+h, :]) * 128
        feed_dict = {
            self.model_x: img_batch,
            self.model_y: label_batch}
        losses = sess.run(self.loss,
                          feed_dict=feed_dict)
        k = min(params.top_k, len(boxes))
        min_indices = np.argsort(losses)[:k]
        print('pick ' ,min_indices)
        self.query += len(boxes)
        
        block = []
        for index in min_indices:
            xi, yi, w, h = boxes[index]
            block += [(xi+xxi, yi+yyi, zi) for xxi in range(w) for yyi in range(h) for zi in range(zt)]
        block = list(set(block))
        print('block size:', len(block))
        self.avg_block_size.append(len(block))

        # ldg on mask
        x_m = np.clip(x_nat - params.eps, 0, 1)
        x_p = np.clip(x_nat + params.eps, 0, 1)
        x_adv = self.ldg_block_seg(x_nat, y, sess, ibatch, block, x_m, x_p)
        
        if self.success:
            print("attack success!")
            self.queries.append(self.query)
        else:
            print("attack failed")
        assert np.amax(np.abs(x_adv-x_nat)) < params.eps + 0.0001
        print("num of re-inserted pixels:", self.insert_count)
        print("num of perturbed pixels:", self.put_count)
        print("num of queries:", self.query)
        self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
        return x_adv
        
    # lazy double greedy with decreasing size
    def perturb_ldg_dec(self, x_nat, y, sess, ibatch):
        self.query = 0
        self.insert_count = 0
        self.put_count = 0
        self.query_exceed = False
        self.success = False
        _, xt, yt, zt = x_nat.shape
        resize = params.resize

        x_m = np.clip(x_nat - params.eps, 0, 1)
        x_p = np.clip(x_nat + params.eps, 0, 1)

        self.dec_block = [(xi, yi, zi) for xi in range(xt) for yi in range(yt) for zi in range(zt)]

        x_adv = np.copy(x_nat)

        rounds = 0
        while(resize>=1 and len(self.dec_block) > 0 and not self.success and not self.query_exceed):
            print('round:', rounds)
            x_adv = self.ldg_block_seg(x_adv, y, sess, ibatch, self.dec_block, x_m, x_p, Resize=resize)
            resize = int(resize * params.dec_scale)
            rounds+=1
        
        if self.success:
            print("attack success!")
            self.queries.append(self.query)
        else:
            print("attack failed")
        assert np.amax(np.abs(x_adv-x_nat)) < params.eps + 0.0001
        print("num of re-inserted pixels:", self.insert_count)
        print("num of perturbed pixels:", self.put_count)
        print("num of queries:", self.query)
        self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
        return x_adv
    
    # lazy double greedy with decreasing size - ver2 (only one first pass)
    def perturb_ldg_dec_v2(self, x_nat, y, sess, ibatch):
        #parameter initialize
        self.query = 0
        self.insert_count = 0
        self.put_count = 0
        self.success = False
        _, xt, yt, zt = x_nat.shape
        resize = params.resize

        # empty&full set initialize
        x_m = np.clip(x_nat - params.eps, 0, 1)
        x_p = np.clip(x_nat + params.eps, 0, 1)
        
        img_batch = np.concatenate([x_m, x_p])
        label_batch = np.concatenate([y, y])
        
        feed_dict = {
            self.model_x: img_batch,
            self.model_y: label_batch}
        losses, correct_prediction = sess.run([self.loss, self.correct_prediction],
                                              feed_dict=feed_dict)
        
        self.query += 2
        success = np.array([i for i in range(2)])[np.invert(correct_prediction)]
        
        # early success
        np.random.shuffle(success)
        for i in success:
            self.success = True
            self.queries.append(self.query)
            print("attack success!")
            print("num of re-inserted pixels:", self.insert_count)
            print("num of perturbed pixels:", self.put_count)
            print("num of queries:", self.query)
            self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
            return np.reshape(img_batch[i], (1, *img_batch[i].shape))

        cur_m = losses[0]
        cur_p = losses[1]

        # spare x_m and x_p for comparison
        x_m_spare = np.copy(x_m)
        x_p_spare = np.copy(x_p)

        # create original pixel set
        orig_block = [(xi, yi, zi) for xi in range(xt) for yi in range(yt) for zi in range(zt)]
        
        # first pass -----------------------------------------------

        queue = PeekablePriorityQueue()

        # select anchor pixels
        anchor_block = []
        selected = set()
        for index in orig_block:
            if index not in selected:
                anchor_block.append(index)
                xi, yi, zi = index
                for xxi in range(resize):
                    for yyi in range(resize):
                        selected.add((xi+xxi, yi+yyi, zi))
        anchor_block = sorted(anchor_block)
        
        num_pixels = len(anchor_block)

        orig_block_set = set(orig_block)
        #start = time.time()
        batch_size = min(params.batch_size, num_pixels)
        num_batches = num_pixels//batch_size
        block_index1, block_index2 = 0, 0
        for ith_batch in range(num_batches+1):
            if ith_batch == num_batches:
                if num_pixels%batch_size == 0:
                    break
                else:
                    batch_size = num_pixels%batch_size
            img_batch_m = np.tile(x_m, (batch_size, 1, 1, 1))
            img_batch_p = np.tile(x_p, (batch_size, 1, 1, 1))
            img_batch = np.concatenate([img_batch_m, img_batch_p])
            label_batch = np.tile(y, (2*batch_size))
            for j in range(batch_size):
                xb, yb, zb = anchor_block[block_index1]
                block_index1 += 1
                for xxi in range(resize):
                    for yyi in range(resize):
                        if (xb+xxi, yb+yyi, zi) in orig_block_set:
                            img_batch[j, xb+xxi, yb+yyi, zb] = x_p[0, xb+xxi, yb+yyi, zb]
                            img_batch[batch_size + j, xb+xxi, yb+yyi, zb] = x_m[0, xb+xxi, yb+yyi, zb]
            feed_dict = {
                self.model_x: img_batch,
                self.model_y: label_batch}
            losses, correct_prediction = sess.run([self.loss, self.correct_prediction],
                                                  feed_dict=feed_dict)
            
            self.query += 2 * batch_size
            success = np.array([i for i in range(2*batch_size)])[np.invert(correct_prediction)]
            
            # early success
            np.random.shuffle(success)
            for i in success:
                self.success = True
                self.queries.append(self.query)
                print("attack success!")
                print("num of re-inserted pixels:", self.insert_count)
                print("num of perturbed pixels:", self.put_count)
                print("num of queries:", self.query)
                self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
                return np.reshape(img_batch[i], (1, *img_batch[i].shape))
            
            for pos in range(losses.size//2):
                xb, yb, zb = anchor_block[block_index2]
                block_index2 += 1
                
                pi = losses[pos] - cur_m
                mi = losses[batch_size+pos] - cur_p

                queue.put(Greedy((xb,yb,zb), pi, mi, False))

        #end = time.time()
        #print('first pass time:', end-start)

        # iterative second passes ---------------------------------

        rounds = 0
        while(True):
            print('second pass round:', rounds)
            
            #x_adv = self.ldg_block_seg(x_adv, y, sess, ibatch, self.dec_block, x_m, x_p, Resize=resize)
            new_queue = PeekablePriorityQueue()

            while not queue.empty():
                candid = queue.get()
                second = None

                if not queue.empty():
                    second = queue.peek()

                # update candid
                xi, yi, zi = candid.loc
                img_batch = np.concatenate([x_m, x_p])
                for xxi in range(resize):
                    for yyi in range(resize):
                        if (xi+xxi, yi+yyi, zi) in orig_block_set:
                            img_batch[0, xi+xxi, yi+yyi, zi] = x_p[0, xi+xxi, yi+yyi, zi]
                            img_batch[1, xi+xxi, yi+yyi, zi] = x_m[0, xi+xxi, yi+yyi, zi]
                y_batch = np.tile(y, 2)
                
                losses, correct_prediction = sess.run([self.loss, self.correct_prediction],
                                                      feed_dict={self.model_x: img_batch,
                                                                 self.model_y: y_batch})
                self.query += 2
                success = np.array([0, 1])[np.invert(correct_prediction)]
                
                # early success
                np.random.shuffle(success)
                for i in success:
                    self.success = True
                    self.queries.append(self.query)
                    print("attack success!")
                    print("num of re-inserted pixels:", self.insert_count)
                    print("num of perturbed pixels:", self.put_count)
                    print("num of queries:", self.query)
                    self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
                    return np.reshape(img_batch[i], (1, *img_batch[i].shape))

                # max query check
                if params.max_q and self.query >= params.max_q:
                    print("attack failed")
                    print("num of re-inserted pixels:", self.insert_count)
                    print("num of perturbed pixels:", self.put_count)
                    print("num of queries:", self.query)
                    self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
                    if np.random.uniform(0, 1, 1) >= 0.5:
                        return x_m
                    else:
                        return x_p

                # update candid
                candid.update(losses[0]-cur_m, losses[1]-cur_p)

                # determine put or insert
                if not second or candid <= second:
                    self.put_count += 1
                    xi, yi, zi = candid.loc
                    new_queue.put(candid)
                    if candid.getDir():
                        for xxi in range(resize):
                            for yyi in range(resize):
                                if (xi+xxi, yi+yyi, zi) in orig_block_set:
                                    x_m[0, xi+xxi, yi+yyi, zi] = x_p[0, xi+xxi, yi+yyi, zi]
                        cur_m = losses[0]
                    else:
                        for xxi in range(resize):
                            for yyi in range(resize):
                                if (xi+xxi, yi+yyi, zi) in orig_block_set:
                                    x_p[0, xi+xxi, yi+yyi, zi] = x_m[0, xi+xxi, yi+yyi, zi]
                        cur_p = losses[1]
                else:
                    self.insert_count+=1
                    queue.put(candid)

            # termination condition
            new_resize = int(resize * params.dec_scale)
            if new_resize == 0:
                break

            # pick new anchor blocks
            new_num_pixels = 0
            for i in range(int(num_pixels * params.dec_keep)):
                candid = new_queue.get()
                xi, yi, zi = candid.loc
                margin = candid.getVal()
                anchor_pixels = []

                for xxi in range(resize):
                    for yyi in range(resize):
                        if (xi+xxi, yi+yyi, zi) in orig_block_set:
                            anchor_pixels.append((xi+xxi, yi+yyi, zi))
                            #re-set x_m, x_p
                            if params.dec_v2_reset == 'y':
                                x_m[0, xi+xxi, yi+yyi, zi] = x_m_spare[0, xi+xxi, yi+yyi, zi]
                                x_p[0, xi+xxi, yi+yyi, zi] = x_p_spare[0, xi+xxi, yi+yyi, zi]

                new_anchor_block = []
                selected = set()
                for index in sorted(anchor_pixels):
                    if index not in selected:
                        new_anchor_block.append(index)
                        new_num_pixels += 1
                        xi, yi, zi = index
                        for xxi in range(new_resize):
                            for yyi in range(new_resize):
                                selected.add((xi+xxi, yi+yyi, zi))
                new_margin = margin / len(new_anchor_block)
                for index in new_anchor_block:
                    queue.put(Greedy(index, new_margin, new_margin))
    
            #update parameters
            resize = new_resize
            num_pixels = new_num_pixels
            rounds += 1
            if new_num_pixels == 0:
                break

            if params.dec_v2_reset == 'y':
                #update cur_m, cur_p
                cur_m, num_corrects = sess.run([self.loss, self.num_correct],
                                               feed_dict={self.model_x:x_m,
                                                          self.model_y:y})
                self.query += 1
                if num_corrects == 0:
                    self.success = True
                    self.queries.append(self.query)
                    print("attack success!")
                    print("num of re-inserted pixels:", self.insert_count)
                    print("num of perturbed pixels:", self.put_count)
                    print("num of queries:", self.query)
                    self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
                    return x_m

            
                cur_p, num_corrects = sess.run([self.loss, self.num_correct],
                                               feed_dict={self.model_x:x_p,
                                                          self.model_y:y})
                self.query += 1
                if num_corrects == 0:
                    self.success = True
                    self.queries.append(self.query)
                    print("attack success!")
                    print("num of re-inserted pixels:", self.insert_count)
                    print("num of perturbed pixels:", self.put_count)
                    print("num of queries:", self.query)
                    self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
                    return x_p
        
        print("attack failed")
        assert np.amax(np.abs(x_m-x_p)) < 0.0001
        assert np.amax(np.abs(x_m-x_nat)) < params.eps + 0.0001
        print("num of re-inserted pixels:", self.insert_count)
        print("num of perturbed pixels:", self.put_count)
        print("num of queries:", self.query)
        self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
        if np.random.uniform(0, 1, 1) >= 0.5:
            return x_m
        else:
            return x_p
    
    # lazy double greedy ({-1, 1}) with block partitions
    def perturb_ldg_block(self, x_nat, y, sess, ibatch):
        self.query = 0
        self.insert_count = 0
        self.put_count = 0
        x_adv = np.copy(x_nat)
        x_m = np.clip(x_nat - params.eps, 0, 1)
        x_p = np.clip(x_nat + params.eps, 0, 1)

        block_size = params.block_size
        _, xt, yt, zt = x_nat.shape

        assert (xt%block_size==0 and yt%block_size==0)

        blocks = self.block_partition((xt, yt, zt))

        # admm variables
        yk_li = []
        for block in blocks:
            yk_li.append(np.zeros(len(block)))
        rho = params.admm_rho
        tau = params.admm_tau

        img_indices = [(xi,yi,zi) for xi in range(xt) for yi in range(yt) for zi in range(zt)]

        iter_round = 0
        accumul_put_count = 0
        accumul_insert_count = 0
        while(iter_round < params.admm_iter):
            print('{}th round...'.format(iter_round))
            indices_count = dict()
            for index in img_indices:
                indices_count[index] = 0

            x_adv_new = np.copy(x_adv)
            block_num = 0
            block_results = []
            for block in blocks:
                #print('attacking {}th block...'.format(block_num))
                #local variable(x) update
                block_result = self.ldg_block_seg(x_adv, y, sess, ibatch, block,
                                                  x_m, x_p, yk_li[block_num], rho)
                block_results.append(block_result)
                block_num+=1
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
            print("round re-inserts:", self.insert_count - accumul_insert_count)
            print("round perturbs:", self.put_count - accumul_put_count)
            accumul_insert_count = self.insert_count
            accumul_put_count = self.put_count

            #global variable(z) update
            x_adv = np.copy(x_adv_new)

            #admm update(yk)
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

                # admm update
                rho *= tau
                
                #admm termination condition
                if pixel_change == 0:
                    self.admm_converge_stat[iter_round] += 1
                    break

            if params.early_stop == 'y':
            
                num_correct = sess.run(self.num_correct,
                                              feed_dict={self.model_x: x_adv,
                                                         self.model_y: y})
                self.query += 1

                assert np.amax(np.abs(x_adv-x_nat)) < params.eps+0.0001
                if num_correct == 0:
                    print("attack success!")
                    print("num of re-inserted pixels:", self.insert_count)
                    print("num of perturbed pixels:", self.put_count)
                    print("num of queries:", self.query)

                    self.ratios.append((self.put_count + 0.0001) / (self.insert_count + self.put_count + 0.0001))
                    self.queries.append(self.query)
                    self.block_success_stat[iter_round] += 1
                    return x_adv

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
    
    #lazy double greedy with {-1, 0, 1} 
    def perturb_ldg_v2(self, x_nat, y, sess, ibatch):

        queue = PeekablePriorityQueue()
        _, xt, yt, zt = x_nat.shape
        x_m = np.clip(x_nat - params.eps, 0, 1)
        x_p = np.clip(x_nat + params.eps, 0, 1)
        x_n = np.copy(x_nat)
        img_batch = np.concatenate([x_m, x_p, x_n])
        y_batch = np.tile(y, (3))
        losses = sess.run(self.loss, feed_dict={self.model_x:img_batch,
                                                self.model_y:y_batch})
        cur_m = losses[0]
        cur_p = losses[1]
        cur_n = losses[2]

        loss_gains = list()

        label_batch = np.tile(y, (yt*zt*2))
        for xi in range(xt):

            img_batch_m = np.tile(x_m, (yt*zt*2, 1, 1, 1))
            for yi in range(yt):
                for zi in range(zt):
                    img_batch_m[2*zt*yi+2*zi, xi, yi, zi] = x_n[0, xi, yi, zi]
                    img_batch_m[2*zt*yi+2*zi+1, xi, yi, zi] = x_p[0, xi, yi, zi]
            feed_dict = {
                self.model_x: img_batch_m,
                self.model_y: label_batch}
            xents_m = sess.run(self.loss,
                               feed_dict=feed_dict)

            img_batch_n = np.tile(x_n, (yt*zt*2, 1, 1, 1))
            for yi in range(yt):
                for zi in range(zt):
                    img_batch_n[2*zt*yi+2*zi, xi, yi, zi] = x_m[0, xi, yi, zi]
                    img_batch_n[2*zt*yi+2*zi+1, xi, yi, zi] = x_p[0, xi, yi, zi]
            feed_dict = {
                self.model_x: img_batch_n,
                self.model_y: label_batch}
            xents_n = sess.run(self.loss,
                               feed_dict=feed_dict)

            img_batch_p = np.tile(x_p, (yt*zt*2, 1, 1, 1))
            for yi in range(yt):
                for zi in range(zt):
                    img_batch_p[2*zt*yi+2*zi, xi, yi, zi] = x_m[0, xi, yi, zi]
                    img_batch_p[2*zt*yi+2*zi+1, xi, yi, zi] = x_n[0, xi, yi, zi]
            feed_dict = {
                self.model_x: img_batch_p,
                self.model_y: label_batch}
            xents_p = sess.run(self.loss,
                               feed_dict=feed_dict)

            for pos in range(xents_n.size//2):
                h = pos//zt
                c = pos%zt
                mton = xents_m[2*pos] - cur_m
                mtop = xents_m[2*pos+1] - cur_m
                ntom = xents_n[2*pos] - cur_n
                ntop = xents_n[2*pos+1] - cur_n
                ptom = xents_p[2*pos] - cur_p
                pton = xents_p[2*pos+1] - cur_p
                queue.put(ExpandedGreedy([xi,h,c], mton, mtop, ntom, ntop, ptom, pton))

        insert_count = 0
        put_count = 0
        y_batch = np.tile(y, 9)
        while not queue.empty():
            candid = queue.get()
            second = None

            if not queue.empty():
                second = queue.peek()

            xi, yi, zi = candid.loc
            img_batch = np.concatenate([np.tile(x_m, (3,1,1,1)), np.tile(x_n, (3,1,1,1)), np.tile(x_p,(3,1,1,1))])
            img_batch[1, xi, yi, zi] = x_n[0, xi, yi, zi]
            img_batch[2, xi, yi, zi] = x_p[0, xi, yi, zi]
            img_batch[4, xi, yi, zi] = x_m[0, xi, yi, zi]
            img_batch[5, xi, yi, zi] = x_p[0, xi, yi, zi]
            img_batch[7, xi, yi, zi] = x_m[0, xi, yi, zi]
            img_batch[8, xi, yi, zi] = x_n[0, xi, yi, zi]
            losses, correct_prediction = sess.run([self.loss, self.correct_prediction],
                                                  feed_dict={self.model_x: img_batch,
                                                             self.model_y: y_batch})
            success = np.array([i for i in range(len(img_batch))])[np.invert(correct_prediction)]
            for i in success:
                print("attack success")
                print("num of re-inserted pixels:", insert_count)
                print("num of perturbed pixels:", put_count)
                self.ratios.append((put_count + 0.0001) / (insert_count + put_count+0.0001))
                plot(loss_gains, ibatch)
                return np.reshape(img_batch[i], (1,*img_batch[i].shape))

            candid.update(losses[1]-losses[0], losses[2]-losses[0],
                          losses[4]-losses[3], losses[5]-losses[3],
                          losses[7]-losses[6], losses[8]-losses[6],
                          )

            if not second or candid < second:
                put_count += 1
                xi, yi, zi = candid.loc
                direction = candid.getDir()
                if direction in (0, 5):
                    x_m[0, xi, yi, zi] = x_n[0, xi, yi, zi]
                    x_p[0, xi, yi, zi] = x_n[0, xi, yi, zi]
                elif direction in (1, 3):
                    x_m[0, xi, yi, zi] = x_p[0, xi, yi, zi]
                    x_n[0, xi, yi, zi] = x_p[0, xi, yi, zi]
                elif direction in (2, 4):
                    x_n[0, xi, yi, zi] = x_m[0, xi, yi, zi]
                    x_p[0, xi, yi, zi] = x_m[0, xi, yi, zi]
                else:
                    raise Exception("not valid direction")
                loss_gains.append(candid.getVal())
            else:
                insert_count+=1
                queue.put(candid)
        plot(loss_gains, ibatch)
        print("attack failed")
        print("num of re-inserted pixels:", insert_count)
        print("num of pertubed pixels:", put_count)
        self.ratios.append((put_count + 0.0001) / (insert_count + put_count+0.0001))

        assert np.amax(np.absolute(x_n - x_p)) < 0.001 and np.amax(np.absolute(x_n-x_m)) < 0.001

        return x_m

    #lazy random greedy
    def perturb_lrg(self, x_nat, y, sess, ibatch):

        queue = PeekablePriorityQueue()
        _, xt, yt, zt = x_nat.shape
        x_m = np.clip(x_nat - params.eps, 0, 1)
        x_p = np.clip(x_nat + params.eps, 0, 1)
        cur_m = sess.run(self.loss, feed_dict={self.model_x:x_m,
                                               self.model_y:y})
        cur_p = sess.run(self.loss, feed_dict={self.model_x:x_p,
                                               self.model_y:y})

        #start = time.time()
        loss_gains = list()
        for xi in range(xt):
            img_batch_m = np.tile(x_m, (yt*zt, 1, 1, 1))
            img_batch_p = np.tile(x_p, (yt*zt, 1, 1, 1))
            img_batch = np.concatenate((img_batch_m, img_batch_p))
            label_batch = np.tile(y, (2*yt*zt))
            for yi in range(yt):
                for zi in range(zt):
                    img_batch[zt*yi+zi, xi, yi, zi] = x_p[0, xi, yi, zi]
                    img_batch[zt*yt+zt*yi+zi, xi, yi, zi] = x_m[0, xi, yi, zi]
            feed_dict = {
                self.model_x: img_batch,
                self.model_y: label_batch}
            losses = sess.run(self.loss,
                              feed_dict=feed_dict)
            for pos in range(losses.size//2):
                h = pos//zt
                c = pos%zt
                pi = losses[pos] - cur_m
                mi = losses[yt*zt+pos] - cur_p
                queue.put(Greedy([xi,h,c], pi, mi, random=True))

        #end = time.time()

        insert_count = 0
        put_count = 0
        while not queue.empty():
            candid = queue.get()
            second = None

            if not queue.empty():
                second = queue.peek()

            xi, yi, zi = candid.loc
            img_batch = np.concatenate([np.tile(x_m, (2,1,1,1)), np.tile(x_p,(2,1,1,1))])
            img_batch[1, xi, yi, zi] = x_p[0, xi, yi, zi]
            img_batch[3, xi, yi, zi] = x_m[0, xi, yi, zi]
            y_batch = np.tile(y, 4)
            losses, correct_prediction = sess.run([self.loss, self.correct_prediction],
                                                  feed_dict={self.model_x: img_batch,
                                                             self.model_y: y_batch})
            success = np.array([0,1,2,3])[np.invert(correct_prediction)]
            for i in success:
                print("attack success")
                print("num of re-inserted pixels:", insert_count)
                print("num of perturbed pixels:", put_count)
                self.ratios.append((put_count + 0.0001) / (insert_count + put_count+0.0001))
                plot(loss_gains, ibatch)
                return np.reshape(img_batch[i], (1,*img_batch[i].shape))

            candid.update(losses[1]-losses[0], losses[3]-losses[2])

            if not second or candid < second:
                put_count += 1
                xi, yi, zi = candid.loc
                if random.uniform(0, 1) <= candid.getDir():
                    x_m[0, xi, yi, zi] = x_p[0, xi, yi, zi]
                    loss_gains.append(losses[1]-losses[0])
                else:
                    x_p[0, xi, yi, zi] = x_m[0, xi, yi, zi]
                    loss_gains.append(losses[3]-losses[2])
            else:
                insert_count+=1
                queue.put(candid)
        plot(loss_gains, ibatch)
        print("attack failed")
        print("num of re-inserted pixels:", insert_count)
        print("num of pertubed pixels:", put_count)
        self.ratios.append((put_count + 0.0001) / (insert_count + put_count+0.0001))

        assert np.amax(np.absolute(x_m - x_p)) < 0.001

        return x_m

# evaluate perturbed images
def run_attack(x_adv, sess, attack, x_full_batch, y_full_batch, percentage_mean):

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

        dict_adv = {attack.model_x: x_batch,
                    attack.model_y: y_batch}
        cur_corr, y_pred_batch, correct_prediction = \
            sess.run([attack.num_correct, attack.predictions, attack.correct_prediction],
                     feed_dict=dict_adv)
        total_corr += cur_corr
        y_pred.append(y_pred_batch)
        success.append(np.array(np.nonzero(np.invert(correct_prediction)))+ibatch*eval_batch_size)

    success = np.concatenate(success, axis=1)
    np.save('out/'+params.attack_type+'_success.npy', success)
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

        dict_adv = {attack.model_x: x_batch,
                    attack.model_y: y_batch}
        cur_corr, y_pred_batch = sess.run([attack.num_correct, attack.predictions],
                                          feed_dict=dict_adv)
        total_corr += cur_corr

    accuracy = total_corr / num_eval_examples
    print('nat Accuracy: {:.2f}%'.format(100.0 * accuracy))
    y_pred = np.concatenate(y_pred, axis=0)
    np.save('pred.npy', y_pred)
    print('Output saved at pred.npy')

if __name__ == '__main__':
    import json
    #import sys

    with open('config.json') as config_file:
        config = json.load(config_file)


    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    with tf.Session(config=configs) as sess:
        # set attack
        attack = LazyGreedyAttack(sess, model,
                                  params.eps,
                                  params.loss_func)
        # Iterate over the samples batch-by-batch
        num_eval_examples = params.sample_size
        eval_batch_size = 1
        target_indices = np.load('./imagenet_out/intersection_norm.npy')
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
                img_batch = np.reshape(img_batch, (-1, *img_batch.shape))
                x_candid.append(img_batch)
                y_candid.append(y_batch)
            x_candid = np.concatenate(x_candid, axis=0)
            y_candid = np.array(y_candid)
            logits, preds = sess.run([attack.logits, attack.predictions],
                                     feed_dict={attack.model_x: x_candid,
                                                attack.model_y: y_candid})
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
        np.save('./imagenet_out/tensorflow_{}'.format(params.sample_size), attack_set)

        percentages = []
        total_times = []
        times = []
        print('Iterating over {} batches\n'.format(num_batches))
        for ibatch in range(num_batches):
            print('attacking {}th image...'.format(ibatch))
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            x_batch = x_full_batch[bstart:bend, :]
            y_batch = y_full_batch[bstart:bend]

            start = time.time()

            '''fig = np.reshape(x_batch, (299, 299, 3)).astype(np.uint8).squeeze()
            pic = Image.fromarray(fig)
            pic.save("out/images/"+str(ibatch)+"nat.png")'''
            
            x_batch_adv = attack.perturb(x_batch, y_batch, sess, ibatch)
            
            '''fig = np.reshape(x_batch_adv, (299, 299, 3)).astype(np.uint8).squeeze()
            pic = Image.fromarray(fig)
            pic.save("out/images/"+str(ibatch)+"adv.png")'''
            
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
        if (params.attack_type == 'ldg_dec') or (params.attack_type == 'ldg_dec_v2') :
            np.save('imagenet_out/{}_{}_{}_{}_{}_{}.npy'.format(params.attack_type,
                                                                params.resize,
                                                                params.dec_select,
                                                                params.dec_scale,
                                                                params.dec_keep,
                                                                params.max_q), attack.queries)
        pixel_checker(x_full_batch, x_adv)
        print('Examples stored in {}'.format(path))
        print('Average queries: {:.2f}'.format(np.mean(attack.queries)))
        if params.attack_type == 'ldg_mask':
            print('Average block size: {:.2f}'.format(np.mean(attack.avg_block_size)))
        if not all(x == 0 for x in attack.block_success_stat.values()):
            print('success round count:', attack.block_success_stat)
        for key, val in vars(params).items():
            print('{}={}'.format(key,val))

        if params.test == 'y':
            # error checking
            if np.amax(x_adv) > 1.0001 or \
                    np.amin(x_adv) < -0.0001 or \
                    np.isnan(np.amax(x_adv)):
                print('Invalid pixel range. Expected [0,1], fount [{},{}]'.format(np.amin(x_adv),
                                                                                  np.amax(x_adv)))
            else:
                run_attack(x_adv, sess, attack, x_full_batch, y_full_batch, percentage_mean)
