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
    parser.add_argument('--eps', default='0.05', help='Attack eps', type=float)
    parser.add_argument('--sample_size', default=1000, help='sample size', type=int)
    parser.add_argument('--model_dir', default='nat', help='model name', type=str)
    parser.add_argument('--attack_type', default='bandit', help='attack type', type=str)
    parser.add_argument('--fd_eta', default=0.1, type=float)
    parser.add_argument('--max_q', default=10000, help = 'max queries', type=int)
    parser.add_argument('--image_lr', default=0.005, type=float)
    parser.add_argument('--online_lr', default=100, type=int)
    parser.add_argument('--exploration', default=0.01, type=int)
    parser.add_argument('--grad_iters', default = 1, type=int)
    parser.add_argument('--tile_size', default = 50, type=int)
    parser.add_argument('--batch_size', default = 200, type=int)
    parser.add_argument('--nes', action='store_true')
    
    params = parser.parse_args()
    for key, val in vars(params).items():
        print('{}={}'.format(key,val))


class BanditAttack:
    def __init__(self, sess, model):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
             point."""
        
        self.model = model
        self.queries = []
        self.query = 0
        self.success = False
        self.query_exceed = False
        
        self.model_x = tf.placeholder(tf.float32, (None, 299, 299, 3))
        self.model_y = tf.placeholder(tf.int64, None)
        
        self.logits, self.predictions = model(sess, self.model_x, params.model_dir)
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.logits, labels = self.model_y)
        self.correct_prediction = tf.equal(self.predictions, self.model_y)
        self.num_correct = tf.reduce_sum(
            tf.cast(self.correct_prediction, tf.int32))
        self.loss = y_xent
    
    def norm(self, t):
        norm_vec = np.reshape(np.sqrt(np.sum(np.power(t, 2), axis=(1,2,3))), (-1, 1, 1, 1))
        norm_vec += np.ones_like(norm_vec) * 1e-8
        return norm_vec

    def eq_step(self, x, q, lr):
        real_x = (x+1)/2
        pos = real_x*np.exp(lr*q)
        neg = (1-real_x)*np.exp(-lr*q)
        new_x = pos / (pos+neg)
        return new_x*2-1

    def linf_step(self, x, q, lr):
        return x + lr * np.sign(q)

    def linf_proj(self, image, x_nat, eps=params.eps):
        proj = np.clip(image, x_nat-eps, x_nat+eps)
        return proj

    # choose attack type
    def perturb(self, x_nat, y, sess, ibatch):
        batch, xt, yt, zt = x_nat.shape
        image = np.copy(x_nat)
        self.query = 0
        self.success=  False

        # Initial setup
        prior_size = params.tile_size
        total_queries = np.zeros_like(y).astype(np.float32)
        prior = np.zeros((batch, prior_size, prior_size, zt))
        dim = prior_size * prior_size * zt
        prior_step = self.eq_step
        image_step = self.linf_step
        proj = self.linf_proj

        # resizer
        resize_in = tf.placeholder(tf.float32, shape=(batch, prior_size, prior_size, zt))
        resize_out = tf.image.resize_images(resize_in, [xt, yt])

        # Loss function
        def L(img):
            loss, predictions = sess.run([self.loss, self.predictions],
                                         feed_dict={self.model_x:img,
                                                    self.model_y:y})
            return loss, predictions

        #original classifications
        #orig_images = tf.identity(image)
        orig_classes = np.copy(y)
        correct_classified_mask = np.ones_like(y, dtype=np.float32)
        #total_ims = tf.reduce_sum(correct_classified_mask)
        not_dones_mask = np.copy(correct_classified_mask)

        while not (np.amax(total_queries) > params.max_q):
            if not params.nes:
                exp_noise = params.exploration*np.random.randn(batch, prior_size, prior_size, zt)/(dim**0.5)
                q1 = sess.run(resize_out, feed_dict={resize_in:prior+exp_noise})
                q2 = sess.run(resize_out, feed_dict={resize_in:prior-exp_noise})
                l1, _ = L(image+params.fd_eta*q1/self.norm(q1))
                l2, _ = L(image+params.fd_eta*q2/self.norm(q2))
                est_deriv = (l1-l2)/(params.fd_eta*params.exploration)
                est_grad = np.reshape(est_deriv, (-1, 1, 1, 1))*exp_noise
                prior = prior_step(prior, est_grad, params.online_lr)
            else:
                prior = np.zeros_like(image)
                for _ in range(params.grad_iters):
                    exp_noise = np.random.randn(batch, xt, yt, zt)/(dim*0.5)
                    est_deriv = (L(image+params.fd_eta*exp_noise)[0] - L(image-params.fd_eta*exp_noise)[0])/params.fd_eta
                    prior += np.reshape(est_deriv, (-1, 1, 1, 1))*exp_noise

            prior = prior*np.reshape(not_dones_mask, (-1, 1, 1, 1))

            upsampler = tf.image.resize_images(prior, [xt, yt])
            upsampler = sess.run(upsampler)
            new_im = image_step(image, upsampler, params.image_lr)
            image = proj(new_im, x_nat)
            image = np.clip(image, 0, 1)

            total_queries += 2*params.grad_iters*not_dones_mask
            not_dones_mask = not_dones_mask*(L(image)[1]==orig_classes).astype(np.float32)

            #new_losses = L(image)[0]
            success_mask = (1-not_dones_mask)*correct_classified_mask
            num_success = np.sum(success_mask)
            current_success_rate = (num_success / np.sum(correct_classified_mask))
            success_queries = (np.sum(success_mask*total_queries)/num_success)
            #not_done_loss = sess.run(tf.reduce_sum(new_losses*not_dones_mask)/tf.reduce_sum(not_dones_mask))
            max_curr_queries = np.amax(total_queries)

            print("Queries: %d | Success rate: %f | Average queries: %f" %(max_curr_queries, current_success_rate, success_queries))
            if current_success_rate == 1.0:
                break
        
        self.query = success_mask*total_queries
        for query in self.query:
            self.queries.append(query)
        return image



# evaluate perturbed images
def run_attack(x_adv, sess, attack, x_full_batch, y_full_batch):

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

    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    with tf.Session(config=configs) as sess:
        # set attack
        attack = BanditAttack(sess, model)
        # Iterate over the samples batch-by-batch
        num_eval_examples = params.sample_size
        eval_batch_size = min(params.batch_size, num_eval_examples)
        assert num_eval_examples%eval_batch_size==0
        #num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
        x_adv = [] # adv accumulator

        indices = np.load('./../data/indices_untargeted.npy')

        bstart = 0
        x_full_batch = []
        y_full_batch = []
        print('loading image data')
        while(True):
            x_candid = []
            y_candid = []
            for i in range(100):
                img_batch, y_batch = get_image(indices[bstart+i], IMAGENET_PATH)
                img_batch = np.reshape(img_batch, (-1, *img_batch.shape))
                if i == 0:
                    x_candid = img_batch
                    y_candid = np.array([y_batch])
                else:
                    x_candid = np.concatenate([x_candid, img_batch])
                    y_candid = np.concatenate([y_candid, [y_batch]])
            logits, preds = sess.run([attack.logits, attack.predictions],
                                     feed_dict={attack.model_x: x_candid,
                                                attack.model_y: y_candid})
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
            end = time.time()
            print('attack time taken:{}'.format(end - start))
            total_times.append(end-start)
            if attack.success:
                times.append(end-start)
            print()

        print('Storing examples')
        x_adv = np.concatenate(x_adv, axis=0)
        if (params.attack_type == 'ldg_dec') or (params.attack_type == 'ldg_dec_v2') :
            np.save('imagenet_out/{}_{}_{}_{}_{}_{}.npy'.format(params.attack_type,
                                                                params.resize,
                                                                params.dec_select,
                                                                params.dec_scale,
                                                                params.dec_keep,
                                                                params.max_q), attack.queries)
        print('Average queries: {:.2f}'.format(np.mean(attack.queries)))
        for key, val in vars(params).items():
            print('{}={}'.format(key,val))

        # error checking
        if np.amax(x_adv) > 1.0001 or \
                np.amin(x_adv) < -0.0001 or \
                np.isnan(np.amax(x_adv)):
            print('Invalid pixel range. Expected [0,1], fount [{},{}]'.format(np.amin(x_adv),
                                                                              np.amax(x_adv)))
        else:
            run_attack(x_adv, sess, attack, x_full_batch, y_full_batch)
