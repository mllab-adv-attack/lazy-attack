import cv2
import tensorflow as tf
import math
import numpy as np
import time
import itertools

from attacks.lazy_local_search_helper import LazyLocalSearchHelper
from attacks.graph_cut_helper import GraphCutHelper

class LazyLocalSearchAttack(object):
    def __init__(self, model, args, **kwargs):
        self.loss_func = args.loss_func
        self.max_queries = args.max_queries
        self.epsilon = args.epsilon

        self.lls_iter = args.lls_iter
        self.batch_size = args.batch_size
        self.lls_block_size = args.lls_block_size
        self.no_hier = args.no_hier

        self.noise_size = args.noise_size
        self.image_range = args.image_range

        self.history = {'step': [], 'block_size': [], 'num_batch': [], 'num_queries': [], 'loss': [], 'success':[]}

        self.lazy_local_search = LazyLocalSearchHelper(model, args)
        self.graph_cut_helper = GraphCutHelper(args)

        self.model = model
        self.x_input = model.x_input
        self.y_input = model.y_input
        self.logits = model.logits
        self.preds = model.predictions

        if self.loss_func == 'xent':
            self.losses = model.y_xent
        elif self.loss_func == 'cw':
            self.losses = model.loss_cw
        else:
            tf.logging.info('Loss function must be xent or cw')
            sys.exit()

        if not args.targeted:
            self.losses = -self.losses

    @staticmethod
    def _split_block(upper_left, lower_right, block_size):
        blocks = []
        xs = np.arange(upper_left[0], lower_right[0], block_size)
        ys = np.arange(upper_left[1], lower_right[1], block_size)
        for x, y in itertools.product(xs, ys):
            for c in range(3):
                blocks.append([[x, y], [x + block_size, y + block_size], c])
        return blocks

    def _perturb_image(self, image, noise):
        adv_image = image + cv2.resize(noise[0, ...], (image.shape[1], image.shape[2]), interpolation=cv2.INTER_NEAREST)
        adv_image = np.clip(adv_image, 0, self.image_range)
        return adv_image

    def perturb(self, image, label, index, sess):
        # Set random seed by index for the reproducibility
        np.random.seed(index)

        # Class variables
        self.width = image.shape[1]
        self.height = image.shape[2]

        # Local variables
        adv_image = np.copy(image)
        num_queries = 0
        lls_block_size = self.lls_block_size

        # Initialize noise
        noise = -self.epsilon * np.ones([1, self.noise_size, self.noise_size, 3], dtype=image.dtype)

        # for batch ordering
        flip_count = np.zeros([1, self.noise_size, self.noise_size, 3])

        # for unary term, gain := f(x=1) - f(x=-1) (**lower is better !!**)
        latest_gain = np.zeros([1, self.noise_size, self.noise_size, 3])

        # Split image into blocks
        blocks = self._split_block([0, 0], [self.noise_size, self.noise_size], lls_block_size)

        # Prepare for batch
        num_blocks = len(blocks)
        batch_size = self.batch_size if self.batch_size > 0 else num_blocks
        curr_order = np.random.choice(num_blocks, int(0.5*num_blocks))

        step = 0
        self.history = {'step': [], 'block_size': [], 'num_batch': [], 'num_queries': [], 'loss': [], 'success':[]}
        while True:
            num_batches = int(math.ceil(len(curr_order) / batch_size))

            for i in range(num_batches):
                # Construct a mini-batch
                bstart = i * batch_size
                bend = min(bstart + batch_size, len(curr_order))
                blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]

                # Run lazy local search
                noise, queries, loss, success = self.lazy_local_search.perturb(
                    image, noise, label, sess, blocks_batch, flip_count, latest_gain)

                num_queries += queries
                tf.logging.info("Block size: {}, batch: {}, loss: {:.4f}, num queries: {}".format(
                    lls_block_size, i, loss, num_queries))
                # tf.logging.info("total flip count : {}, latest gain : {:.4f}/ {:.4f}".format(np.sum(flip_count)/lls_block_size**2, np.max(latest_gain), np.min(latest_gain)))

                self.history['step'].append(step)
                self.history['block_size'].append(lls_block_size)
                self.history['num_batch'].append(i)
                self.history['num_queries'].append(num_queries)
                self.history['loss'].append(loss)
                self.history['success'].append(success)

                # If query count exceeds max queries, then return False
                if num_queries > self.max_queries:
                    return adv_image, num_queries, False

                # Update adversarial image
                adv_image = self._perturb_image(image, noise)

                # If success, return True
                if success:
                    return adv_image, num_queries, True
          
            if step > 0: 
                mask = np.ones([self.noise_size//lls_block_size, self.noise_size//lls_block_size, 3], dtype=np.int32)
                assignment = np.zeros([self.noise_size//lls_block_size, self.noise_size//lls_block_size, 3], dtype=np.int32)
                marginal_gain = np.zeros([self.noise_size//lls_block_size, self.noise_size//lls_block_size, 3], dtype=np.float32)

                for block in blocks:
                    upper_left, lower_right, c = block
                    x, y = upper_left
                    h = x//lls_block_size
                    w = y//lls_block_size
                    assignment[h, w, c] = np.sign(noise[0, x, y, c]).astype(np.int32)
                    marginal_gain[h, w, c] = latest_gain[0, x, y, c] 
                 
                for idx in curr_order:
                    upper_left, lower_right, c = blocks[idx]
                    x, y = upper_left
                    h = x//lls_block_size
                    w = y//lls_block_size
                    mask[h, w, c] = 0
                 
                for c in range(3):
                    mask_channel = mask[:, :, c]
                    assignment_channel = assignment[:, :, c]
                    marginal_gain_channel = marginal_gain[:, :, c]
                    self.graph_cut_helper.create_graph(mask_channel, assignment_channel, marginal_gain_channel)
                    result = self.graph_cut_helper.solve()
                    
                    hs, ws = np.where(mask_channel==1)
                    for h, w in zip(hs, ws):
                        x = h*lls_block_size
                        y = w*lls_block_size
                        noise[0, x:x+lls_block_size, y:y+lls_block_size, c] = result[h, w]*self.epsilon
            
            adv_image = self._perturb_image(image, noise)
            loss = sess.run(self.losses, feed_dict={self.x_input: adv_image, self.y_input: label})
            tf.logging.info('loss: {}'.format(loss[0]))        

            # If block size >= 2, then split blocks
            if not self.no_hier and lls_block_size > 1 and (step + 1) % self.lls_iter == 0:
                lls_block_size //= 2
                blocks = self._split_block([0, 0], [self.noise_size, self.noise_size], lls_block_size)
                num_blocks = len(blocks)
                batch_size = self.batch_size if self.batch_size > 0 else num_blocks

            curr_order = np.random.choice(num_blocks, int(0.5*num_blocks))
            step += 1
