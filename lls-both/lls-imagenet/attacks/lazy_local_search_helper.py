import cv2
import tensorflow as tf
import numpy as np
import heapq
import math
import sys


class LazyLocalSearchHelper(object):
    def __init__(self, model, args, **kwargs):
        self.epsilon = args.epsilon
        self.targeted = args.targeted
        self.loss_func = args.loss_func
        self.image_range = args.image_range

        # Network Setting
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

        if not self.targeted:
            self.losses = -self.losses

    def _perturb_image(self, image, noise):
        adv_image = image + cv2.resize(noise[0, ...], (image.shape[1], image.shape[2]), interpolation=cv2.INTER_NEAREST)
        adv_image = np.clip(adv_image, 0, self.image_range)
        return adv_image

    def _flip_noise(self, noise, block):
        noise_new = np.copy(noise)
        upper_left, lower_right, channel = block
        noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] *= -1
        return noise_new

    def perturb(self, image, noise, label, sess, blocks):
        # Local variables
        priority_queue = []
        num_queries = 0

        # Check if blocks are in the working set
        A = np.zeros([len(blocks)], np.int32)
        for i, block in enumerate(blocks):
            upper_left, _, channel = block
            x = upper_left[0]
            y = upper_left[1]
            # If flipped, set A to be 1.
            if noise[0, x, y, channel] > 0:
                A[i] = 1

        # Calculate current loss
        image_batch = self._perturb_image(image, noise)
        label_batch = np.copy(label)
        losses, preds = sess.run([self.losses, self.preds],
                                 feed_dict={self.x_input: image_batch, self.y_input: label_batch})
        num_queries += 1
        curr_loss = losses[0]

        # Early stopping
        if self.targeted:
            if preds == label:
                return noise, num_queries, curr_loss, True
        else:
            if preds != label:
                return noise, num_queries, curr_loss, True

        # Main loop
        for _ in range(1):
            # Lazy Greedy Insert
            indices, = np.where(A == 0)

            batch_size = 100
            num_batches = int(math.ceil(len(indices) / batch_size))

            for ibatch in range(num_batches):
                bstart = ibatch * batch_size
                bend = min(bstart + batch_size, len(indices))

                image_batch = np.zeros([bend - bstart, image.shape[1], image.shape[2], 3], image.dtype)
                noise_batch = np.zeros([bend - bstart, noise.shape[1], noise.shape[2], 3], image.dtype)
                label_batch = np.tile(label, bend - bstart)

                for i, idx in enumerate(indices[bstart:bend]):
                    noise_batch[i:i + 1, ...] = self._flip_noise(noise, blocks[idx])
                    image_batch[i:i + 1, ...] = self._perturb_image(image, noise_batch[i:i + 1, ...])

                losses, preds = sess.run([self.losses, self.preds],
                                         feed_dict={self.x_input: image_batch, self.y_input: label_batch})

                # Early stopping
                success_indices, = np.where(preds == label) if self.targeted else np.where(preds != label)
                if len(success_indices) > 0:
                    noise[0, ...] = noise_batch[success_indices[0], ...]
                    curr_loss = losses[success_indices[0]]
                    num_queries += success_indices[0] + 1
                    return noise, num_queries, curr_loss, True

                num_queries += bend - bstart

                # Push into the priority queue
                for i in range(bend - bstart):
                    idx = indices[bstart + i]
                    margin = losses[i] - curr_loss
                    heapq.heappush(priority_queue, (margin, idx))

            # Pick the best element and insert it into working set
            if len(priority_queue) > 0:
                best_margin, best_idx = heapq.heappop(priority_queue)
                if best_margin <= 0:
                    curr_loss += best_margin
                    noise = self._flip_noise(noise, blocks[best_idx])
                    A[best_idx] = 1

            # Add elements into working set
            while len(priority_queue) > 0:
                # Pick the best element
                cand_margin, cand_idx = heapq.heappop(priority_queue)
                # Re-evalulate the element
                image_batch = self._perturb_image(
                    image, self._flip_noise(noise, blocks[cand_idx]))
                label_batch = np.copy(label)

                losses, preds = sess.run([self.losses, self.preds],
                                         feed_dict={self.x_input: image_batch, self.y_input: label_batch})
                num_queries += 1
                margin = losses[0] - curr_loss

                # If the cardinality has not changed, add the element
                if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
                    # If there is no element that has negative margin, then break
                    if margin > 0:
                        break
                    # Update noise
                    curr_loss = losses[0]
                    noise = self._flip_noise(noise, blocks[cand_idx])
                    A[cand_idx] = 1
                    # Early stopping
                    if self.targeted:
                        if preds == label:
                            return noise, num_queries, curr_loss, True
                    else:
                        if preds != label:
                            return noise, num_queries, curr_loss, True
                # If the cardinality has changed, push the element into the priority queue
                else:
                    heapq.heappush(priority_queue, (margin, cand_idx))

            priority_queue = []

            # Lazy Greedy Delete
            indices, = np.where(A == 1)

            batch_size = 100
            num_batches = int(math.ceil(len(indices) / batch_size))

            for ibatch in range(num_batches):
                bstart = ibatch * batch_size
                bend = min(bstart + batch_size, len(indices))

                image_batch = np.zeros([bend - bstart, image.shape[1], image.shape[2], 3], image.dtype)
                noise_batch = np.zeros([bend - bstart, noise.shape[1], noise.shape[2], 3], image.dtype)
                label_batch = np.tile(label, bend - bstart)

                for i, idx in enumerate(indices[bstart:bend]):
                    noise_batch[i:i + 1, ...] = self._flip_noise(noise, blocks[idx])
                    image_batch[i:i + 1, ...] = self._perturb_image(image, noise_batch[i:i + 1, ...])

                losses, preds = sess.run([self.losses, self.preds],
                                         feed_dict={self.x_input: image_batch, self.y_input: label_batch})

                # Early stopping
                success_indices, = np.where(preds == label) if self.targeted else np.where(preds != label)
                if len(success_indices) > 0:
                    noise[0, ...] = noise_batch[success_indices[0], ...]
                    curr_loss = losses[success_indices[0]]
                    num_queries += success_indices[0] + 1
                    return noise, num_queries, curr_loss, True

                num_queries += bend - bstart

                # Push into the priority queue
                for i in range(bend - bstart):
                    idx = indices[bstart + i]
                    margin = losses[i] - curr_loss
                    heapq.heappush(priority_queue, (margin, idx))

            # Pick the best element and remove it from working set
            if len(priority_queue) > 0:
                best_margin, best_idx = heapq.heappop(priority_queue)
                if best_margin < 0:
                    curr_loss += best_margin
                    noise = self._flip_noise(noise, blocks[best_idx])
                    A[best_idx] = 0

            # Delete elements into working set
            while len(priority_queue) > 0:
                # pick the best element
                cand_margin, cand_idx = heapq.heappop(priority_queue)
                # Re-evalulate the element
                image_batch = self._perturb_image(
                    image, self._flip_noise(noise, blocks[cand_idx]))
                label_batch = np.copy(label)

                losses, preds = sess.run([self.losses, self.preds],
                                         feed_dict={self.x_input: image_batch, self.y_input: label_batch})
                num_queries += 1
                margin = losses[0] - curr_loss

                # If the cardinality has not changed, remove the element
                if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
                    # If there is no element that has negative margin, then break
                    if margin >= 0:
                        break
                    # Update noise
                    curr_loss = losses[0]
                    noise = self._flip_noise(noise, blocks[cand_idx])
                    A[cand_idx] = 0
                    # Early stopping
                    if self.targeted:
                        if preds == label:
                            return noise, num_queries, curr_loss, True
                    else:
                        if preds != label:
                            return noise, num_queries, curr_loss, True
                # If the cardinality has changed, push the element into the priority queue
                else:
                    heapq.heappush(priority_queue, (margin, cand_idx))

            priority_queue = []

        return noise, num_queries, curr_loss, False

