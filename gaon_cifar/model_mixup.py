# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Model(object):
    """ResNet model."""

    def __init__(self, mode, mixup_alpha=2.0):
        """ResNet constructor.

        Args:
          mode: One of 'train' and 'eval'.
        """
        self.mode = mode
        self.mixup_alpha = 2.0
        self._build_model()

    def add_internal_summaries(self):
        pass

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self, x_input, y_input):
        assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):

            if self.mode == 'train':
                layer_mix = tf.random.uniform([], 0, 3, dtype=tf.float32)
            else:
                layer_mix = tf.constant(-1)

            dist = tfp.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
            lam = dist.sample()

            self.x_input = tf.placeholder(
                tf.float32,
                shape=[None, 32, 32, 3])

            self.y_input = tf.placeholder(tf.int64, shape=None)
            y_one_hot = tf.one_hot(self.y_input, 10, on_value=1.0, off_value=0.0, dtype=tf.float32)

            self.y_input2 = tf.placeholder(tf.float32, shape=[None, 10])
            self.y_input2_argmax = tf.argmax(self.y_input2, axis=1)

            input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                                           self.x_input)
            # mixup
            x, y_one_hot = self._auto_mixup(input_standardized, y_one_hot, layer_mix, lam)

            x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        # Uncomment the following codes to use w28-10 wide residual network.
        # It is more memory efficient than very deep residual network and has
        # comparably good performance.
        # https://arxiv.org/pdf/1605.07146v1.pdf
        filters = [16, 160, 320, 640]

        # Update hps.num_residual_units to 9

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in range(1, 5):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        # mixup
        x, y_one_hot = self._auto_mixup(x, y_one_hot, layer_mix, lam)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in range(1, 5):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        # mixup
        x, y_one_hot = self._auto_mixup(x, y_one_hot, layer_mix, lam)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in range(1, 5):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, 0.1)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            self.pre_softmax = self._fully_connected(x, 10)
            self.softmax = tf.nn.softmax(self.pre_softmax)

        self.predictions = tf.argmax(self.pre_softmax, 1)
        self.correct_prediction = tf.equal(self.predictions, self.y_input)
        self.num_correct = tf.reduce_sum(
            tf.cast(self.correct_prediction, tf.int32))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))
        self.correct_prediction2 = tf.equal(self.predictions, self.y_input2_argmax)
        self.num_correct2 = tf.reduce_sum(
            tf.cast(self.correct_prediction2, tf.int32))
        self.accuracy2 = tf.reduce_mean(
            tf.cast(self.correct_prediction2, tf.float32))

        with tf.variable_scope('costs'):
            self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pre_softmax, labels=self.y_input)
            self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
            self.mean_xent = tf.reduce_mean(self.y_xent)
            self.weight_decay_loss = self._decay()

        with tf.variable_scope('costs2'):
            self.y_xent2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.pre_softmax, labels=self.y_input2)
            self.xent2 = tf.reduce_sum(self.y_xent2, name='y_xent2')
            self.mean_xent2 = tf.reduce_mean(self.y_xent2)
            self.weight_decay_loss = self._decay()

        with tf.variable_scope('cw_loss'):
            label_mask = tf.one_hot(self.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            self.correct_logit = tf.reduce_sum(label_mask * self.pre_softmax,
                                               axis=1)
            self.wrong_logit = tf.reduce_max((1 - label_mask) * self.pre_softmax,
                                             axis=1)
            self.cw_loss = -tf.nn.relu(self.correct_logit - self.wrong_logit + 50)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(
                inputs=x,
                decay=.9,
                center=True,
                scale=True,
                activation_fn=None,
                updates_collections=None,
                is_training=(self.mode == 'train'))

    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, 0.1)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    @staticmethod
    def _mixup_data(x, y, lam):
        batch_size = tf.shape(x)[0]
        index = tf.range(batch_size)
        index = tf.random.shuffle(index)
        mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
        mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
        return mixed_x, mixed_y

    @staticmethod
    def _no_mixup_data(x, y):
        return x, y

    def _auto_mixup(self, x, y, layer_mix, lam):
        new_x, new_y = tf.cond(
            tf.equal(layer_mix, 0),
            lambda: self._mixup_data(x, y, lam),
            lambda: self._no_mixup_data(x, y)
        )

        return new_x, new_y
