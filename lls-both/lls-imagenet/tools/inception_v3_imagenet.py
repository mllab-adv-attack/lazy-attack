"""This script is borrowed from https://github.com/labsix/limited-blackbox-attacks"""

from tools.utils import optimistic_restore
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import functools
import os

SIZE = 299

# to make this work, you need to download:
# http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
# and decompress it in the `data` directory


def _get_model(reuse):
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    func = nets.inception.inception_v3
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1001, is_training=False, reuse=reuse)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return network_fn

def _preprocess(image, height, width, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

# input is [batch, 256, 256, 3], pixels in [0, 1]
# output is [batch, 10]
class Model():
    def __init__(self):
        network_fn = _get_model(reuse=False)
        size = network_fn.default_image_size

        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        self.y_input = tf.placeholder(dtype=tf.int32, shape=[None])

        preprocessed = _preprocess(self.x_input, size, size)
        self.logits, _ = network_fn(preprocessed)
        self.logits = self.logits[:,1:] # ignore background class
        self.predictions = tf.argmax(self.logits, 1)

        self.probs = tf.nn.softmax(self.logits)

        batch_nums = tf.range(0, limit=tf.shape(self.logits)[0])
        indices = tf.stack([batch_nums, self.y_input], axis=1)

        """cw loss"""
        self.ground_truth_logits = tf.gather_nd(params=self.logits, indices=indices)
        top_2 = tf.nn.top_k(self.logits, k=2)
        max_indices = tf.where(tf.equal(top_2.indices[:, 0], self.y_input), top_2.indices[:, 1], top_2.indices[:, 0])
        max_indices = tf.stack([batch_nums, max_indices], axis=1)
        max_logits = tf.gather_nd(params=self.logits, indices=max_indices)
        self.loss_cw = max_logits - self.ground_truth_logits

        self.predictions = tf.argmax(self.logits, 1, output_type=tf.int32)
        self.correct_prediction = tf.equal(self.predictions, self.y_input)

        self.num_correct = tf.reduce_sum(
            tf.cast(self.correct_prediction, tf.int32))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))

        with tf.variable_scope('costs'):
            self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.y_input)
            self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
            self.mean_xent = tf.reduce_mean(self.y_xent)
