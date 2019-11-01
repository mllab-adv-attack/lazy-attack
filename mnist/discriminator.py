import tensorflow as tf


def get_shape(tensor):
    return tensor.get_shape().as_list()


class Discriminator(object):
    def __init__(self):
        pass

    def __call__(self, x):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            # input_layer = tf.reshape(x, [-1, 28, 28, 1])

            conv1 = tf.layers.conv2d(
                inputs=x,
                filters=8,
                kernel_size=4,
                strides=2,
                padding="valid",
                activation=None)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=16,
                kernel_size=4,
                strides=2,
                padding="valid",
                activation=None)

            in1 = tf.contrib.layers.instance_norm(conv2)
            conv2 = tf.nn.leaky_relu(in1, alpha=0.2)

            conv3 = tf.layers.conv2d(
                inputs=conv2,
                filters=32,
                kernel_size=4,
                strides=2,
                padding="valid",
                activation=None)

            in2 = tf.contrib.layers.instance_norm(conv3)
            conv3 = tf.nn.leaky_relu(in2, alpha=0.2)
            flat = tf.layers.flatten(conv3)
            logits = tf.layers.dense(flat, 1)

            #probs = tf.nn.sigmoid(logits)

            return logits
