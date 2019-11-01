# Implementation of ResNet modified from
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def get_shape(tensor):
    print(tensor.get_shape().as_list())


def generator(x, f_dim=64, c_dim=1, drop=0, unet=False, is_training=True):
    ngf = f_dim

    net = tf.layers.conv2d(x, filters=ngf, kernel_size=5, strides=2, padding='same',
                           kernel_initializer=tf.variance_scaling_initializer(scale=2))
    net = tf.layers.batch_normalization(net, training=True)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, filters=ngf * 2, kernel_size=5, strides=2, padding='same',
                           kernel_initializer=tf.variance_scaling_initializer(scale=2))
    net = tf.layers.batch_normalization(net, training=True)
    net = tf.nn.relu(net)

    net = tf.layers.conv2d_transpose(net, filters=ngf, kernel_size=5, strides=2, padding='same', use_bias=False,
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))
    net = tf.layers.batch_normalization(net, training=True)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d_transpose(net, filters=c_dim, kernel_size=5, strides=2, padding='same', use_bias=False,
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))

    net = tf.nn.tanh(net)

    return net


# helper function for convolution -> instance norm -> relu
def ConvInstNormRelu(x, filters, kernel_size=3, strides=1):
    Conv = tf.layers.conv2d(
        inputs=x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        activation=None)

    InstNorm = tf.contrib.layers.instance_norm(Conv)

    return tf.nn.relu(InstNorm)


# helper function for trans convolution -> instance norm -> relu
def TransConvInstNormRelu(x, filters, kernel_size=3, strides=2):
    TransConv = tf.layers.conv2d_transpose(
        inputs=x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        activation=None)

    InstNorm = tf.contrib.layers.instance_norm(TransConv)

    return tf.nn.relu(InstNorm)


# helper function for residual block of 2 convolutions with same num filters
# in the same style as ConvInstNormRelu
def ResBlock(x, training, filters=32, kernel_size=3, strides=1):
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        activation=None)

    conv1_norm = tf.layers.batch_normalization(conv1, training=training)

    conv1_relu = tf.nn.relu(conv1_norm)

    conv2 = tf.layers.conv2d(
        inputs=conv1_relu,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        activation=None)

    conv2_norm = tf.layers.batch_normalization(conv2, training=training)

    return x + conv2_norm


def generator_v2(x, f_dim=8, c_dim=1, drop=0, unet=False, is_training=True):
    with tf.variable_scope('g_weights', reuse=tf.AUTO_REUSE):
        # input_layer = tf.reshape(x, [-1, 28, 28, 1])

        ngf = f_dim

        # define first three conv + inst + relu layers
        c1 = ConvInstNormRelu(x, filters=ngf, kernel_size=3, strides=1)
        d1 = ConvInstNormRelu(c1, filters=ngf*2, kernel_size=3, strides=2)
        d2 = ConvInstNormRelu(d1, filters=ngf*4, kernel_size=3, strides=2)

        # define residual blocks
        rb1 = ResBlock(d2, is_training, filters=ngf*4)
        rb2 = ResBlock(rb1, is_training, filters=ngf*4)
        rb3 = ResBlock(rb2, is_training, filters=ngf*4)
        rb4 = ResBlock(rb3, is_training, filters=ngf*4)

        # upsample using conv transpose
        u1 = TransConvInstNormRelu(rb4, filters=ngf*2, kernel_size=3, strides=2)
        u2 = TransConvInstNormRelu(u1, filters=ngf, kernel_size=3, strides=2)

        # final layer block
        out = tf.layers.conv2d_transpose(
            inputs=u2,
            filters=x.get_shape()[-1].value, # or 3 if RGB image
            kernel_size=3,
            strides=1,
            padding="same",
            activation=None)

        # out = tf.contrib.layers.instance_norm(out)

        return tf.nn.tanh(out)

