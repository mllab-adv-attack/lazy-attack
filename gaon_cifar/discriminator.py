import tensorflow as tf


def get_shape(tensor):
    return tensor.get_shape().as_list()


def lkrelu(x, slope=0.01):
    return tf.maximum(slope * x, x)


def encoder_layer(inputs,
                  filters=16,
                  kernel=3,
                  stride=2,
                  batch_norm=True,
                  padding='valid',
                  is_training=True,
                  ):
    """Builds a generic encoder layer made of Conv2D-IN-LeakyReLU
    IN is optional, LeakyReLU may be replaced by ReLU
    """

    x = inputs
    x = tf.layers.conv2d(x, filters, kernel_size=[kernel, kernel], strides=[stride, stride], padding=padding)
    if batch_norm:
        x = tf.layers.batch_normalization(x, momentum=0.8, training=is_training)
    x = lkrelu(x, slope=0.2)
    x = tf.layers.dropout(x, rate=0.25, training=is_training)
    return x


# PatchGAN
class Discriminator(object):
    def __init__(self, patch, is_training, stddev=0.02, center=True, scale=True):
        self.patch = patch
        self._is_training = is_training
        self._stddev = stddev

        self._center = center
        self._scale = scale
        self._prob = 0.5  # constant from pix2pix paper

    def __call__(self, inputs):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

            inputs = (inputs/255) * 2 - 1

            inputs = encoder_layer(inputs, 32, 3, 2, batch_norm=False, is_training=self._is_training)
            inputs = encoder_layer(inputs, 64, 3, 1, is_training=self._is_training)
            inputs = encoder_layer(inputs, 128, 3, 1, is_training=self._is_training)
            inputs = encoder_layer(inputs, 256, 3, 1, is_training=self._is_training)

            if self.patch:
                inputs = tf.layers.conv2d(inputs, 1, kernel_size=[3, 3], strides=[1, 1], padding='same')
                print(get_shape(inputs))
            else:
                inputs = tf.layers.flatten(inputs)
                inputs = tf.layers.dense(inputs, 1)

        return inputs
