import tensorflow as tf


def get_shape(tensor):
    return tensor.get_shape().as_list()


class Discriminator(object):
    def __init__(self):
        pass

    def __call__(self, x):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            # input_layer = tf.reshape(x, [-1, 28, 28, 1])

            # [0, 1] --> [-1, 1]
            x = (x-0.5)*2

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

            # probs = tf.nn.sigmoid(logits)

            return logits


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
class Pix2pixDiscriminator(object):
    def __init__(self, patch, is_training, f_dim=32, multi_class=False):
        self.patch = patch
        self._is_training = is_training

        self._prob = 0.5  # constant from pix2pix paper
        self.f_dim = f_dim
        self.multi_class = multi_class

    def __call__(self, inputs, noises):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

            inputs = inputs * 2 - 1
            inputs = tf.concat([inputs, noises], axis=-1)

            inputs = encoder_layer(inputs, self.f_dim, 3, 2, batch_norm=False, is_training=self._is_training)
            inputs = encoder_layer(inputs, self.f_dim*2, 3, 1, is_training=self._is_training)
            inputs = encoder_layer(inputs, self.f_dim*4, 3, 1, is_training=self._is_training)
            inputs = encoder_layer(inputs, self.f_dim*8, 3, 1, is_training=self._is_training)

            if self.patch:
                if self.multi_class:
                    inputs = tf.layers.conv2d(inputs, 10, kernel_size=[3, 3], strides=[1, 1], padding='same')
                else:
                    inputs = tf.layers.conv2d(inputs, 1, kernel_size=[3, 3], strides=[1, 1], padding='same')
                # print(get_shape(inputs))
            else:
                inputs = tf.layers.flatten(inputs)
                if self.multi_class:
                    inputs = tf.layers.dense(inputs, 10)
                else:
                    inputs = tf.layers.dense(inputs, 1)

        return inputs


class DenseDiscriminator(object):
    def __init__(self, is_training, f_dim=512, multi_class=False):
        self._is_training = is_training

        self._prob = 0.5  # constant from pix2pix paper
        self.f_dim = f_dim
        self.multi_class = multi_class

    def __call__(self, inputs, noises):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

            inputs = tf.concat([inputs, noises])
            inputs = tf.layers.dense(inputs, self.f_dim)
            inputs = lkrelu(inputs, 0.2)

            inputs = tf.layers.dense(inputs, self.f_dim)
            inputs = lkrelu(inputs, 0.2)
            inputs = tf.layers.dropout(inputs, rate=0.4)

            inputs = tf.layers.dense(inputs, self.f_dim)
            inputs = lkrelu(inputs, 0.2)
            inputs = tf.layers.dropout(inputs, rate=0.4)

            if self.multi_class:
                inputs = tf.layers.dense(inputs, 10)
            else:
                inputs = tf.layers.dense(inputs, 1)

        return inputs
