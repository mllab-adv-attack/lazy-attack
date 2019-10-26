import tensorflow as tf

def get_shape(tensor):
    return tensor.get_shape().as_list()


def batch_norm(*args, **kwargs):
    with tf.name_scope('bn'):
        bn = tf.layers.batch_normalization(*args, **kwargs)
    return bn


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

    def _build_layer(self, name, inputs, k, bn=True, use_dropout=False):
        layer = dict()
        with tf.variable_scope(name):
            layer['filters'] = tf.get_variable('filters', [3, 3, get_shape(inputs)[-1], k])
            layer['conv'] = tf.nn.conv2d(inputs, layer['filters'], strides=[1, 2, 2, 1], padding='SAME')
            layer['bn'] = batch_norm(layer['conv'], center=self._center, scale=self._scale, training=self._is_training) if bn else layer['conv']
            layer['dropout'] = tf.nn.dropout(layer['bn'], self._prob) if use_dropout else layer['bn']
            layer['fmap'] = lkrelu(layer['dropout'], slope=0.2)
        return layer
    '''
    def __call__(self, inputs):
        discriminator = dict()

        with tf.variable_scope('discriminator', initializer=tf.truncated_normal_initializer(stddev=self._stddev),
                               reuse=tf.AUTO_REUSE):

            # C64-C128-C256-C512 -> PatchGAN
            discriminator['l1'] = self._build_layer('l1', inputs, 32, bn=False)
            discriminator['l2'] = self._build_layer('l2', discriminator['l1']['fmap'], 64)
            discriminator['l3'] = self._build_layer('l3', discriminator['l2']['fmap'], 128)
            discriminator['l4'] = self._build_layer('l4', discriminator['l3']['fmap'], 256)
            with tf.variable_scope('l5'):
                l5 = dict()
                l5['filters'] = tf.get_variable('filters', [4, 4, get_shape(discriminator['l4']['fmap'])[-1], 1])
                l5['conv'] = tf.nn.conv2d(discriminator['l4']['fmap'], l5['filters'], strides=[1, 1, 1, 1], padding='SAME')
                l5['bn'] = batch_norm(l5['conv'], center=self._center, scale=self._scale, training=self._is_training)
                l5['fmap'] = tf.nn.sigmoid(l5['bn'])
                discriminator['l5'] = l5

        return discriminator['l5']['bn']
    '''

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
