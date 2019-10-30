# Implementation of ResNet modified from
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def get_shape(tensor):
    print(tensor.get_shape().as_list())

################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
    drop, data_format):
  """
  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = tf.layers.dropout(inputs, rate=drop, training=training)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                drop, training, name, data_format):
  """Creates one layer of blocks for the ResNet model.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    drop, data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, drop, data_format)

  return tf.identity(inputs, name)


def generator(x, f_dim=64, c_dim=3, n_down=2, n_blocks=6, drop=0, is_training=True):
    ngf = f_dim
    inputs = x
    data_format='channels_last'

    # normalize ([0, 255] -> [-1, 1])
    inputs = (inputs/255) * 2 - 1

    # for GPU; channels_first
    if data_format=='channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(inputs, filters=ngf, kernel_size=3, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, is_training, data_format)
    inputs = tf.nn.relu(inputs)

    n_downsampling = n_down
    for i in range(n_downsampling):
        mult = 2**i
        inputs = conv2d_fixed_padding(inputs, filters=ngf*mult*2, kernel_size=3, strides=2, data_format=data_format)
        inputs = batch_norm(inputs, is_training, data_format)
        inputs = tf.nn.relu(inputs)

    mult = 2**n_downsampling
    inputs = block_layer(
          inputs=inputs, filters=ngf*mult, bottleneck=False,
          block_fn=_building_block_v1, blocks=n_blocks,
          strides=1, drop=drop, training=is_training,
          name='block_layer_G{}'.format(1), data_format=data_format)

    for i in range(n_downsampling):
        mult = 2**(n_downsampling-i)
        # fix this 2d transpose
        inputs = tf.layers.conv2d_transpose(inputs, filters=int(ngf*mult/2), kernel_size=3, strides=(2,2),
                 padding='same', kernel_initializer=tf.truncated_normal_initializer(0.0, 0.02), data_format=data_format, use_bias=False)
        inputs = batch_norm(inputs, is_training, data_format)
        inputs = tf.nn.relu(inputs)

    inputs = tf.layers.conv2d(inputs, filters=c_dim, kernel_size=3, strides=1,
      padding='same', use_bias=True, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.02), data_format=data_format)
    inputs = tf.nn.tanh(inputs)

    if data_format=='channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    return inputs


def unet_generator(x, is_training=True):

    inputs = x
    inputs = (inputs/255) * 2 - 1

    conv1 = tf.layers.conv2d(inputs, 16, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    conv1 = tf.layers.conv2d(conv1, 16, 3,  activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    #get_shape(pool1)

    conv2 = tf.layers.conv2d(pool1, 32, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    conv2 = tf.layers.conv2d(conv2, 32, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    #get_shape(pool2)

    conv3 = tf.layers.conv2d(pool2, 64, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    conv3 = tf.layers.conv2d(conv3, 64, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2)
    #get_shape(pool3)

    conv4 = tf.layers.conv2d(pool3, 128, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    conv4 = tf.layers.conv2d(conv4, 128, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    drop4 = tf.layers.dropout(conv4, training=is_training)
    pool4 = tf.layers.max_pooling2d(drop4, 2, 2)
    #get_shape(pool4)

    conv5 = tf.layers.conv2d(pool4, 256, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    conv5 = tf.layers.conv2d(conv5, 256, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    drop5 = tf.layers.dropout(conv5, training=is_training)
    #get_shape(drop5)

    up6 = tf.keras.layers.UpSampling2D()(drop5)
    up6 = tf.layers.conv2d(up6, 128, 2, activation='relu', padding='same',
                           kernel_initializer=tf.initializers.he_normal())
    merge6 = tf.concat([drop4, up6], axis=3)
    conv6 = tf.layers.conv2d(merge6, 128, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    conv6 = tf.layers.conv2d(conv6, 128, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    #get_shape(conv6)

    up7 = tf.keras.layers.UpSampling2D()(conv6)
    up7 = tf.layers.conv2d(up7, 64, 2, activation='relu', padding='same',
                           kernel_initializer=tf.initializers.he_normal())
    merge7 = tf.concat([conv3, up7], axis=3)
    conv7 = tf.layers.conv2d(merge7, 64, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    conv7 = tf.layers.conv2d(conv7, 64, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    #get_shape(conv7)

    up8 = tf.keras.layers.UpSampling2D()(conv7)
    up8 = tf.layers.conv2d(up8, 32, 2, activation='relu', padding='same',
                           kernel_initializer=tf.initializers.he_normal())
    merge8 = tf.concat([conv2, up8], axis=3)
    conv8 = tf.layers.conv2d(merge8, 32, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    conv8 = tf.layers.conv2d(conv8, 32, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    #get_shape(conv8)

    up9 = tf.keras.layers.UpSampling2D()(conv8)
    up9 = tf.layers.conv2d(up9, 16, 2, activation='relu', padding='same',
                           kernel_initializer=tf.initializers.he_normal())
    merge9 = tf.concat([conv1, up9], axis=3)
    conv9 = tf.layers.conv2d(merge9, 16, 3, activation='relu', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    outputs = tf.layers.conv2d(conv9, 3, 3, activation='tanh', padding='same',
                             kernel_initializer=tf.initializers.he_normal())
    #get_shape(outputs)

    return outputs
