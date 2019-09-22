# Implementation of ResNet modified from
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


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
    data_format):
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

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
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
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format)

  return tf.identity(inputs, name)


def generator(x, f_dim, output_size, c_dim, is_training=True):
    ngf = f_dim
    inputs = x
    data_format='channels_first'

    # for GPU; channels_first
    if data_format=='channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(inputs, filters=ngf, kernel_size=3, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, is_training, data_format)
    inputs = tf.nn.relu(inputs)

    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        inputs = conv2d_fixed_padding(inputs, filters=ngf*mult*2, kernel_size=3, strides=2, data_format=data_format)
        inputs = batch_norm(inputs, is_training, data_format)
        inputs = tf.nn.relu(inputs)

    mult = 2**n_downsampling
    inputs = block_layer(
          inputs=inputs, filters=ngf*mult, bottleneck=False,
          block_fn=_building_block_v1, blocks=6,
          strides=1, training=is_training,
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


def generate_pgd_common(x,
                        bounds,
                        model_fn,
                        attack_params,
                        one_hot_labels,
                        perturbation_multiplier):
    """Common code for generating PGD adversarial examples.
    Args:
      x: original examples.
      bounds: tuple with bounds of image values, bounds[0] < bounds[1].
      model_fn: model function with signature model_fn(images).
      attack_params: parameters of the attack.
      one_hot_labels: one hot label vector to use in the loss.
      perturbation_multiplier: multiplier of adversarial perturbation,
        either +1.0 or -1.0.
    Returns:
      Tensor with adversarial examples.
    Raises:
      ValueError: if attack parameters are invalid.
    """
    # parse attack_params
    # Format of attack_params: 'EPS_STEP_NITER'
    # where EPS - epsilon, STEP - step size, NITER - number of iterations
    epsilon = attack_params['eps']
    step_size = attack_params['step_size']
    niter = attack_params['num_steps']

    # rescale epsilon and step size to image bounds
    epsilon = float(epsilon) / 255.0 * (bounds[1] - bounds[0])
    step_size = float(step_size) / 255.0 * (bounds[1] - bounds[0])

    # clipping boundaries
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    # compute starting point
    start_x = x + tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    # main iteration of PGD
    loop_vars = [0, start_x]

    def loop_cond(index, _):
        return index < niter

    def loop_body(index, adv_images):
        logits = model_fn(adv_images)
        loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=one_hot_labels,
                logits=logits))
        perturbation = step_size * tf.sign(tf.gradients(loss, adv_images)[0])
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images

    with tf.control_dependencies([start_x]):
        _, result = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars,
            back_prop=False,
            parallel_iterations=1)
        return result


def generate_pgd(x, y, bounds, model_fn, attack_params):
    # pylint: disable=g-doc-args
    """Generats non-targeted PGD adversarial examples.
    See generate_pgd_common for description of arguments.
    Returns:
      tensor with adversarial examples.
    """
    # pylint: enable=g-doc-args

    # compute one hot predicted class
    logits = model_fn(x)
    num_classes = tf.shape(logits)[1]
    one_hot_labels = tf.one_hot(y, num_classes)

    return generate_pgd_common(x, bounds, model_fn, attack_params,
                               one_hot_labels=one_hot_labels,
                               perturbation_multiplier=1.0)


class Model(object):

    def __init__(self, mode, model, eps):

        self.mode = mode
        self.model = model
        self.imp_eps = eps
        self.attack_params = {
            'eps': 8,
            'step_size': 2,
            'num_steps': 20,
        }
        self.bounds = (0, 255)

        self._build_model()

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        is_training = True if self.model == 'train' else False

        with tf.variable_scope('infer_input', reuse=tf.AUTO_REUSE):
            self.x_input = tf.placeholder(
                tf.float32,
                shape=[None, 32, 32, 3])
            self.y_input = tf.placeholder(tf.int64, shape=None)

        safe_generator = tf.make_template('generator', generator, f_dim=64, output_size=32, c_dim=3, is_training=is_training)

        with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
            self.G_out = safe_generator(self.x_input)

            self.x_safe = tf.clip_by_value(self.x_input + self.imp_eps * self.G_out, 0, 255)
            

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):

            self.x_attacked = generate_pgd(self.x_safe, self.y_input, self.bounds, self.model.fprop, self.attack_params)
        with tf.variable_scope('orig_loss', reuse=tf.AUTO_REUSE):
            self.orig_pre_softmax = self.model.fprop(self.x_input)
            self.orig_softmax = tf.nn.softmax(self.orig_pre_softmax)

            self.orig_predictions = tf.argmax(self.orig_pre_softmax, 1)
            self.orig_correct_prediction = tf.equal(self.orig_predictions, self.y_input)
            self.orig_num_correct = tf.reduce_sum(
                tf.cast(self.orig_correct_prediction, tf.int32))
            self.orig_accuracy = tf.reduce_mean(
                tf.cast(self.orig_correct_prediction, tf.float32))

            self.orig_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.orig_pre_softmax, labels=self.y_input)
            self.orig_xent = tf.reduce_sum(self.orig_y_xent, name='orig_y_xent')
            self.orig_mean_xent = tf.reduce_mean(self.orig_y_xent)
        
        with tf.variable_scope('gen_loss', reuse=tf.AUTO_REUSE):
            self.gen_pre_softmax = self.model.fprop(self.x_safe)
            self.gen_softmax = tf.nn.softmax(self.gen_pre_softmax)

            self.gen_predictions = tf.argmax(self.gen_pre_softmax, 1)
            self.gen_correct_prediction = tf.equal(self.gen_predictions, self.y_input)
            self.gen_num_correct = tf.reduce_sum(
                tf.cast(self.gen_correct_prediction, tf.int32))
            self.gen_accuracy = tf.reduce_mean(
                tf.cast(self.gen_correct_prediction, tf.float32))

            self.gen_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.gen_pre_softmax, labels=self.y_input)
            self.gen_xent = tf.reduce_sum(self.gen_y_xent, name='gen_y_xent')
            self.gen_mean_xent = tf.reduce_mean(self.gen_y_xent)

        with tf.variable_scope('safe_loss', reuse=tf.AUTO_REUSE):
            self.pre_softmax = self.model.fprop(self.x_attacked)
            self.softmax = tf.nn.softmax(self.pre_softmax)

            self.predictions = tf.argmax(self.pre_softmax, 1)
            self.correct_prediction = tf.equal(self.predictions, self.y_input)
            self.num_correct = tf.reduce_sum(
                tf.cast(self.correct_prediction, tf.int32))
            self.accuracy = tf.reduce_mean(
                tf.cast(self.correct_prediction, tf.float32))

            self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pre_softmax, labels=self.y_input)
            self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
            self.mean_xent = tf.reduce_mean(self.y_xent)
        
