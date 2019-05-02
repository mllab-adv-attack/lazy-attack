import cv2
import tensorflow as tf
import math
import numpy as np
import itertools

from attacks.lazy_local_search_helper import LazyLocalSearchHelper

from tools.imagenet_labels import label_to_name
from PIL import Image

class LazyLocalSearchAttack(object):
  """Lazy Local Search Attack with Hierarchical Method and Mini-batch Technique"""

  def __init__(self, model, args, **kwargs):
    """Initialize attack method.
    
    Args:
      model: tensorflow model
      args: arguments
    """
    # Setting
    self.loss_func = args.loss_func
    self.max_queries = args.max_queries
    self.epsilon = args.epsilon
    self.batch_size = args.batch_size
    self.block_size = args.block_size
    self.no_hier = args.no_hier
    self.max_iters = args.max_iters

    # Load lazy local search helper
    self.lazy_local_search = LazyLocalSearchHelper(model, args)
 
  def _perturb_image(self, image, noise):
    """Perturb an image with a noise. First, resize the noise with the size of the image. Then add 
    the resized noise to the image. Finally, clip the value of the perturbed image into [0, 1].
    
    Args:
      image: numpy array with size [1, 299, 299, 3], original image
      noise: numpy array with size [1, 256, 256, 3], noise
      
    Returns:
      adv_iamge: numpy array with size [1, 299, 299, 3], perturbed image   
    """
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0., 1.)
    return adv_image

  def _split_block(self, upper_left, lower_right, block_size):
    """Split an image into blocks with block size. Note that a block consists of 
    [upper_left, lower_right, channel]
    
    Args:
      upper_left: [x, y], the coordinate of the upper left of an image
      lower_right: [x, y], the coordinate of the lower right of an image
      block_size: int, the size of a block

    Returns:
      blocks: list, the set of blocks
    """
    blocks = []
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):
      for c in range(3):
        blocks.append([[x, y], [x+block_size, y+block_size], c])
    return blocks
  
  def perturb(self, image, label, index, sess):
    """Perturb an image with (or without) target label
    
    Args:
      image: numpy array with size [1, 299, 299, 3], original image
      label: numpy array with size [1], the label of image (or target label)
      index: int, the index of image
      sess: tensorflow session
       
    Returns:
      adv_image: numpy array with size [1, 299, 299, 3], adversarial image
      num_queries: int, query count
      success: boolean, True if attack is successful
    """
    # Set random seed by index for the reproducibility
    np.random.seed(index)

    # Class variables
    self.width = image.shape[1]
    self.height = image.shape[2]

    # Local variables
    adv_image = np.copy(image)
    loss = 0.
    num_queries = 0
    block_size = self.block_size
    upper_left = [0, 0]
    lower_right = [256, 256]

    # Split image into blocks   
    blocks = self._split_block(upper_left, lower_right, block_size) 

    # Initialize noise
    noise = -self.epsilon*np.ones([1, 256, 256, 3], dtype=np.float32)
    
    # Prepare for batch
    num_blocks = len(blocks)
    batch_size = self.batch_size if self.batch_size > 0 else num_blocks
    curr_order = np.random.permutation(num_blocks)

    np.save('./out/orig_image.npy', image)

    values, indices = self.lazy_local_search.initial_top_k(image, label, sess)

    tf.logging.info("orig top 5 classes: {} / {} / {} / {} / {}".format(
        label_to_name(indices[0,0]),
        label_to_name(indices[0,1]),
        label_to_name(indices[0,2]),
        label_to_name(indices[0,3]),
        label_to_name(indices[0,4])))
        
    tf.logging.info("orig top 5 probs: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
        values[0,0], values[0,1], values[0,2], values[0,3], values[0,4]))

    orig_image = Image.fromarray(np.ndarray.astype(image[0, :, :, :]*255, np.uint8))
    orig_image.save('./out/orig_image.jpg')

    adv_image = self._perturb_image(image, noise)
    
    values, indices = self.lazy_local_search.initial_top_k(adv_image, label, sess)
    
    tf.logging.info("initial top 5 classes: {} / {} / {} / {} / {}".format(
        label_to_name(indices[0,0]),
        label_to_name(indices[0,1]),
        label_to_name(indices[0,2]),
        label_to_name(indices[0,3]),
        label_to_name(indices[0,4])))
        
    tf.logging.info("initial top 5 probs: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
        values[0,0], values[0,1], values[0,2], values[0,3], values[0,4]))
    
    np.save('./out/noise_init.npy', noise)
    adv_image = Image.fromarray(np.ndarray.astype(adv_image[0, :, :, :]*255, np.uint8))
    adv_image.save('./out/adv_image_init.jpg')
    adv_noise = Image.fromarray(np.ndarray.astype(noise[0, :, :, :]*255, np.uint8))
    adv_noise.save('./out/adv_noise_init.jpg')

    # Main loop
    while True:
      # Run batch
      num_batches = int(math.ceil(num_blocks/batch_size))
      for i in range(num_batches):
        # Construct a mini-batch
        bstart = i*batch_size
        bend = min(bstart + batch_size, num_blocks)
        blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]
        # Run lazy local search
        noise, queries, loss, success, top_k = self.lazy_local_search.perturb(
          image, noise, label, sess, blocks_batch)
        num_queries += queries
        values, indices = top_k
        tf.logging.info("Block size: {}, batch: {}, loss: {:.4f}, num queries: {}".format(
          block_size, i, loss, num_queries))
        tf.logging.info("top 5 classes: {} / {} / {} / {} / {}".format(
            label_to_name(indices[0,0]),
            label_to_name(indices[0,1]),
            label_to_name(indices[0,2]),
            label_to_name(indices[0,3]),
            label_to_name(indices[0,4])))
            
        tf.logging.info("top 5 probs: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
            values[0,0], values[0,1], values[0,2], values[0,3], values[0,4]))
        # If query count exceeds max queries, then return False
        if num_queries > self.max_queries:
          return adv_image, num_queries, False
        # Update adversarial image
        adv_image = self._perturb_image(image, noise)
        # If success, return True
        if success:
          np.save('./out/noise_{}.npy'.format(block_size), noise)
          adv_img = Image.fromarray(np.ndarray.astype(adv_image[0, :, :, :]*255, np.uint8))
          adv_img.save('./out/adv_image_{}.jpg'.format(block_size))
          adv_noise = Image.fromarray(np.ndarray.astype(noise[0, :, :, :]*255, np.uint8))
          adv_noise.save('./out/adv_noise_{}.jpg'.format(block_size))
          return adv_image, num_queries, True

      np.save('./out/noise_{}.npy'.format(block_size), noise)
    
      adv_image = Image.fromarray(np.ndarray.astype(adv_image[0, :, :, :]*255, np.uint8))
      adv_image.save('./out/adv_image_{}.jpg'.format(block_size))
      adv_noise = Image.fromarray(np.ndarray.astype(noise[0, :, :, :]*255, np.uint8))
      adv_noise.save('./out/adv_noise_{}.jpg'.format(block_size))
    
      # If block size >= 2, then split blocks
      if not self.no_hier and block_size >= 2:
        block_size //= 2
        blocks = self._split_block(upper_left, lower_right, block_size)
        num_blocks = len(blocks)
        batch_size = self.batch_size if self.batch_size > 0 else num_blocks
        curr_order = np.random.permutation(num_blocks)
      # Otherwise, shuffle the order of batches
      else:
        curr_order = np.random.permutation(num_blocks)
