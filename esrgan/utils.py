import os
import tensorflow as tf
import tensorflow_gan as tfgan
from absl import logging
import PIL
import numpy as np

# Utility functions for data processing
def scale(lr_img, hr_img, hr_crop_size, scale):
    """Crops each HR image to hr_crops_size and LR image to hr_crop_size/scale
    
    Args : 
        lr_img : Tensors representing LR images present in the DIV2K dataset.
        hr_img : Tensors representing HR images present in the DIV2K dataset.
        hr_crop_size : Size to which HR images are resized. 
        scale : Integer representing the upsampling factor used to calculate LR image size. 
    
    Returns : 
        Uniformly scaled tensors of LR and HR images present in the dataset. 
    """
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_scaled = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_scaled = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_scaled, hr_img_scaled

def random_flip(lr_img, hr_img):
    """Randomly flips LR and HR images for data augmentation."""
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))

def random_rotate(lr_img, hr_img):
    """Randomly rotates LR and HR images for data augmentation."""
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)

def preprocess_input(image):
    """Preprocessing of images done before calculating loss functions during training."""
    image = image[..., ::-1]
    mean = -tf.constant([103.939, 116.779, 123.68])
    return tf.nn.bias_add(image, mean)

# Utility functions for training
def visualize_results(image_lr, 
                      generated, 
                      image_hr, 
                      image_dir = '',
                      step = 0, 
                      train = True):
    
    """ Creates an image grid using Tf-GAN's image grid function and 
        saves the results as a .png image.
    
    Args:
        image_lr : batch of tensors representing LR images.
        generated : batch of tensors representing generated images.
        image_hr : batch of tensors representing HR images.
        image_dir : Directory to save the results.
        step : Number of steps completed, for naming purposes. 
        train : Training or Validation.   
    """
    # Resizing all images to 256x256 just for visualization
    size = 256
    resized_lr = tf.image.resize(image_lr, [size, size], method=tf.image.ResizeMethod.BILINEAR)
    resized_gen = tf.image.resize(generated, [size, size], method=tf.image.ResizeMethod.BILINEAR)
    resized_hr = tf.image.resize(image_hr, [size, size], method=tf.image.ResizeMethod.BILINEAR)

    # Stack an image from the batch of LR, Generated and HR so that image grid can display result in this order.
    stack = tf.stack([resized_lr[0], resized_gen[0], resized_hr[0]])
    
    # Generate an image grid using tf-gan's image grid function.
    image_grid = tfgan.eval.python_image_grid(resized_lr[:3], grid_shape=(1, 3))
    result = PIL.Image.fromarray(image_grid.astype(np.uint8))
    
    if train:
        os.makedirs(image_dir + 'training_results', exist_ok= True)
        result.save(image_dir + 'training_results/' + 'step_{}.png'.format(step))
    else: 
        os.makedirs(image_dir + 'validation_results', exist_ok= True)
        result.save(image_dir + 'validation_results/' + 'step_{}.png'.format(step))


def network_interpolation(alpha = 0.2,
                          phase_1_path = None,
                          phase_2_path = None):
    """ Network interpolation as explained in section 3.4 in the paper, that basically balances
    the effect of PSNR oriented methods and GAN based methods. 
    Args:
        alpha : interpolation parameter. 
        phase_1_path : path to the network saved after phase 1 training. 
        phase_2_path : path to the network saved after phase 2 training.
    Returns: 
        Interpolated generator network.  
    
    """
    psnr_generator = tf.keras.model.load_model(phase_1_path)
    gan_generator = tf.keras.models.load_model(phase_2_path)

    for variables_1, variables_2 in zip(gan_generator.trainable_variables, psnr_generator.trainable_variables):
        variables_1.assign((1 - alpha) * variables_2 + alpha * variables_1)

    return gan_generator

# Utility functions for evaluation
def get_frechet_inception_distance(real_images, generated_images, batch_size,
                                   num_inception_images):
  """Get Frechet Inception Distance between real and generated images.
  
  Args:
    real_images: Real images minibatch.
    generated_images: Generated images minibatch.
    batch_size: Batch dimension.
    num_inception_images: Number of images to run through Inception at once.
  
  Returns:
    Frechet Inception distance. A floating-point scalar.
  
  Raises:
    ValueError: If the minibatch size is known at graph construction time, and
      doesn't batch `batch_size`.
  """
  # Validate input dimensions.
  real_images.shape[0:1].assert_is_compatible_with([batch_size])
  generated_images.shape[0:1].assert_is_compatible_with([batch_size])

  # Resize input images.
  size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
  resized_real_images = tf.image.resize(
      real_images, [size, size], method=tf.image.ResizeMethod.BILINEAR)
  resized_generated_images = tf.image.resize(
      generated_images, [size, size], method=tf.image.ResizeMethod.BILINEAR)

  # Compute Frechet Inception Distance.
  num_batches = batch_size // num_inception_images
  fid = tfgan.eval.frechet_inception_distance(
      resized_real_images, resized_generated_images, num_batches=num_batches)

  return fid


def get_inception_scores(images, batch_size, num_inception_images):
    """ Calculate Inception score for images. 

    Args:
        images : A batch of tensors representing series of images.
        batch_size : Integer representing batch dimension.
        num_inception_images : Number of images to run through Inception at once.
    
    Returns:
        Inception scores : A tensor consisting of scores for each batch of images. 
    
    Raises:
        ValueError: If `batch_size` is incompatible with the first dimension of
        `images`.
        ValueError: If `batch_size` isn't divisible by `num_inception_images`.
    """

    # Validate inputs.
    images.shape[0:1].assert_is_compatible_with([batch_size])
    if batch_size % num_inception_images != 0:
        raise ValueError('`batch_size` must be divisible by `num_inception_images`.')
    
    # Resize images.
    size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
    resized_images = tf.image.resize(
        images, [size, size], method=tf.image.ResizeMethod.BILINEAR)

    # Run images through Inception.
    num_batches = batch_size // num_inception_images
    inc_score = tfgan.eval.inception_score(
        resized_images, num_batches=num_batches)

    return inc_score

def get_psnr(real, generated):
    """Calculate PSNR values for the given samples of images.

    Args: 
        real: batch of tensors representing real images.
        generated : batch of tensors representing generated images. 
    
    Returns:
        PSNR value for the given batch of real and generated images.
    """
    return tf.reduce_mean(tf.image.psnr(generated, real, max_val = 256.0))
