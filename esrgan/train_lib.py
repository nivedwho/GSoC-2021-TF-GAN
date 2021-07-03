
"""Train the ESRGAN model
See https://arxiv.org/abs/1809.00219 for details about the model. 
"""

import tensorflow as tf
import tensorflow_gan as tfgan
import collections, os
from absl import logging
from networks import ESRGAN_G, ESRGAN_D
from losses import pixel_loss, PerceptualLoss, RealitivisticAverageLoss
from utils import preprocess_input

HParams = collections.namedtuple('HParams', [
    'batch_size','model_dir',
    'phase_1', 'phase_2',
    'hr_dimension','data_dir',
    'manual', 'print_steps',
    'total_steps', 'decay_steps',
    'decay_factor', 'lr', 
    'beta_1','beta_2', 
    'init_lr','loss_type', 
    'lambda_','eta', 
    'checkpoint_path'])

def warmup_generator(HParams, data):
    """ Pre-trains the generator network with pixel-loss as proposed in the paper and
        saves the network inside the model directory specified.  
    
    Args:
        HParams : Training parameters as proposed in the paper. 
        data : Dataset consisting of LR and HR image pairs.
    """

    #Stores mean L1 values and PSNR values obtained during training.
    metric = tf.keras.metrics.Mean()
    psnr_metric = tf.keras.metrics.Mean()

    #If phase_1 training is done and needs to be continued, load the generator model. 
    if HParams.phase_1: 
        generator = tf.keras.load_model(HParams.model_dir + '/Phase_1/generator/')
    #If pre-trained generator model is not available, start training from the beginning.
    else:
        generator = ESRGAN_G()
    
    G_optimizer = _get_optimizer()
    
    def _train_step(image_lr, image_hr):
        """ Calculates the L1 Loss and gradients at each step, and updates the gradient which results in 
            the imporvement of PSNR values. 
        Args : 
            image_lr : batch of tensors representing LR images. 
            image_hr : batch of tensors representing HR images.

        Returns : 
            PSNR values and generator loss obtained in each step. 
        """
        with tf.GradientTape() as tape:
    
            fake = generator(image_lr)
            
            gen_loss = pixel_loss(image_hr, fake) * (1.0 / HParams.batch_size)
            psnr = get_psnr(hr, fake)
            
            gradient = tape.gradient(gen_loss, generator.trainable_variables)
            G_optimizer.apply_gradients(zip(gradient, generator.trainable_variables))
            
            return psnr, gen_loss
    
    step = 0
    for lr, hr in data.take(HParams.total_steps):
        step += 1
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)
        
        psnr, gen_loss = _train_step(lr, hr)
        
        #Calculate the mean loss and PSNR values obtained during training.
        metric(gen_loss)
        psnr_metric(psnr)

        if step % HParams.print_steps == 0:
            logging.info("Step: {}\tGenerator Loss: {}\tPSNR: {}".format(
                step, metric.result(), psnr_metric.result()))

        #Modify the learning rate as mentioned in the paper.
        if step % HParams.decay_steps == 0:
            G_optimizer.learning_rate.assign(G_optimizer.learning_rate * HParams.decay_factor)

    #Save the generator model inside model_dir, which is then used in phase_2 of training.
    os.makedirs(HParams.model_dir + '/Phase_1/generator', exist_ok = True)
    generator.save(HParams.model_dir + '/Phase_1/generator')
    logging.info("Saved pre-trained generator network succesfully!")

def train_esrgan(HParams, data):
    """Loads the pre-trained generator model and trains the ESRGAN network using L1 Loss, Perceptual loss and 
    RaGAN loss function.

    Args:  
        HParams : Training parameters as proposed in the paper. 
        data : Dataset consisting of LR and HR image pairs.
    """
    #If the phase 2 training is done and for re-training the ESRGAN model load the saved generator and discriminator networks.
    if HParams.phase_2:
        generator = tf.keras.models.load_model(HParams.model_dir + 'Phase_2/generator/')
        discriminator = tf.keras.models.load_model(HParams.model_dir + 'Phase_2/discriminator/')
    #If Phase 2 training is not done, then load the pre-trained generator model.
    else:
        try: 
            generator = tf.keras.models.load_model(HParams.model_dir + '/Phase_1/generator')
        except:
            raise FileNotFoundError('Pre-trained Generator model not found! Please check the Phase_1 folder under the model directory.')
    
        discriminator = ESRGAN_D()
    
    #Generator learning rate is set as 1 x 10^-4. 
    G_optimizer = _get_optimizer(lr = 0.0001)
    D_optimizer = _get_optimizer()

    #Define RaGAN loss for generator and discriminator networks.
    ra_gen = RealitivisticAverageLoss(discriminator, type_ = "G")
    ra_disc = RealitivisticAverageLoss(discriminator, type_ = "D")
    
    #Define the Perceptual loss function and pass 'imagenet' as the weight for the VGG-19 network.
    perceptual_loss = PerceptualLoss(
        weight = "imagenet",
        input_shape = [HParams.hr_size, HParams.hr_size, 3],
        loss_type = 'L1'
    )

    gen_metric = tf.keras.metrics.Mean()
    disc_metric = tf.keras.metrics.Mean()
    psnr_metric = tf.keras.metrics.Mean()

    def train_step(image_lr, image_hr):
        """ Calculates the L1 Loss, Perceptual loss and RaGAN loss, to train both generator and discriminator networks 
            of the ESRGAN model.
        Args : 
            image_lr : batch of tensors representing LR images. 
            image_hr : batch of tensors representing HR images.

        Returns : 
            PSNR values, generator loss and discriminator obtained in each step. 
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen = generator(image_lr)
            
            fake = preprocess_input(gen)
            image_lr = preprocess_input(image_lr)
            image_hr = preprocess_input(image_hr)

            percep_loss = tf.reduce_mean(perceptual_loss(image_hr, fake)) 
            l1_loss = pixel_loss(image_hr, fake) 
            loss_RaG = ra_gen(image_hr, fake) 
            
            disc_loss = ra_disc(image_hr, fake)
            gen_loss = percep_loss + HParams.lambda_ * loss_RaG + HParams.eta * l1_loss

            gen_loss = gen_loss * (1.0 / HParams.batch_size)
            disc_loss = disc_loss * (1.0 / HParams.batch_size)
            
            psnr = get_psnr(image_hr, fake)

            disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            D_optimizer.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

            gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables) 
            G_optimizer.apply_gradients(zip(gen_grad, generator.trainable_variables))

            return gen_loss, disc_loss, psnr
    
    step = 0
    #Modify learning rate at each of these steps
    decay_list = [50000, 100000, 200000, 300000]
    for lr, hr in data.take(HParams.steps):
        step += 1
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)
  
        gen_loss, disc_loss, psnr = train_step(lr, hr)

        gen_metric(gen_loss)
        disc_metric(disc_loss)
        psnr_metric(psnr)

        if step % HParams.print_steps == 0:
            logging.info("Step: {}\tGenerator Loss: {}\tDiscriminator: {}\tPSNR: {}".format(
                step, gen_metric.result(), disc_metric.result(), psnr_metric.result()))
        
        #Modify the learning rate as mentioned in the paper.
        if step >= decay_list[0]: 
            G_optimizer.learning_rate.assign(
              G_optimizer.learning_rate * 0.5)
    
            D_optimizer.learning_rate.assign(
                D_optimizer.learning_rate * 0.5)
            
            decay_list.pop(0)
        
def _get_optimizer(lr = 0.0002):
    """Returns the Adam optimizer with the specified learning rate."""
    return tf.optimizers.Adam(
        learning_rate = lr,
        beta_1 = 0.9,
        beta_2 = 0.99
    )