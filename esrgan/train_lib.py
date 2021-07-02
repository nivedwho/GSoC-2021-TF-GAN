#Train
import tensorflow as tf
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
    metric = tf.keras.metrics.Mean()
    psnr_metric = tf.keras.metrics.Mean()

    if HParams.phase_1: 
        generator = tf.keras.load_model(HParams.model_dir + '/Phase_1/generator/')
    else:
        generator = ESRGAN_G()
    
    G_optimizer = _get_optimizer()
    
    def _train_step(image_lr, image_hr):
        with tf.GradientTape() as tape:
    
            fake = generator(image_lr)
            
            gen_loss = pixel_loss(image_hr, fake) * (1.0 / HParams.batch_size)
            psnr = tf.reduce_mean(tf.image.psnr(fake, image_hr,max_val = 256.0))
            
            gradient = tape.gradient(loss, generator.trainable_variables)
            G_optimizer.apply_gradients(zip(gradient, generator.trainable_variables))
            
            return psnr, gen_loss
    
    step = 0
    for lr, hr in data.take(HParams.total_steps):
        step += 1
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)
        
        psnr, gen_loss = _train_step(lr, hr)
        
        metric(gen_loss)
        psnr_metric(psnr)

        if step % HParams.print_steps == 0:
            logging.info("Step: {}\tGenerator Loss: {}\tPSNR: {}".format(
                step, metric.result(), psnr_metric.result()))

        if step % HParams.decay_steps == 0:
            G_optimizer.learning_rate.assign(G_optimizer.learning_rate * HParams.decay_factor)

    os.makedirs(HParams.model_dir + '/Phase_1/generator', exist_ok = True)
    generator.save(HParams.model_dir + '/Phase_1/generator')

def train_esrgan(HParams, data):
    optimizer = _get_optimizer()

    if HParams.phase_2:
        generator = tf.keras.models.load_model(HParams.model_dir + 'Phase_2/generator/')
        discriminator = tf.keras.models.load_model(HParams.model_dir + 'Phase_2/discriminator/')
    else:
        try: 
            generator = tf.keras.models.load_model(HParams.model_dir + '/Phase_1/generator')
        except:
            raise FileNotFoundError('Pre-trained Generator model not found! Please check the Phase_1 folder under the model directory.')
    
        discriminator = ESRGAN_D()
    
    G_optimizer = optimizer
    G_optimizer.learning_rate.assign(0.00001)
    D_optimizer = optimizer

    ra_gen = RealitivisticAverageLoss(discriminator, type_ = "G")
    ra_disc = RealitivisticAverageLoss(discriminator, type_ = "D")
    perceptual_loss = PerceptualLoss(
        weight = "imagenet",
        input_shape = [HParams.hr_size, HParams.hr_size, 3],
        loss_type = 'L1'
    )

    gen_metric = tf.keras.metrics.Mean()
    disc_metric = tf.keras.metrics.Mean()
    psnr_metric = tf.keras.metrics.Mean()

    def train_step(image_lr, image_hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen = generator(image_lr)
            pred = gen
            
            fake = preprocess_input(fake)
            image_lr = preprocess_input(image_lr)
            image_hr = preprocess_input(image_hr)

            percep_loss = tf.reduce_mean(perceptual_loss(image_hr, fake)) 
            l1_loss = pixel_loss(image_hr, fake) 
            loss_RaG = ra_gen(image_hr, fake) 
            
            disc_loss = ra_disc(image_hr, fake)

            gen_loss = percep_loss + HParams.lambda_ * loss_RaG + HParams.eta * l1_loss

            gen_loss = gen_loss * (1.0 / HParams.batch_size)
            disc_loss = disc_loss * (1.0 / HParams.batch_size)

            disc_metric(disc_loss) 
            gen_metric(gen_loss)

            psnr_metric(tf.reduce_mean(tf.image.psnr(fake, image_hr, max_val = 256.0)))
            
            disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            D_optimizer.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

            gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables) 
            G_optimizer.apply_gradients(zip(gen_grad, generator.trainable_variables))

            return pred
    
    step = 0
    decay_list = [50000, 100000, 200000, 300000]

    for lr, hr in data.take(HParams.steps):
        step += 1
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)
  
        fake = train_step(lr, hr)

        if step % HParams.print_steps == 0:
            logging.info("Step: {}\tGenerator Loss: {}\tDiscriminator: {}\tPSNR: {}".format(
                step, gen_metric.result(), disc_metric.result(), psnr_metric.result()))
        
        if step >= decay_list[0]: 
            G_optimizer.learning_rate.assign(
              G_optimizer.learning_rate * 0.5)
    
            D_optimizer.learning_rate.assign(
                D_optimizer.learning_rate * 0.5)
            
            decay_list.pop(0)
        
def _get_optimizer():
    return tf.optimizers.Adam(
        learning_rate = 0.0002,
        beta_1 = 0.9,
        beta_2 = 0.99
    )