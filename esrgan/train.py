import collections 
import tensorflow as tf
import model, utils, dataset, losses
from functools import partial
import os
import time
from absl import logging

HParams = collections.namedtuple('HParams', [
    'batch_size','model_dir',
    'hr_dimension','data_dir',
    'strategy',
    'manual', 'total_steps', 
    'decay_steps', 'decay_factor', 
    'beta_1','beta_2', 
    'init_lr', 'loss_type', 
    'lambda_', 'eta', 
    'checkpoint_path'])


def pretrain_generator(HParams, data):
    strategy = utils.getstrategy()  
    
    generator = model.ESRGAN_G()
    
    metric = tf.keras.metrics.Mean()
    psnr_metric = tf.keras.metrics.Mean()

    data = iter(strategy.experimental_distribute_dataset(data))
 
    G_optimizer = tf.optimizers.Adam(
        learning_rate = 0.0002,
        beta_1 = 0.9,
        beta_2 = 0.99
    )

    checkpoint = tf.train.Checkpoint(
        G = generator,
        G_optimizer = G_optimizer
    )
    
    status = utils.load_checkpoint(checkpoint, "phase_1", HParams.model_dir)
    logging.debug("phase_1 status object: {}".format(status))

    start_time = time.time()

    def _step_fn(image_lr, image_hr):
        logging.debug("Starting Distributed Step")
        with tf.GradientTape() as tape:
            fake = generator(image_lr)

            loss = lossstrategyes.pixel_loss(image_hr, fake) * (1.0 / HParams.batch_size)
        
        psnr_metric(
            tf.reduce_mean(
                tf.image.psnr(
                    fake, image_hr,
                    max_val = 256.0
                )
            )
        )
        gen_vars = list(set(generator.trainable_variables))
  
        gradient = tape.gradient(loss, gen_vars)
        G_optimizer.apply_gradients(zip(gradient, gen_vars))

        mean_loss = metric(loss)
        logging.debug("Ending Distributed Step")
        return tf.cast(G_optimizer.iterations, tf.float32)
    

    def train_step(image_lr, image_hr):
        distributed_metric = strategy.run(_step_fn, args=(image_lr, image_hr))
        
        mean_metric = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, distributed_metric, axis = None
        )

        return mean_metric
    
    while True:
        image_lr, image_hr = next(data)

        num_steps = train_step(image_lr, image_hr)

        if num_steps >= HParams.total_steps:
            return 

        if not num_steps % HParams.decay_steps:
            G_optimizer.learning_rate.assign(
                    G_optimizer.learning_rate * HParams.decay_factor
                )                    

            if psnr_metric.result() > previous_loss:
                utils.save_checkpoint(checkpoint, "phase_1", HParams.model_dir)

            previous_loss = psnr_metric.result()
    
    if status:
        status.assert_consumed()
        logging.info("consumed checkpoint for phase_1 successfully")
        status = None
    
    if not num_steps % HParams.decay_steps:
        logging.debug(
            "Learning Rate: %s" %
            G_optimizer.learning_rate.numpy)
        
        G_optimizer.learning_rate.assign(
                G_optimizer.learning_rate * HParams.decay_factor
            )  

        logging.debug(
            "Decayed Learning Rate by %f."
            "Current Learning Rate %s" % (
                HParams.decay_factor, G_optimizer.learning_rate)
            )
    
    if not num_steps % 1000:
        logging.info(
            "[WARMUP] Step: {}\tGenerator Loss: {}"
            "\tPSNR: {}\tTime Taken: {} sec".format(
                num_steps,
                metric.result(),
                psnr_metric.result(),
                time.time() -
                start_time))

        if psnr_metric.result() > previous_loss:
          utils.save_checkpoint(checkpoint, "phase_1", HParams.model_dir)
        
        previous_loss = psnr_metric.result()
        start_time = time.time()

def train_esrgan(HParams, data):
    strategy = utils.getstrategy()
    
    generator = model.ESRGAN_G()
    discriminator = model.ESRGAN_D()

    optimizer = tf.optimizers.Adam(
        learning_rate = 0.0002,
        beta_1 = 0.9,
        beta_2 = 0.99
    )

    G_optimizer = optimizer
    D_optimizer = optimizer

    ra_gen = losses.RealitivisticAverageLoss(discriminator, type_ = "G")
    ra_disc = losses.RealitivisticAverageLoss(discriminator, type = "D")

    status = None
    checkpoint = tf.train.Checkpoint(
        G = generator,
        G_optimizer = G_optimizer,
        D = discriminator, 
        D_optimizer = D_optimizer
    )

    hot_start = tf.train.Checkpoint(
        G = generator,
        G_optimizer = G_optimizer
    )

    status = utils.load_checkpoint(hot_start, "phase_1", HParams.model_dir)
    G_optimizer.learning_rate.assign(0.0002)


    logging.debug("phase status object: {}".format(status))

    gan_metric = tf.keras.metrics.Mean()
    disc_metric = tf.keras.metrics.Mean()
    psnr_metric = tf.keras.metrics.Mean()
    
    logging.debug("Loading Perceptual Model")

    perceptual_loss = utils.PerceptualLoss(
        weights = "imagenet",
        input_shape = [HParams.hr_dimension, HParams.hr_dimension, 3],
        loss_type = HParams.loss_type
    )

    logging.debug("Loaded Model")

    def _step_fn(image_lr, image_hr):
        logging.debug("Starting Distributed Step")
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake = generator(image_lr)
            fake = utils.preprocess_input(image_lr)

            image_lr = utils.preprocess_input(image_lr)
            image_hr = utils.preprocess_input(image_hr)
            percep_loss = tf.reduce_mean(perceptual_loss(image_hr, fake))
            logging.debug("Calculated Perceptual Loss")

            l1_loss = losses.pixel_loss(image_hr, fake)
            logging.debug("Calculated Pixel Loss")
    
            loss_RaG = ra_gen(image_hr, fake)
            logging.debug("Calculated Relativistic"+"Averate (RA) Loss for Generator")
            
            disc_loss = ra_disc(image_hr, fake)
            logging.debug("Calculated RA Loss Discriminator")
            
            gen_loss = percep_loss + HParams.lambda_ * loss_RaG + HParams.eta * l1_loss
            logging.debug("Calculated Generator Loss")


            disc_metric(disc_loss)
            gan_metric(gen_loss)

            gen_loss = gen_loss * (1.0/ HParams.batch_size)
            disc_loss = disc_loss * (1.0/ HParams.batch_size)

            psnr_metric(
                tf.reduce_mean(
                    tf.image.psnr(
                        fake, image_hr, max_val = 256.0
                        )
                    )
                )
        
        disc_grad = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )
        logging.debug("Calculated gradient for Discriminator")
        
        D_optimizer.apply_gradient(
            zip(disc_grad, discriminator.trainable_variables)
        )
        logging.debug("Applied gradients to Discriminator")

        gen_grad = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        logging.debug("Calculated gradient for Generator")
        
        G_optimizer.apply_gradient(
            zip(gen_grad, generator.trainable_variables)
        )
        logging.debug("Applied gradients to Generator")

        return tf.case(D_optimizer, tf.float32)
        

    def train_step(image_lr, image_hr):
        distributed_iterations = strategy.run(_step_fn, args=(image_lr, image_hr))

        num_steps = strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            distributed_iterations, axis=None)
        
        return num_steps

    last_psnr = 0

    while True:
        imgs_lr, imgs_hr = next(data)
        num_step = train_step(imgs_lr, imgs_hr)

        if num_step >= HParams.total_steps:
            return

        for _step in HParams.decay_steps.copy():
            if num_step >= _step:
                HParams.decay_steps.pop(0)
                
                g_current_lr = strategy.reduce(
                    tf.distribute.ReduceOp.MEAN,
                    G_optimizer.learning_rate, axis = None
                )

                d_current_lr = strategy.reduce(
                    tf.distribute.ReduceOp.MEAN,
                    D_optimizer.learning_rate, axis = None
                )

                G_optimizer.learning_rate.assign(
                    G_optimizer.learning_rate * HParams.decay_factor
                )
                D_optimizer.learning_rate.assign(
                    D_optimizer.learning_rate * HParams.decay_factor
                )