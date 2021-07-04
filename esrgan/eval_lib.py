import tensorflow as tf
from absl import logging
import utils 
import collections

HParams = collections.namedtuple('HParams', [
    'batch_size', 'num_steps',
    'num_images', 'image_dir',
    'eval_real_images', 'num_inception_images'])

def evaluate(hparams, generator, data):
    """ Runs an evaluation loop and calculates the mean FID, Inception and PSNR scores observed
        on the validation dataset.
    
    Args:
        hparams: Parameters for evaluation.
        generator : The trained generator network.
        data : Validation DIV2K dataset.
    """
    fid_metric = tf.keras.metrics.Mean()
    inc_metric = tf.keras.metrics.Mean()
    psnr_metric = tf.keras.metrics.Mean()
    step = 0
    for lr, hr in data.take(hparams.num_steps):
        step += 1
        # Generate fake images for evaluating the model
        gen = generator(lr)

        if step % hparams.num_steps//10:
            utils.visualize_results(lr, gen, hr, 
                                image_dir = hparams.image_dir, 
                                step = step)

        # Compute Frechet Inception Distance.
        fid_score = utils.get_frechet_inception_distance(hr, 
                                                   gen, 
                                                   hparams.batch_size,
                                                   hparams.num_images)
        fid_metric(fid_score)

        # Compute Inception Scores.
        if hparams.eval_real_images: 
            inc_score = utils.get_inception_scores(hr,
                                             hparams.batch_size,
                                             hparams.num_inception_images)
        else:
            inc_score = utils.get_inception_scores(gen,
                                             hparams.batch_size,
                                             hparams.num_inception_images)
        inc_metric(inc_score)
        
        # Compute PSNR values. 
        psnr = utils.get_psnr(hr, gen, hparams.batch_size)
        psnr_metric(psnr)
        
    logging.info('FID Score :{}\tInception Score :{}\tPSNR value{}'
                .format(fid_metric.result(), inc_metric.result(), psnr_metric.result()))
