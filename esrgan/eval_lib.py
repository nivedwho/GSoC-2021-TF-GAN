import tensorflow as tf
from absl import logging
from utils import get_frechet_inception_distance, get_inception_scores, get_psnr

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
    
    for lr, hr in data.take(hparams.num_steps):
        # Generate fake images for evaluating the model
        gen = generator(lr)

        # Compute Frechet Inception Distance.
        fid_score = get_frechet_inception_distance(hr, 
                                                   gen, 
                                                   hparams.batch_size,
                                                   hparams.num_images)
        fid_metric(fid_score)

        # Compute Inception Scores.
        if hparams.eval_real_images: 
            inc_score = get_inception_scores(hr,
                                             hparams.batch_size,
                                             hparams.num_inception_images)
        else:
            inc_score = get_inception_scores(gen,
                                             hparams.batch_size,
                                             hparams.num_inception_images)
        inc_metric(inc_score)
        
        # Compute PSNR values. 
        psnr = get_psnr(hr, gen, hparams.batch_size)
        psnr_metric(psnr)
        
    logging.info('FID Score :{}\tInception Score :{}\tPSNR value{}'
                .format(fid_metric.result(), inc_metric.result(), psnr_metric.result()))
