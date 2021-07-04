from absl import flags
from absl import logging

import tensorflow as tf
import train_lib, data_provider, utils

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')
flags.DEFINE_string('model_dir', '/content/', 'Directory where the models for evaluation are stored.')
flags.DEFINE_string('data_dir', '/content/datasets', 'Directory where the models for evaluation are stored.')
flags.DEFINE_integer('num_steps', 1000, 'The number of steps for evaluation.')
flags.DEFINE_integer('num_inception_images', 32 , 'The number of images passed for evaluation at each step.')
flags.DEFINE_string('image_dir', '/content/results', 'Directory to save images generated during evaluation.')
flags.DEFINE_boolean('eval_real_images', False, 'Whether Phase 1 training is done or not')


FLAGS = flags.FLAGS

def main():
    hparams = eval_lib.HParams(FLAGS.batch_size, 
                               FLAGS.num_steps, FLAGS.num_inception_images, 
                               FLAGS.image_dir, FLAGS.eval_real_images)
    generator = tf.keras.models.load_model(FLAGS.mode_dir + '/Phase_2/interpolated_generator')
    data = data_provider.get_div2k_data(data_dir = FLAGS.data_dir, mode = 'valid')
    eval_lib.evaluate(hparams, generator, data)
    