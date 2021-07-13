from absl import flags, logging, app
#from tensorflow_gan.examples.esrgan import train_lib
#from tensorflow_gan.examples.esrgan import data_provider
import train_lib
import data_provider


flags.DEFINE_integer('batch_size', 32,
                     'The number of images in each batch.')
flags.DEFINE_string('model_dir', '/content/',
                    'Directory to save and load trained models')
flags.DEFINE_boolean('phase_1', False,
                     'Whether Phase 1 training is done or not')
flags.DEFINE_boolean('phase_2', False,
                     'Whether Phase 2 training is done or not')
flags.DEFINE_integer('hr_dimension', 256,
                     'Dimension of a HR image.')
flags.DEFINE_string('data_dir', '/content/datasets',
                    'Directory to save DIV2K dataset.')
flags.DEFINE_integer('print_steps', 1000,
                     'Steps at which values are displayed during training.')
flags.DEFINE_integer('total_steps', 600000,
                     'The maximum number of steps for training.')
flags.DEFINE_integer('decay_steps', 200000,
                     'Step at which learning rate is modified.')
flags.DEFINE_float('decay_factor', 0.2,
                   'Factor by which learning rate is modified.')
flags.DEFINE_float('lr', 0.0002,
                   'Value of initial learning rate for phase 1 generator')
flags.DEFINE_float('beta_1', 0.9,
                   'Optimizer parameters')
flags.DEFINE_float('beta_2', 0.99,
                   'Optimizer parameters')
flags.DEFINE_float('init_lr', 0.00005,
                   'Value of initial learning rate for phase 2 generator')
flags.DEFINE_string('loss_type', 'L1',
                    'L1 or L2 loss while computing perceptual loss.')
flags.DEFINE_float('lamda_', 0.005,
                   'Constant while computing generator loss')
flags.DEFINE_float('eta', 0.02,
                   'Constant while computing generator loss')
flags.DEFINE_string('image_dir', 'L1',
                    'Directory to save images generated during training.')


FLAGS = flags.FLAGS

def main(_):
  hparams = train_lib.HParams(FLAGS.batch_size, FLAGS.model_dir,
                              FLAGS.phase_1, FLAGS.phase_2,
                              FLAGS.hr_dimension, FLAGS.data_dir,
                              FLAGS.print_steps, FLAGS.total_steps,
                              FLAGS.decay_steps, FLAGS.decay_factor,
                              FLAGS.lr, FLAGS.beta_1,
                              FLAGS.beta_2, FLAGS.init_lr,
                              FLAGS.loss_type, FLAGS.lamda_,
                              FLAGS.eta, FLAGS.image_dir)

  data = data_provider.get_div2k_data(hparams, mode='train')
  train_lib.pretrain_generator(hparams, data)
  train_lib.train_esrgan(hparams, data)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
