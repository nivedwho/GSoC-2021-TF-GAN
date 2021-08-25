import tensorflow as tf
from keras.models import Model
from tensorflow.keras import Sequential
from keras.layers import Input, Embedding, Dropout, Conv2D
from keras.layers import RNN, LSTMCell, Bidirectional
from keras.layers import UpSampling2D, BatchNormalization, LeakyReLU, Flatten, Dense
from attention import SpatialAttention, ChannelAttention
import utils


class GLU(tf.keras.layers.Layer):
  def __init__(self):
    super(GLU, self).__init__(name='GLU')

  def call(self, x):
    nc = x.shape[-1]
    assert nc % 2 == 0, 'channels dont divide 2!'
    nc = int(nc/2)
    if len(x.shape) == 4:
        return x[:, :, :, :nc] * tf.sigmoid(x[:, :, :, nc:])
    if len(x.shape) == 3:
        return x[:, :, :nc] * tf.sigmoid(x[:, :, nc:])
    if len(x.shape) == 2:
        return x[:, :nc] * tf.sigmoid(x[:, nc:])

class Conv(tf.keras.layers.Layer):
  def __init__(self, channels,
                kernel=3, stride=1,
                pad=0, pad_type='zero',
                use_bias=True,
                name='Conv'):
    super(Conv, self).__init__(name=name)
    self.channels = channels
    self.kernel = kernel
    self.stride = stride
    self.pad = pad
    self.pad_type = pad_type
    self.use_bias = use_bias
    self.conv = tf.keras.layers.Conv2D(filters=self.channels,
                                        kernel_size=self.kernel,
                                        strides=self.stride,
                                        use_bias=self.use_bias,
                                        name=self.name)

  def call(self, x, training=None, mask=None):
    if self.pad > 0:
      x = utils.pad_input(x, self.pad, self.pad_type, self.stride, self.kernel)
    x = self.conv(x)
    return x

def conv1x1(filters, bias=False, name='conv1x1'):
  return Conv(channels=filters, kernel=1,
              stride=1, pad=0, use_bias=bias, name=name)


def conv3x3(filters, name='conv1x1'):
  return Conv(channels=filters, kernel=3, stride=1,
              pad=1, use_bias=False, name=name)
  

def upBlock(out_plane, name):
  block = tf.keras.Sequential([
      UpSampling2D(size=2, interpolation='nearest'),
      conv3x3(out_plane*2, name=name),
      BatchNormalization(momentum=0.9, epsilon=1e-5,
                          center=True, scale=True),
      GLU()])
  return block

def block3x3_leakyRelu(ngf):
  block = []
  block += [Conv(ngf, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv_code')]
  block += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
  block += [tf.keras.layers.LeakyReLU(0.2)]
  return tf.keras.Sequential(block)

class ResBlock(tf.keras.layers.Layer):
  def __init__(self, channel_nums):
      super(ResBlock, self).__init__()
      self.channel_nums = channel_nums
      self.model = self._build()
  def _build(self):
    model = []
    model += [tf.keras.layers.Conv2D(self.channel_nums, kernel_size=3,
                                      strides=1, use_bias=False)]
    model += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True),
          GLU()]
    model += [tf.keras.layers.Conv2D(self.channel_nums, kernel_size=3,
                                      strides=1, padding="same", use_bias=False)]
    model += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
    return tf.keras.Sequential(model)

  def call(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class RNN_Encoder(tf.keras.Model):
  def __init__(self, input_dim=vocab_size, embed_dim=256, dropout_rate=0.5,
                hidden_size=128):
    super(RNN_Encoder, self).__init__(name='RNN_Encoder')
    self.embedding = Embedding(input_dim, embed_dim, embeddings_initializer='uniform')
    self.dropout = Dropout(rate=dropout_rate)

    self.lstm_layer = RNN([LSTMCell(units=hidden_size, dropout=0.5)], 
                          return_sequences=True, return_state=True)
    self.bidirectional = Bidirectional(self.lstm_layer)

  def call(self, x):
    x = self.embedding(x)
    x = self.dropout(x)
    output, fw_state, bw_state = self.bidirectional(x)
    return output, fw_state, bw_state


class CNN_Encoder(tf.keras.Model):
  def __init__(self, embed_dim):
    super(CNN_Encoder, self).__init__(name='CNN_Encoder')
    self.inception_v3_preprocess = tf.keras.applications.inception_v3.preprocess_input

    self.pre_trained_model = tf.keras.applications.InceptionV3(
        input_shape=(299, 299, 3), include_top=False, weights='imagenet')
    for layer in self.pre_trained_model.layers:
        layer.trainable = False

    self.image_features_extract_model = tf.keras.Model(
        inputs=self.pre_trained_model.input, outputs=self.pre_trained_model.get_layer('mixed7').output)
    for layer in self.image_features_extract_model.layers:
        layer.trainable = False

    self.flatten = Flatten()
    self.dense = Dense(embed_dim, use_bias=True)
    self.Conv = Conv2D(filters=embed_dim, kernel_size=3, 
                       strides=1, use_bias=False)

  def preprocess(self, x):
    x = ((x + 1) / 2) * 255.0
    x = tf.image.resize(x, [299, 299], method=tf.image.ResizeMethod.BILINEAR)
    x = self.inception_v3_preprocess(x)
    return x

  def call(self, x):
    x = self.preprocess(x)

    code = self.pre_trained_model(x)
    feature = self.image_features_extract_model(x)

    feature = self.Conv(feature)
    code = self.flatten(code)
    code = self.dense(code)

    return feature, code

class CA_Net(tf.keras.Model):
  def __init__(self, c_dim=100):
    super(CA_Net, self).__init__(name='conditional_augmentation')
    self.dim = c_dim
  
  def fc(self, x):
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(self.dim * 2, use_bias = True)(x)
    x = tf.keras.layers.ReLU()(x)
    return x

  def encode(self, text):
    mean = text[:, :self.dim]
    logvar = text[:, self.dim:]
    return mean, logvar

  def reparametrize(self, mean, logvar): 
    return tf.exp(logvar * 0.5) * tf.random.normal(tf.shape(mean)) + mean

  def call(self, sent_emb):
    sent_emb = self.fc(sent_emb)
    mean, logvar = self.encode(sent_emb)
    c_code = self.reparametrize(mean, logvar)
    return c_code, mean, logvar

def fully_connected(units, use_bias=True, sn=False):
  x = tf.keras.layers.Flatten()
  return tf.keras.layers.Dense(units, use_bias=use_bias)


class INIT_STAGE_G(tf.keras.layers.Layer):
  def __init__(self, ngf, name='Generator_64'):
    super(INIT_STAGE_G, self).__init__(name=name)
    self.gf = ngf

    self.model = self.build_model()

  def build_model(self):
    model = []

    model += [fully_connected(units=self.gf*4*4*2, use_bias=False)]
    model += [batch_norm()]
    model += [GLU()]
    model += [tf.keras.layers.Reshape(target_shape=[4, 4, self.gf])]

    model += [upBlock(self.gf // 2, name='up_block_1')]
    model += [upBlock(self.gf // 4, name='up_block_2')]
    model += [upBlock(self.gf // 8, name='up_block_3')]        
    model += [upBlock(self.gf // 16, name='up_block_4')]
      
    return Sequential(model)

  def call(self, c_z_code, training=True, mask=None):
    h_code = self.model(c_z_code, training=training)

    return h_code

class NEXT_STAGE_G(tf.keras.layers.Layer):
  def __init__(self, ngf, scale, name='Generator_128'):
    super(NEXT_STAGE_G, self).__init__(name=name)
    self.gf_dim = ngf
    self.scale = scale
    self.att = SpatialAttention(channels=self.gf_dim)
    self.channel_att = ChannelAttention(channels=self.scale * self.scale)

    self.model = self.build_model()

  def build_model(self):
    model = []
    for i in range(2):
        model += [ResBlock(self.gf_dim * 3, name='resblock_' + str(i))]
    
    model += [upBlock(self.gf_dim, name='up_block')]
    model = Sequential(model)
    
    return model

  def call(self, inputs, training=True):
    h_code, c_code, word_emb, mask = inputs
    c_code, att = self.att([h_code, c_code, word_emb, mask])
    c_code_channel, att_channel = self.channel_att([c_code, word_emb])

    h_c_code = tf.concat([h_code, c_code, c_code_channel], axis=-1)

    h_code = self.model(h_c_code, training=training)

    return c_code, h_code

class GET_IMAGE_G(tf.keras.layers.Layer):
  def __init__(self, ngf):
    super(GET_IMAGE_G, self).__init__()
    self.gf_dim = ngf
    self.model = self._build()

  def _build(self):
    model = []
    model += [Conv(channels=3, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='g_64_logit')]
    model += [tf.keras.layers.Activation(tf.keras.activations.tanh)]
    return tf.keras.Sequential(model)

  def call(self, h_code):
    out_img = self.model(h_code)
    return out_img

class Generator(tf.keras.Model):
  def __init__(self, ngf, name='Generator'):
    super(Generator, self).__init__(name=name)
    self.ngf = ngf
    self.h_net1 = INIT_STAGE_G(self.ngf * 16)
    self.img_net1 = GET_IMAGE_G(self.ngf)

    self.h_net2 = NEXT_STAGE_G(self.ngf, scale=64)
    self.img_net2 = GET_IMAGE_G(self.ngf)

    self.h_net3 = NEXT_STAGE_G(self.ngf, scale=128)
    self.img_net3 = GET_IMAGE_G(self.ngf)

  def call(self, inputs, training=True, mask=None):
    sent_emb, z_code, word_embs, mask = inputs
    
    fake_imgs = []
    att_maps = []

    c_code, mu, logvar = conditional_augmentation(sent_emb)
    c_z_code = tf.concat([c_code, z_code], axis=-1)

    h_code1 = self.h_net1(c_z_code, training=training)
    fake_img1 = self.img_net1(h_code1)
    fake_imgs.append(fake_img1)

    c_code, h_code2 = self.h_net2([h_code1, c_code, word_embs, mask], training=training)
    fake_img2 = self.img_net2(h_code2)
    fake_imgs.append(fake_img2)

    c_code, h_code3 = self.h_net3([h_code2, c_code, word_embs, mask], training=training)
    fake_img3 = self.img_net3(h_code3)
    fake_imgs.append(fake_img3)

    return fake_imgs


def block3x3_leakyRelu(ngf):
  block = []
  block += [Conv(ngf, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, name='conv_code')]
  block += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
  block += [tf.keras.layers.LeakyReLU(0.2)]
  return tf.keras.Sequential(block)

def downBlock(ngf):
  block = []
  block += [Conv(ngf, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv_code')]
  block += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
  block += [tf.keras.layers.Activation(tf.keras.layers.LeakyReLU(0.2))]
  return tf.keras.Sequential(block)

def encode_image_by_16times(ndf):
  block = []
  block += [Conv(ndf, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='convv')]
  block += [tf.keras.layers.LeakyReLU(0.2)]
  
  block += [Conv(ndf * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv_1')]
  block += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
  block += [tf.keras.layers.LeakyReLU(0.2)]

  block += [Conv(ndf * 4, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv_2')]
  block += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
  block += [tf.keras.layers.LeakyReLU(0.2)]
 
  block += [Conv(ndf * 8, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, name='conv_3')]
  block += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
  block += [tf.keras.layers.LeakyReLU(0.2)]

  return tf.keras.Sequential(block)


class D_GET_LOGITS(tf.keras.layers.Layer):
  def __init__(self, ndf, embed_dim):
    super(D_GET_LOGITS, self).__init__()
    self.df_dim = ndf
    self.embed_dim = embed_dim
    self.conv_block_1 = block3x3_leakyRelu(ndf)
    self.conv_block_2 = block3x3_leakyRelu(ndf)
    self.conv_block_3 = block3x3_leakyRelu(ndf)

    self.outlogits_1 = Conv(channels = 1, kernel=4, stride=4, use_bias=True, name='lol')
    self.outlogits_2 = Conv(channels = 1, kernel=4, stride=4, use_bias=True, name='lol')
    self.outlogits_3 = Conv(channels = 1, kernel=4, stride=4, use_bias=True, name='lol')

    self.outlogits_4 = Conv(channels = 1, kernel=4, stride=4, use_bias=True, name='lol')
    self.outlogits_5 = Conv(channels = 1, kernel=4, stride=4, use_bias=True, name='lol')
    self.outlogits_6 = Conv(channels = 1, kernel=4, stride=4, use_bias=True, name='lol')

  def call(self, x_64, x_128, x_256, sent_emb):
    sent_emb = tf.reshape(sent_emb, shape=[-1, 1, 1, self.embed_dim])
    sent_emb = tf.tile(sent_emb, multiples=[1, 4, 4, 1])
    
    h_64 =  tf.concat([x_64, sent_emb], axis=-1)
    h_128, h_256  = tf.concat([x_128, sent_emb], axis=-1),  tf.concat([x_256, sent_emb], axis=-1)

    h_c_64, h_c_128, h_c_256 = self.conv_block_1(h_64), self.conv_block_2(h_128), self.conv_block_3(h_256)

    
    x_64_uc_logit, x_128_uc_logit, x_256_uc_logit = self.outlogits_1(x_64), self.outlogits_2(x_128), self.outlogits_3(x_256)
    x_64_c_logit, x_128_c_logit, x_256_c_logit = self.outlogits_4(h_c_64), self.outlogits_5(h_c_128), self.outlogits_6(h_c_256)

    uncond_logits = [x_64_uc_logit, x_128_uc_logit, x_256_uc_logit]
    cond_logits = [x_64_c_logit, x_128_c_logit, x_256_c_logit]

    return uncond_logits, cond_logits
  
class D_NET64(tf.keras.layers.Layer):
  def __init__(self, ndf):
    super(D_NET64, self).__init__()
    self.ndf = ndf
    self.img_code_s16 = encode_image_by_16times(self.ndf)
  
  def call(self, x):
    x_code4 = self.img_code_s16(x, training=True)
    return x_code4

class D_NET128(tf.keras.layers.Layer):
  def __init__(self, ndf):
    super(D_NET128, self).__init__()
    self.ndf = ndf
    self.img_code_s16 = encode_image_by_16times(ndf)
    self.img_code_s32 = downBlock(ndf * 16)
    self.img_code_s32_1 = block3x3_leakyRelu(ndf * 16)
  
  def call(self, x_var):
    x_code8 = self.img_code_s16(x_var, training=True)
    x_code4 = self.img_code_s32(x_code8, training=True)
    x_code4 = self.img_code_s32_1(x_code4, training=True)
    return x_code4

class D_NET256(tf.keras.layers.Layer):
  def __init__(self, ndf):
    super(D_NET256, self).__init__()
    self.ndf =  ndf
    self.img_code_s16 = encode_image_by_16times(ndf)
    self.img_code_s32 = downBlock(ndf * 16)
    self.img_code_s64 = downBlock(ndf * 32)
    self.img_code_s64_1 = block3x3_leakyRelu(ndf * 16)
    self.img_code_s64_2 = block3x3_leakyRelu(ndf * 8)
  
  def call(self, x_var):
    x_code16 = self.img_code_s16(x_var, training=True)
    x_code8 = self.img_code_s32(x_code16, training=True)
    x_code4 = self.img_code_s64(x_code8, training=True)
    x_code4 = self.img_code_s64_1(x_code4, training=True)
    x_code4 = self.img_code_s64_2(x_code4, training=True)
    return x_code4

class Discriminator(tf.keras.Model):
  def __init__(self, ndf = 64, embed_dim = 256):
    super(Discriminator, self).__init__()
    self.ndf = ndf  
    self.embed_dim = embed_dim
    self.d_64 = D_NET64(self.ndf)
    self.d_128 = D_NET128(self.ndf)
    self.d_256 = D_NET256(self.ndf)
    self.get_logits = D_GET_LOGITS(self.ndf, self.embed_dim)

  def call(self, inputs):
    x_64, x_128, x_256, sent_emb = inputs
    x_64, x_128, x_256 = self.d_64(x_64), self.d_128(x_128), self.d_256(x_256)
    uncond_logits, cond_logits = self.get_logits(x_64, x_128, x_256, sent_emb)
    return uncond_logits, cond_logits
