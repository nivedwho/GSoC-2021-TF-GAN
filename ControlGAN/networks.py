import tensorflow as tf

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
      
def conv1x1(x, num_filters):
  return tf.keras.layers.Conv2D(filters=num_filters, kernel_size=1, 
                                stride=1, padding="same", use_bias=True)

def conv3x3(x, num_filters):
  return tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, 
                                stride=1, padding="same", use_bias=True)

def upBlock(out_plane):
  model = []
  model += [tf.keras.layers.UpSampling2D(size = 2, interpolation = 'nearest')]
  model += [conv3x3(out_plane * 2)]
  model += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
  model += [GLU()]
  model = tf.keras.Sequential(model)
  return model

def fully_connected(units, use_bias=True, sn=False):
  x = tf.keras.layers.Flatten()(x)
  return tf.keras.layers.Dense(units, use_bias=use_bias)

def batch_norm(momentum=0.9, epsilon=1e-5):
  return tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon,
                                            center=True, scale=True)

class ResBlock(tf.keras.layers.Layer):
  def __init__(self, channel_num):
    super(ResBlock, self).__init__()
    self.channel_nums = channel_num
    self.block = self._build()

  def _build(self):
    model = []
    model += [tf.keras.layers.Conv2D(self.channel_nums, kernel_size=3,
                                     strides=1, use_bias=False)]
    model += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, 
                                                 center=True, scale=True)]
    model += [GLU()]
    model += [tf.keras.layers.Conv2D(self.channel_nums, kernel_size=3,
                                     strides=1, padding="same", use_bias=False)]
    model += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, 
                                                 center=True, scale=True)]
    return tf.keras.Sequential(model)

  def call(self, x):
    residual = x
    out = self.block(x)
    out += residual
    return out

class RNN_Encoder(tf.keras.Model):
  def __init__(self, input_dim = vocab_size, embed_dim = 256, dropout_rate = 0.5,
               hidden_size = 128 ):
    super(RNN_Encoder, self).__init__(name = 'RNN_Encoder')
    self.embedding = tf.keras.layers.Embedding(input_dim, embed_dim, embeddings_initializer = 'uniform')
    self.dropout = tf.keras.layers.Dropout(rate = dropout_rate) 
    self.lstm_layer = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(units = hidden_size, dropout=0.5)], return_sequences = True, return_state= True)
    self.bidirectional = tf.keras.layers.Bidirectional(self.lstm_layer)
  
  def call(self, x):
    x = self.embedding(x)
    x = self.dropout(x)
    output, fw_state, bw_state = self.bidirectional(x)
    return output, fw_state, bw_state

class CNN_Encoder(tf.keras.Model):
  def __init__(self, embed_dim):    
    super(CNN_Encoder, self).__init__(name= 'CNN_Encoder')
    self.inception_v3_preprocess = tf.keras.applications.inception_v3.preprocess_input

    self.pre_trained_model =  tf.keras.applications.InceptionV3(input_shape = (299, 299, 3), include_top = False, weights = 'imagenet')
    for layer in self.pre_trained_model.layers:
      layer.trainable = False

    self.image_features_extract_model = tf.keras.Model(inputs = self.pre_trained_model.input, outputs = self.pre_trained_model.get_layer('mixed7').output)  
    for layer in self.image_features_extract_model.layers:
      layer.trainable = False

    self.flatten = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(embed_dim, use_bias = True)
    self.Conv = tf.keras.layers.Conv2D(filters = embed_dim, kernel_size= 3, strides= 1, use_bias= False)

  def call(self, x):
    x = self.inception_v3_preprocess(x)

    code = self.pre_trained_model(x)
    feature = self.image_features_extract_model(x)

    feature = self.Conv(feature)
    code = self.flatten(code)
    code = self.dense(code)

    return feature, code

class CA_NET(tf.keras.Model):
  def __init__(self, c_dim, name='CA_NET'):
    super(CA_NET, self).__init__(name=name)
    self.c_dim = c_dim
    self.model = self._build()

  def reparametrize(self, mean, logvar): 
    return tf.exp(logvar * 0.5) * tf.random.normal(tf.shape(mean)) + mean
 
  def encode(self, text):
    mean = text[:, :c_dim]
    logvar = text[:, c_dim:]
    return mean, logvar

  def build(self):
    model = []
    model += [tf.keras.layers.FullyConnected(units=self.c_dim * 2, name='mu_fc')]
    model += [tf.keras.layers.Relu()]
    model = tf.keras.Model.Sequential(model)

    return model
  
  def call(self, sent_emb, training=True, mask=None):
    x = self.model(sent_emb, training=training)

    mean, logvar = encode(sent_emb)
    c_code = reparametrize(mean, logvar)

    return c_code, mean, logvar
    
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
  block += [conv3x3(ngf)]
  block += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
  block += [tf.keras.layers.LeakyReLU(0.2)]
  return tf.keras.Model.Sequential(block)

def downBlock(ngf):
  block = []
  block += [tf.keras.layers.Conv2D(ngf, kernel_size= 4, strides = 2, padding="same", use_bias=False)]
  block += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
  block += [tf.keras.layers.Activation(tf.keras.layers.LeakyReLU(0.2))]
  return tf.keras.Model.Sequential(block)

def encode_image_by_16times(ndf):
  block = []
  block += [tf.keras.layers.Conv2D(ndf, kernel_size=4, strides=2, padding="same", use_bias=False)]
  block += [tf.keras.layers.LeakyReLU(0.01)]
  
  block += [tf.keras.layers.Conv2D(ndf * 2, kernel_size=3, strides=1, padding="same", use_bias=False)]
  block += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
  block += [tf.keras.layers.LeakyReLU(0.2)]

  block += [tf.keras.layers.Conv2D(ndf * 4, kernel_size=3, strides=1, padding="same", use_bias=False)]
  block += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
  block += [tf.keras.layers.LeakyReLU(0.2)]
 
  block += [tf.keras.layers.Conv2D(ndf * 8, kernel_size=3, strides=1, padding="same", use_bias=False)]
  block += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, center=True, scale=True)]
  block += [tf.keras.layers.LeakyReLU(0.2)]

  return tf.keras.Model.Sequential(block)

class D_GET_LOGITS(tf.keras.layers.Layer):
  def __init__(self, ndf, embed_dim):
    super(D_GET_LOGITS, self).__init__()
    self.df_dim = ndf
    self.embed_dim = embed_dim
    self.outlogits = tf.keras.layers.Sequential([
        tf.keras.layers.Conv2D(1, kernel_size=4, stride=4, use_bias=True)
        tf.keras.activations.sigmoid()])
    
  def call(self, x_64, x_128, x_256, sent_emb):
    sent_emb = tf.reshape(sent_emb, shape=[-1, 1, 1, self.embed_dim])
    sent_emb = tf.tile(sent_emb, multiples=[1, 4, 4, 1])
    
    h_64, h_128, h_256 =  tf.concat([x_64, sent_emb], axis=-1),  tf.concat([x_128, sent_emb], axis=-1),  tf.concat([x_256, sent_emb], axis=-1)
    h_c_64, h_c_128, h_c_256 = block3x3_leakyRelu(h_64), block3x3_leakyRelu(h_64), block3x3_leakyRelu(h_64)
    
    x_64_uc_logit, x_128_uc_logit, x_256_uc_logit = self.outlogits(x_64), self.outlogits(x_128), self.outlogits(x_256)
    x_64_c_logit, x_128_c_logit, x_256_c_logit = self.outlogits(h_c_64), self.outlogits(h_c_128), self.outlogits(h_c_256)

    uncond_logits = [x_64_uc_logit, x_128_uc_logit, x_256_uc_logit]
    cond_logits = [x_64_c_logit, x_128_c_logit, x_256_c_logit]

    return uncond_logits, cond_logits
  
 
class D_NET64(tf.keras.layers.Layer):
  def __init__(self, ndf):
    super(D_NET64, self).__init__()
    self.ndf = ndf
    self.img_code_s16 = encode_image_by16times(self.ndf)
  
  def call(self, x):
    x_code4 = self.img_code_s16(x_var, training=True)
    return x_code4

class D_NET128(tf.keras.layers.Layer):
  def __init__(self, ndf):
    super(D_NET128, self).__init__()
    self.ndf = ndf
    self.img_code_s16 = encode_image_by16times(ndf)
    self.img_code_s32 = downBlock(ndf * 16)
    self.img_code_s32_1 = block3x3_leakyRelu(ngf * 16)
  
  def call(self, x_var):
    x_code8 = self.img_code_s16(x_var, training=True)
    x_code4 = self.img_code_s32(x_code8, training=True)
    x_code4 = self.img_code_s32_1(x_code4, training=True)
    return x_code4

class D_NET256(tf.keras.layers.Layer):
  def __init__(self, ndf):
    self.ndf =  ndf
    self.img_code_s16 = encode_image_by_16times(ndf)
    self.img_code_s32 = downBlock(ndf * 16)
    self.img_code_s64 = downBlock(ndf * 32)
    self.img_code_s64_1 = Block3x3_leakRelu(ndf * 16)
    self.img_code_s64_2 = Block3x3_leakRelu(ndf * 8)
  
  def call(self, x_var):
    x_code16 = self.img_code_s16(x_var, training=True)
    x_code8 = self.img_code_s32(x_code16, training=True)
    x_code4 = self.img_code_s64(x_code8, training=True)
    x_code4 = self.img_code_s64_1(x_code4, training=True)
    x_code4 = self.img_code_s64_2(x_code4, training=True)
    return x_code4
 
class Discriminator(tf.keras.Model):
  def __init(self, ndf, embed_dim):
    super(Discriminator, self).__init()
    self.ndf = channels  
    sefl.embed_dim = embed_dim
    self.d_64 = D_NET64(self.ndf)
    self.d_128 = D_NET128(self.ndf)
    self.d_256 = D_NET256(self.ndf)
    self.get_logits = D_GET_LOGITS(self.ndf, self.embed_dim)

  def call(self, inputs):
    x_64, x_128, x_256, sent_emb = inputs
    x_64, x_128, x_256 = self.d_64(x_64), self.d_128(x_128), self.d_256(x_256)
    uncond_logits, cond_logits = self.get_logits(x_64, x_128, x_256, sent_emb)
    return uncond_logits, cond_logits
