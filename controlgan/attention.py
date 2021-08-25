import tensorflow as tf
from networks import Conv

def conv1x1(filters, bias=False):
    return Conv(num_filters=filters, kernel_size=1,
                stride=1, padding=0, use_bias=bias)

def func_attention(context, query, gamma1=4.0):
  batch_size, queryL = query.shape[0], query.shape[1] 
  ih, iw = context.shape[1], context.shape[2]
  sourceL = ih * iw 

  context = tf.reshape(context, [batch_size, sourceL, -1]) 
  context = tf.transpose(context, perm=[0, 2, 1])
  attn = tf.matmul(query, context) 
  attn = tf.reshape(attn, [batch_size * sourceL, queryL])
  attn = tf.nn.softmax(attn)

  attn = tf.reshape(attn, [batch_size, sourceL, queryL])
  attn = tf.transpose(attn, perm=[0, 2, 1])
  attn = tf.reshape(attn, [batch_size*queryL, sourceL])

  attn = attn * gamma1
  attn = tf.nn.softmax(attn)
  attn = tf.reshape(attn, [batch_size, queryL, sourceL])

  attn = tf.transpose(attn, perm=[0, 2, 1])

  weightedContext = tf.matmul(context, attn) 

  return weightedContext, tf.reshape(tf.transpose(attn, [0, 2, 1]), [batch_size, ih, iw, queryL])


class SpatialAttention(tf.keras.layers.Layer):
  def __init__(self, idf):
    super(SpatialAttention, self).__init__(name="SpatialAttention")
    self.conv_context = conv1x1(idf)
    self.sm = tf.keras.layers.Softmax()
    self.mask = None
    self.idf = idf

  def applyMask(self, mask):
    self.mask = mask
  
  def call(self, inputs, training=True):
    x, sentence, context, mask = inputs

    ih, iw = x.shape[1], x.shape[2]
    queryL = ih * iw 
    batch_size, sourceL = context.shape[0], context.shape[1]

    x = tf.reshape(x, shape=[batch_size, queryL, -1])

    context = tf.expand_dims(context, axis=1)
    context = self.conv_context(context)
    context = tf.squeeze(context, axis=1)

    attn = tf.matmul(x, context, transpose_b=True)
    attn = tf.reshape(attn, shape=[batch_size * queryL, sourceL])
    
    if self.mask is not None:
      mask = tf.tile(mask, multiples=[queryL, 1])
      attn = tf.where(tf.equal(mask, True), x=tf.constant(-float('inf'), dtype=tf.float32, shape=mask.shape), y=attn)
    
    attn = self.sm(attn)
    attn = tf.reshape(attn, shape=[batch_size, queryL, sourceL])

    weightedContext = tf.matmul(context, attn, transpose_a=True, transpose_b=True)
    weightedContext = tf.reshape(tf.transpose(weightedContext, perm=[0, 2, 1]), shape=[batch_size, ih, iw, -1])
    word_attn = tf.reshape(attn, shape=[batch_size, ih, iw, -1])

    return weightedContext, word_attn

class ChannelAttention(tf.keras.layers.Layer):
  def __init__(self, idf, name='ChannelAttention'):
    super(ChannelAttention, self).__init__(name=name)
    self.word_conv = conv1x1(idf)
    self.idf = idf
    
  def call(self, inputs, training=True):
    weighted_context, word_emb = inputs 

    batch_size, ih, iw, context_ch = weighted_context.shape
    seq_len = word_emb.shape[1] 

    word_emb = tf.expand_dims(word_emb, axis=1)
    word_emb = self.word_conv(word_emb)
    word_emb = tf.squeeze(word_emb, axis=1)

    weighted_context = tf.reshape(weighted_context, [batch_size, ih * iw, -1])
    attn_c = tf.matmul(weighted_context, word_emb, transpose_a=True, transpose_b=True) 
    attn_c = tf.reshape(attn_c, [batch_size * context_ch, seq_len])
    attn_c = tf.nn.softmax(attn_c)

    attn_c = tf.reshape(attn_c, [batch_size, context_ch, seq_len])
    weightedContext_c = tf.matmul(word_emb, attn_c, transpose_a=True, transpose_b=True)
    weightedContext_c = tf.reshape(weightedContext_c, [batch_size, ih, iw, context_ch])

    return weightedContext_c, attn_c
    
