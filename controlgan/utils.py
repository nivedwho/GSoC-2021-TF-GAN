import tensorflow as tf
import numpy as np
from attention import func_attention

def cosine_similarity(x, y):
  xy = tf.reduce_sum(x * y, axis=-1)
  x = tf.norm(x, axis=-1)
  y = tf.norm(y, axis=-1)

  similarity = (xy / ((x * y) + 1e-8))

  return similarity

def word_loss(img_features, words_emb, class_id, gamma2=5.0):
  masks = []
  attn_maps = []
  similarities = []
  words_num = words_emb.shape[1]

  batch_size = words_emb.shape[0]
  label = tf.cast(range(batch_size), tf.int32)

  for i in range(batch_size):
    mask = (class_id.numpy() == class_id[i].numpy()).astype(np.uint8)
    mask[i] = 0
    masks.append(mask.reshape((1, -1)))

    word = tf.expand_dims(words_emb[i, :, :], axis=0)
    word = tf.tile(word, multiples=[batch_size, 1, 1])

    context = img_features

    weiContext, attn = func_attention(context, word)
    attn = tf.expand_dims(attn[i], axis=0)
    attn_maps.append(attn)

    weiContext = tf.transpose(weiContext, perm=[0, 2, 1])

    word = tf.reshape(word, shape=[batch_size * words_num, -1])
    weiContext = tf.reshape(weiContext, shape=[batch_size * words_num, -1])

    row_sim = cosine_similarity(word, weiContext)
    row_sim = tf.reshape(row_sim, shape=[batch_size, words_num])

    row_sim = tf.exp(row_sim * gamma2)
    row_sim = tf.reduce_sum(row_sim, axis=-1, keepdims=True)
    row_sim = tf.math.log(row_sim)

    similarities.append(row_sim)

  similarities = tf.concat(similarities, axis=-1)
  masks = tf.cast(tf.concat(masks, axis=0), tf.float32)

  similarities = similarities * gamma2

  similarities = tf.where(tf.equal(masks, True), x=tf.constant(
      -float('inf'), dtype=tf.float32, shape=masks.shape), y=similarities)

  similarities1 = tf.transpose(similarities, perm=[1, 0])

  loss0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=similarities, labels=label))
  loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=similarities1, labels=label))

  loss = loss0 + loss1

  return loss

def sent_loss(img_feature, sent_emb, class_id, gamma3=10.0):
  batch_size = sent_emb.shape[0]
  label = tf.cast(range(batch_size), tf.int32)

  masks = []

  for i in range(batch_size):
    mask = (class_id.numpy() == class_id[i].numpy()).astype(np.uint8)
    mask[i] = 0
    masks.append(np.reshape(mask, newshape=[1, -1]))

  masks = tf.cast(tf.concat(masks, axis=0), tf.float32)

  cnn_code = tf.expand_dims(img_feature, axis=0)
  rnn_code = tf.expand_dims(sent_emb, axis=0)

  cnn_code_norm = tf.norm(cnn_code, axis=-1, keepdims=True)
  rnn_code_norm = tf.norm(rnn_code, axis=-1, keepdims=True)

  scores0 = tf.matmul(cnn_code, rnn_code, transpose_b=True)
  norm0 = tf.matmul(cnn_code_norm, rnn_code_norm, transpose_b=True)
  scores0 = scores0 / \
      tf.clip_by_value(norm0, clip_value_min=1e-8,
                        clip_value_max=float('inf')) * gamma3

  scores0 = tf.squeeze(scores0, axis=0)

  scores0 = tf.where(tf.equal(masks, True), x=tf.constant(-float('inf'),
                      dtype=tf.float32, shape=masks.shape), y=scores0)
  scores1 = tf.transpose(scores0, perm=[1, 0])

  loss0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=scores0, labels=label))
  loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=scores1, labels=label))

  loss = loss0 + loss1
  return loss


def pad_input(x, pad, pad_type, stride, kernel):
  h = x.shape[1]
  if h % stride == 0:
    pad = pad * 2
  else:
    pad = max(kernel - (h % stride), 0)

  if pad_type == 'reflect':
    padded_input = tf.pad(x, [[0, 0], [pad // 2, pad - pad // 2], 
                          [pad // 2,  pad - pad // 2], [0, 0]], mode='REFLECT')
  else:
    padded_input = tf.pad(x, [[0, 0], [[pad // 2, pad - pad // 2], 
                          [pad // 2,  pad - pad // 2], [0, 0]]])
  return padded_input
