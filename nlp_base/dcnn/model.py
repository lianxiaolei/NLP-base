# coding:utf8

import tensorflow as tf


class DCNN(object):
  def __init__(self, batch_size, sentence_length,
               units, emb_dim, top_k):
    """
    Dynamic CNN neural network class.
    Args:
      batch_size: A int value, the batch size of inputs.
      sentence_length: A int value, max of input sentences' length.
      units: A int list, data channels.
      emb_dim: A int value, word embedding dimension.
      top_k:
      k1:
    """
    self.batch_size = batch_size
    self.sentence_length = sentence_length
    self.units = units
    self.emb_dim = emb_dim
    self.top_k = top_k

  def conv_per_dim(self, x, w, b):
    """
    Convolution for every dim.
    Args:
      x: Input sentences.
      w: Conv kernel.
      b: Conv bias.

    Returns:

    """
    x_unstack = tf.unstack(x, axis=2)
    w_unstack = tf.unstack(w, axis=1)
    b_unstack = tf.unstack(b, axis=-1)
    out = []
    with tf.name_scope('conv_per_dim'):
      for i in range(len(x_unstack)):
        conv = tf.nn.conv1d(x_unstack[i], w_unstack[i], padding='SAME', stride=1)
        conv = tf.add(conv, b_unstack[i])
        conv = tf.nn.relu(conv)
        out.append(conv)
      out = tf.stack(out, axis=2)
    return out

  def k_max_pooling(self, x, k):
    """

    Args:
      x:
      k:

    Returns:

    """
    x_unstack = tf.unstack(x, axis=2)
    out = []
    with tf.name_scope("k_max_pooling"):
      for i in range(len(x_unstack)):
        conv = tf.transpose(x_unstack[i], perm=[0, 2, 1])
        top_k = tf.nn.top_k(conv, k, sorted=False).values
        top_k = tf.transpose(top_k, perm=[0, 2, 1])
        out.append(top_k)
      fold = tf.stack(out, axis=2)
    return fold

  def folding_k_max_pooling(self, x, k):
    """
    
    Args:
      x: 
      k: 

    Returns:

    """
    with tf.name_scope('folding'):
      x_unstack = tf.unstack(x, axis=2)
      out = []
      for i in range(0, len(x_unstack) - 1, 2):
        fold = tf.add(x_unstack[i], x_unstack[i + 1])
        fold = tf.transpose(fold, perm=[0, 2, 1])
        k_max = tf.nn.top_k(fold, k=k).values
        print(k_max)
        fold = tf.transpose(k_max, [0, 2, 1])
        out.append(fold)
      out = tf.stack(out, axis=2)
    return out

  def dense(self, x, w, b, wo, dropout_keep_prob):
    """
    Dense layer.
    Args:
      x: nput tensor.
      w: hidden matmul weights.
      b: hidden bias.
      wo: output matmul weights.
      dropout_keep_prob: dropout's keep prob.

    Returns:

    """
    with tf.name_scope('dense'):
      # w [4,emb,14/2,100]
      h = tf.nn.tanh(tf.matmul(x, w) + b)
      h = tf.nn.dropout(h, dropout_keep_prob)
      o = tf.matmul(h, wo)
    return o

  def DCNN(self, sent, W1, W2, b1, b2, top_k, Wh, bh, Wo, dropout_keep_prob):
    x = self.conv_per_dim(sent, W1, b1)
    x = self.k_max_pooling(x, top_k[0])

    x = self.conv_per_dim(x, W2, b2)
    x = self.folding_k_max_pooling(x, top_k[1])
    # 增加一个int
    # fold_flatten = tf.reshape(fold, [-1, int(top_k * self.emb_dim * self.units[1] / 4)])
    fold_flatten = tf.reshape(x, [-1, int(top_k[1] * self.emb_dim * self.units[1] / 2)])
    # fold_flatten = tf.reshape(fold, [-1, int(top_k*100*14/4)])
    print(fold_flatten.get_shape())
    out = self.dense(fold_flatten, Wh, bh, Wo, dropout_keep_prob)
    return out
