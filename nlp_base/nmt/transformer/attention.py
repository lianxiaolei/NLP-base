"""
Attention calculation
"""

import numpy as np
import tensorflow as tf


def positional_encoding(dim, sequence_length, dtype=tf.float32):
  """
  PE(pos,2i) =sin(pos/10000^(2i/dmodel))
  PE(pos,2i+1) =cos(pos/10000^(2i/dmodel))
  Args:
    dim:
    sequence_length:
    dtype:

  Returns:

  """
  position_mask = np.array([pos / 10000 ** (2 * d) for pos in sequence_length for d in dim])
  position_mask[::2] = np.sin(position_mask[::2])
  position_mask[1::2] = np.cos(position_mask[1::2])
  return position_mask


class Attention(object):
  """
  Attention module contains
  all attention calculation of the paper https://arxiv.org/abs/1706.03762
  """

  def __init__(self, mask=False, num_heads=1, linear_key_dim=50,
               linear_value_dim=50, model_dim=100, drop=0.2):
    assert linear_key_dim % num_heads == 0
    assert linear_value_dim % num_heads == 0

    self.mask = mask
    self.num_heads = num_heads
    self.linear_key_dim = linear_key_dim
    self.linear_value_dim = linear_value_dim
    self.model_dim = model_dim
    self.drop = drop

  def linear_project(self, q, k, v):
    """
    Lineary project of q, k, v with dense.
    Args:
      q: query
      k: key
      v: value

    Returns:

    """
    q = tf.layers.dense(q, units=self.linear_key_dim, use_bias=False)
    k = tf.layers.dense(k, units=self.linear_key_dim, use_bias=False)
    v = tf.layers.dense(v, units=self.linear_value_dim, use_bias=False)

    return q, k, v

  def split_head(self, q, k, v):
    """
    Split the last dim to calculate multi-head-attention.
    Args:
      q: query
      k: key
      v: value

    Returns:

    """

    def split_last_dimension_then_transpose(tensor):
      tensor = tf.reshape(tensor, [-1] + tf.shape(tensor)[1: -1]
                          + [self.num_heads, tf.shape(tensor)[-1] // self.num_heads])
      tensor = tf.transpose(tensor, (0, 2, 1, 3))
      # [batch, heads, seq_len, units/heads]
      return tensor

    qs = split_last_dimension_then_transpose(q)
    ks = split_last_dimension_then_transpose(k)
    vs = split_last_dimension_then_transpose(v)
    return qs, ks, vs

  def concat_heads(self, q, k, v):
    """
    Concat the splited tensor.
    Args:
      q: query
      k: key
      v: value

    Returns:

    """

    def transpose_then_concat_last_two_dimenstion(tensor):
      # [batch_size, max_seq_len, num_heads, dim]
      tensor = tf.transpose(tensor, 0, 2, 1, 3)
      tensor_shape = tf.shape(tensor)
      tensor = tf.reshape(tensor, [-1] + tensor_shape[1: -2] + [tensor_shape[-2] * tensor_shape[-1]])
      return tensor

    q = transpose_then_concat_last_two_dimenstion(q)
    k = transpose_then_concat_last_two_dimenstion(k)
    v = transpose_then_concat_last_two_dimenstion(v)
    return q, k, v

  def multi_head_attention(self, q, k, v):
    qp, kp, vp = self.linear_project(q, k, v)
    qs, ks, vs = self.split_head(qp, kp, vp)
    outputs = self.scaled_dot_product(qs, ks, vs)
    output = self.concat_heads(outputs)
    output = tf.layers.dense(output, self.model_dim)

    return tf.nn.dropout(output, 1.0 - self.drop)

  def scaled_dot_product(self, q, k, v):
    """
    Scaled-dot-product calculation.
    Args:
      q: query
      k: key
      v: value

    Returns:

    """
    key_dim_per_heads = self.linear_key_dim // self.num_heads
    # matmul
    # [batch, heads, seq_len, hunits] * [batch, heads, seq, hunits].transb
    # = [batch, heads, seq_len, seq_len]
    mat = tf.matmul(q, k, transpose_b=True)
    # scaled
    scaled = mat / (key_dim_per_heads ** 0.5)

    if self.mask:
      # [seq_len, word_emb_units / heads]
      # (batch_size, num_heads, query_dim, key_dim) (1, 1, query_dim, key_dim)
      shape_struct = tf.ones_like(scaled[0, 0, :, :])
      # generate a lower triangle matrix
      # then to make a infinity matrix (seq_len, seq_len)
      # [[1., -., -., -.],
      #  [1., 1., -., -.],
      #  [1., 1., 1., -.]]
      # use to_dense to convert the triangle to dense
      attention_mask = tf.linalg.LinearOperatorLowerTriangular(shape_struct).to_dense()
      # [1, 1,seq_len, seq_len]
      attention_mask = tf.reshape(attention_mask, [1, 1] + tf.shape(attention_mask))
      # [batch, heads, seq_len, seq_len]
      tiled_attention_mask = tf.tile(attention_mask, tf.shape(scaled)[:2] + [1, 1])
      pad = tf.ones_like(tiled_attention_mask) * -1e9
      scaled = tf.where(tf.equal(tiled_attention_mask, 0), pad, scaled)

    softmax_scaled = tf.nn.softmax(scaled)
    bott_mat = tf.matmul(softmax_scaled, v)

    return bott_mat
