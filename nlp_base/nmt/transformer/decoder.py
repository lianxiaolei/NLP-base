"""
Decoder module
"""

import tensorflow as tf

from seq2seq.transformer.attention import Attention
from seq2seq.transformer.layers import FFN


class Decoder(object):
  """

  """

  def __init__(self, num_layers=8, num_heads=8, linear_key_dim=50,
               linear_value_dim=50, model_dim=50, ffn_dim=50, dropout=0.2):
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.linear_key_dim = linear_key_dim
    self.linear_value_dim = linear_value_dim
    self.model_dim = model_dim
    self.ffn_dim = ffn_dim
    self.dropout = dropout

  def build(self, input, outputs):
    """
    Build encoder module.
    Args:
      input: position-embedding input

    Returns:

    """
    o1 = tf.identity(input)

    for i in range(1, self.num_layers + 1):
      with tf.variable_scope(f'layer-{i}'):
        # Residual connect between input and multi-head attention
        o2 = self._add_and_norm(o1, self._masked_self_attention(q=o1, k=o1, v=o1), num=1)
        o3 = self._add_and_norm(o2, self._encoder_decoder_attention(q=o2, k=outputs, v=outputs), num=2)
        # Residual connect between mid-output and dense of mid-output
        o4 = self._add_and_norm(o3, self._positional_feed_forward(o3), num=3)
        o1 = tf.identity(o4)

        att = self.self_attention(o1, o1, o1)
        # Residual connect between input and multi-head attention
        o2 = self.add_and_norm(input, att, layer_num=1)

        o3 = self.add_and_norm(o2, self.position_feed_forward(o2), layer_num=2)
        o1 = tf.identity(o3)
    return o3

  def _add_and_norm(self, x, output, num):
    with tf.variable_scope(f'add-and-norm-{num}'):
      add = tf.add(x, output)
      norm = tf.contrib.layers.layer_norm(add)
      return norm

  def _positional_feed_forward(self, x):
    with tf.variable_scope('position-feed-forward'):
      ffn = FFN(self.ffn_dim, self.model_dim, dropout=self.dropout)
      return ffn.dense_relu_dense(ffn)

  def _masked_self_attention(self, q, k, v):
    with tf.variable_scope('masked-self-attention'):
      attention = Attention(num_heads=self.num_heads,
                            mask=True,
                            linear_key_dim=self.linear_key_dim,
                            linear_value_dim=self.linear_value_dim,
                            model_dim=self.model_dim,
                            drop=self.dropout)

      return attention.multi_head_attention(q, k, v)

  def _encoder_decoder_attention(self, q, k, v):
    with tf.variable_scope('encoder-decoder-attention'):
      attention = Attention(num_heads=self.num_heads,
                            mask=False,
                            linear_key_dim=self.linear_key_dim,
                            linear_value_dim=self.linear_value_dim,
                            model_dim=self.model_dim,
                            drop=self.dropout)

      return attention.multi_head_attention(q, k, v)
