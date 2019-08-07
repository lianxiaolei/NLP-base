"""
Main graph
"""

import tensorflow as tf
from hbconfig import Config

from .attention import positional_encoding
from .decoder import Decoder
from .encoder import Encoder


class Graph(object):
  """

  """

  def __init__(self, mode, dtype=tf.float32):
    self.mode = mode
    self.dtype = dtype

  def build_embed(self, inputs, encoder=True, reuse=False):
    """
    Build encoder and decoder inputs with embedding,
      and build positional encoded used by add(lookup_embedding, lookup_123...n.
    Args:
      inputs:
      encoder:
      reuse:

    Returns:

    """
    with tf.variable_scope("embeddings", reuse=reuse, dtype=self.dtype) as scope:
      embedding_encoder = tf.get_variable(
        'en_emb', shape=[Config.data.source_vocab_size, Config.model.model_dim], dtype=self.dtype)
      embedding_decoder = tf.get_variable(
        'de_emb', shape=[Config.data.target_vocab_size, Config.model.model_dim], dtype=self.dtype)

    with tf.variable_scope('positional-encoding'):
      positional_encoded = positional_encoding(Config.model.model_dim,
                                               Config.data.max_seq_length,
                                               dtype=self.dtype)

    # TODO: Question1: Where to use?

    # [1,2,...n,1,2,...n...]
    position_inputs = tf.tile(tf.range(0, Config.data.max_seq_length), [self.batch_size])
    # Reshape position_inputs to [batch_size, max_seq_length]
    position_inputs = tf.reshape(position_inputs, [self.batch_size, Config.data.max_seq_length])

    if encoder:
      embedding_inputs = embedding_encoder
    else:
      embedding_inputs = embedding_decoder
    # TODO: Answer1: This is used at getting positional code to be added with embedding inputs.
    encoded_inputs = tf.add(tf.nn.embedding_lookup(embedding_inputs, inputs),
                            tf.nn.embedding_lookup(positional_encoded, position_inputs))

    return tf.nn.dropout(encoded_inputs, 1. - Config.model.dropout)

  def build_encoder(self, encoder_emb_input, reuse=False):
    """

    Args:
      encoder_emb_inp:
      reuse:

    Returns:

    """
    with tf.variable_scope("Encoder", reuse=reuse):
      encoder = Encoder(num_layers=Config.model.num_layers,
                        num_heads=Config.model.num_heads,
                        linear_key_dim=Config.model.linear_key_dim,
                        linear_value_dim=Config.model.linear_value_dim,
                        model_dim=Config.model.model_dim,
                        ffn_dim=Config.model.ffn_dim)

      return encoder.build(encoder_emb_input)

  def build_decoder(self, decoder_emb_input, encoder_outputs, reuse=False):
    """

    Args:
      decoder_emb_inp:
      encoder_outputs:
      reuse:

    Returns:

    """
    with tf.variable_scope("Decoder", reuse=reuse):
      decoder = Decoder(num_layers=Config.model.num_layers,
                        num_heads=Config.model.num_heads,
                        linear_key_dim=Config.model.linear_key_dim,
                        linear_value_dim=Config.model.linear_value_dim,
                        model_dim=Config.model.model_dim,
                        ffn_dim=Config.model.ffn_dim)

      return decoder.build(decoder_emb_input, encoder_outputs)

  def build_output(self, decoder_outputs, reuse=False):
    # decoder_outputs has shape [batch, heads, seq, seq]
    with tf.variable_scope('output', reuse=reuse):
      logits = tf.layers.dense(decoder_outputs, Config.data.target_vocab_size)

    # TODO: Answer2: Why the param of argmax is logits[0]?
    self.train_predictions = tf.argmax(logits[0], axis=1, name='train/pred_0')
    return logits

  def _filled_next_token(self, inputs, logits, decoder_index):
    """
    Fill next token with current max prob of logits.
    Args:
      inputs:
      logits:
      decoder_index:

    Returns:

    """
    # tf.identity(tf.argmax(logits[0], axis=1, output_type=tf.int32),
    #             f'test/pred_{decoder_index}')

    next_token = tf.slice(tf.argmax(logits, axis=2),
                          begin=[0, decoder_index - 1],
                          size=[self.batch_size, 1])
    left_zero_pads = tf.zeros([self.batch_size, decoder_index], dtype=tf.int32)
    right_zero_pads = tf.zeros([self.batch_size, (Config.data.max_seq_length - decoder_index - 1)], dtype=tf.int32)
    next_token = tf.concat((left_zero_pads, next_token, right_zero_pads), axis=1)

    return inputs + next_token

  def build(self, encoder_inputs, decoder_inputs):
    """
    Build the transformer model.
    Args:
      encoder_inputs:
      decoder_inputs:

    Returns:

    """
    self.batch_size = tf.shape(encoder_inputs)[0]

    encoder_emb_inputs = self.build_embed(encoder_inputs, encoder=True)
    self.encoder_outputs = self.build_encoder(encoder_emb_inputs)

    decoder_emb_inputs = self.build_embed(decoder_inputs, encoder=False, reuse=True)
    decoder_outputs = self.build_decoder(decoder_emb_inputs, self.encoder_outputs)
    output = self.build_output(decoder_outputs)

    if self.mode == tf.estimator.ModeKeys.TRAIN:
      # TODO: Question 3: Why axis is 2?
      # TODO: Answer 3: the dataset shape probabily is [batch, seq_len, words]
      predictions = tf.argmax(output, axis=2)
      return output, predictions
    else:
      next_decoder_inputs = self._filled_next_token(decoder_inputs, output, 1)
      for i in range(2, Config.data.max_seq_length):
        decoder_emb_inputs = self.build_embed(next_decoder_inputs, encoder=False, reuse=True)
        decoder_outputs = self.build_decoder(decoder_emb_inputs, self.encoder_outputs, reuse=True)
        next_output = self.build_output(decoder_outputs, reuse=True)

        next_decoder_inputs = self._filled_next_token(next_decoder_inputs, next_output, i)

      # slice start_token
      decoder_input_start_1 = tf.slice(next_decoder_inputs, [0, 1],
                                       [self.batch_size, Config.data.max_seq_length - 1])
      predictions = tf.concat(
        [decoder_input_start_1, tf.zeros([self.batch_size, 1], dtype=tf.int32)], axis=1)
      return next_output, predictions
