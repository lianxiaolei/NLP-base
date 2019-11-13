# coding:utf8

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class SequenceLabelling(object):
  """
  Sequence labelling model class.
  """

  def __init__(self, flags):
    self.FLAGS = flags

    self.num_classes = self.FLAGS.num_classes
    self.vocab_length = self.FLAGS.vocab_length
    self.word_dim = self.FLAGS.word_dim
    self.rnn_units = self.FLAGS.rnn_units
    self.rnn_layer_num = self.FLAGS.rnn_layer_num
    self.keep_prob = self.FLAGS.keep_prob

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    session_conf = tf.ConfigProto(
      gpu_options=gpu_options,
      allow_soft_placement=True,
      log_device_placement=False
    )
    self.sess = tf.Session(config=session_conf)

    # self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, self.word_dim], name='X')
    self.X = tf.placeholder(dtype=tf.int32, shape=[None, None], name='X')
    self.y = tf.placeholder(dtype=tf.int64, shape=[None, None], name='y')

    self.seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name='seq_len')

  def _lstm_cell(self, reuse=False):
    if reuse:
      tf.get_variable_scope.reuse_variables()
    cell = rnn.LSTMCell(self.rnn_units)
    return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

  def _get_embedding(self):
    self.embedding = tf.get_variable('word_emb', shape=[self.vocab_length, self.word_dim], dtype=tf.float32)

  def _lookup(self, X):
    lookup = tf.nn.embedding_lookup(self.embedding, X + 1)
    return lookup

  def _rnn_units(self, X):
    # Defind rnn base units.
    cell_fw = [self._lstm_cell() for _ in range(self.rnn_layer_num)]
    cell_bw = [self._lstm_cell() for _ in range(self.rnn_layer_num)]

    # Unstack the inputs to list with step items.
    # TODO Why we need to unstack the timestep?
    inputs = tf.unstack(X, self.FLAGS.max_seq_len, axis=1)

    output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw,
                                                          inputs=inputs, dtype=tf.float32)

    # Stack the outputs.
    output = tf.stack(output, axis=1)

    return output

  def _tail(self, X):
    logits = tf.keras.layers.Dense(self.num_classes)(X)
    return logits

  def build_net(self, compile=True):
    self._get_embedding()
    lookup = self._lookup(self.X)

    rnn_outputs = self._rnn_units(lookup)
    self.logits = self._tail(rnn_outputs)

    if compile:
      self._compile(self.logits, self.y)

  def _compile(self, logits, labels):
    labels = tf.cast(labels, tf.int32)

    self.pred = tf.cast(tf.argmax(self.logits, axis=2), tf.int64)
    correct_prediction = tf.equal(self.pred, self.y)
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # seq_mask = tf.sequence_mask(self.seq_len, self.FLAGS.max_seq_len, dtype=tf.int64)
    # accuracy = tf.equal(self.pred * seq_mask, self.y * seq_mask)
    # self.accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(inputs=logits, tag_indices=labels,
                                                                          sequence_lengths=self.seq_len)
    self.loss = tf.reduce_mean(-log_likelihood)

    # Train
    self.train_op = tf.train.AdamOptimizer(self.FLAGS.lr).minimize(self.loss)
