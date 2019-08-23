# coding:utf8

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tqdm import tqdm

from nlp_base.info_extra import data_iterate, word2ind, word2ind_with_seqlen, get_w2i_map, \
  word2ind_without_seg, test_data_iterate, data_iterate_with_seqlen


class SequenceLabelling(object):
  """
    Sequence labelling model class.
    """

  def __init__(self):
    self.FLAGS = tf.flags.FLAGS

    self.num_classes = self.FLAGS.num_classes
    self.vocab_length = self.FLAGS.vocab_length
    self.word_dim = self.FLAGS.word_dim
    self.units = self.FLAGS.units
    self.rnn_layer_num = self.FLAGS.rnn_layer_num
    self.keep_prob = self.FLAGS.keep_prob

    self.sess = tf.Session()

    # self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, self.word_dim], name='X')
    self.X = tf.placeholder(dtype=tf.float32, shape=[None, None], name='X')
    self.y = tf.placeholder(dtype=tf.float32, shape=[None, None], name='y')

    self.seq_len = tf.placeholder(dtype=tf.float32, shape=[None], name='seq_len')

  def _lstm_cell(self, reuse=False):
    if reuse:
      tf.get_variable_scope.reuse_variables()
    cell = rnn.LSTMCell(self.units)
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
    inputs = tf.unstack(X, self.FLAGS.MAX_SEQ_LEN, axis=1)

    output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw,
                                                          inputs=inputs, dtype=tf.float32)

    # Stack the outputs.
    output = tf.stack(output, axis=1)

    return output

  def _tail(self, X):
    logits = tf.keras.layers.Dense(self.num_classes)(X)
    return logits

  def _build_net(self, compile=True):
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

    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(inputs=logits, tag_indices=labels,
                                                                          sequence_lengths=self.seq_len)
    self.loss = tf.reduce_mean(-log_likelihood)

    # Train
    self.train_op = tf.train.AdamOptimizer(self.FLAGS.lr).minimize(self.loss)

    # Defind model saver.
    self.saver = tf.train.Saver()

  def train(self, iterator, train_initializer, dev_initializer=None,
            epoch_num=100, step_num=1000, save_when_acc=0.92, save_when_loss=0.1, dev_step=10):
    X, y, seq_len = iterator.get_next()
    with self.sess.as_default() as sess:
      sess.run(tf.initialize_all_variables())
      for epoch in range(epoch_num):
        sess.run(train_initializer)
        bar = tqdm(range(step_num), ncols=100)
        for step in bar:
          try:
            loss, acc, _ = sess.run([self.loss, self.accuracy, self.train_op],
                                    feed_dict={self.X: X,
                                               self.y: y,
                                               self.seq_len: seq_len})
          except tf.errors.OutOfRangeError:
            sess.run(train_initializer)
          bar.set_description_str("Step:{}\t  Loss:{}\t  Acc:{}".format(step, str(loss)[:5], str(acc)[:5]))

        # Dev
        sess.run(dev_initializer)
        bar = tqdm(range(dev_step), ncols=100)
        for step in bar:
          try:
            acc = sess.run(self.accuracy)
          except tf.errors.OutOfRangeError:
            sess.run(dev_initializer)
          bar.set_description_str("Step:{}\tAcc:{}".format(step, acc))

        if loss < save_when_loss:
          self.saver.save(sess, self.FLAGS.check_point, global_step=epoch)
          print('Saved model done.')
