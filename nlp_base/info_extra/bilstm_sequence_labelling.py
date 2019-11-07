# coding:utf8

import tensorflow as tf

import numpy as np
from tensorflow.contrib import rnn
from tqdm import tqdm
from nlp_base.info_extra import data_iterate, word2ind, word2ind_with_seqlen, get_w2i_map, \
  word2ind_without_seg, test_data_iterate, data_iterate_with_seqlen

MAX_SEQ_LEN = 256

t2i_dict = {'c': 1, 'o': 2, 'b': 3, 'a': 4}

CHECKPOINT = '/home/lian/PycharmProjects/NLP-base/model/checkpoint/seqtag_ckpt'


class SequenceLabelling(object):
  """
  Sequence labelling model class.
  """

  def __init__(self, num_classes, vocab_length, word_dim, units, rnn_layer_num, keep_prob):
    self.num_classes = num_classes
    self.vocab_length = vocab_length
    self.word_dim = word_dim
    self.units = units
    self.rnn_layer_num = rnn_layer_num
    self.keep_prob = keep_prob

    self.sess = tf.Session()

  def _lstm_cell(self, reuse=False):
    if reuse:
      tf.get_variable_scope.reuse_variables()
    cell = rnn.LSTMCell(self.units)
    return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

  def _init_embedding(self):
    self.embedding = tf.Variable(tf.random_normal([self.vocab_length, self.word_dim]), dtype=tf.float32)

  def _lookup(self, X):
    lookup = tf.nn.embedding_lookup(self.embedding, X + 1)
    return lookup

  def _rnn_units(self, X):
    # Defind rnn base units.
    cell_fw = [self._lstm_cell() for _ in range(self.rnn_layer_num)]
    cell_bw = [self._lstm_cell() for _ in range(self.rnn_layer_num)]

    # Unstack the inputs to list with step items.
    # TODO Why we need to unstack the timestep?
    inputs = tf.unstack(X, MAX_SEQ_LEN, axis=1)

    output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw,
                                                          inputs=inputs, dtype=tf.float32)

    # Stack the outputs.
    output = tf.stack(output, axis=1)

    # Reshape output to [batch * times, units * 2]

    # modify01
    # return tf.reshape(output, [-1, self.units * 2])
    return output

  def _tail(self, X):
    logits = tf.keras.layers.Dense(self.num_classes)(X)
    return logits

  def _compile(self, logits, labels):
    # Reshape

    # modify01
    # labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    labels = tf.cast(labels, tf.int32)

    # Loss
    # J = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #   labels=label, logits=tf.cast(logits, tf.float32))
    # self.loss = J

    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(inputs=logits, tag_indices=labels,
                                                                          sequence_lengths=self.seq_len)
    self.loss = tf.reduce_mean(-log_likelihood)

  def build(self, X, y, seq_len, mode='train'):
    self.seq_len = seq_len

    self._init_embedding()
    lookup = self._lookup(X)
    rnn_outputs = self._rnn_units(lookup)
    logits = self._tail(rnn_outputs)

    self._compile(logits, y)

    # Prediction
    # modify01
    # self.pred = tf.cast(tf.argmax(logits, axis=1), tf.int64)
    self.pred = tf.cast(tf.argmax(logits, axis=2), tf.int64)
    # modify01
    # correct_prediction = tf.equal(self.pred, tf.reshape(y, [-1]))
    # modify02-re
    correct_prediction = tf.equal(self.pred, y)
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # seq_mask = tf.sequence_mask(self.seq_len, MAX_SEQ_LEN, dtype=tf.int64)
    # correct_prediction = tf.equal(self.pred * seq_mask, y * seq_mask)

    self.X = X
    self.y = y

    if mode == 'test':
      self.restore()

  def operate(self, lr=1e-3):

    # Train
    self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
    # Defind model saver.
    self.saver = tf.train.Saver()

  def restore(self):
    saver = tf.train.Saver()

    with self.sess.as_default() as sess:
      saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT[:CHECKPOINT.rfind('/')]))
    print('Load model successfully.')

  def inference(self, test_initializer, word_set):
    with self.sess.as_default() as sess:
      sess.run(tf.initialize_all_variables())
      sess.run(train_initializer)
      sess.run(test_initializer)

      for step in range(1):
        x_results, y_predict_results, acc = sess.run([self.X, self.pred, self.accuracy])
        y_predict_results = np.reshape(y_predict_results, x_results.shape)
        for i in range(len(x_results)):
          x_result, y_predict_result = list(filter(lambda x: x, x_results[i])), list(
            filter(lambda x: x, y_predict_results[i]))
          # x_text, y_predict_text = ''.join(id2word[x_result].values), ''.join(id2tag[y_predict_result].values)
          x_text, y_predict_text = x_result, y_predict_result

          print([word_set[idx] for idx in x_text], '\n', y_predict_text)
          print(x_text, '\n', y_predict_text)
          print('-' * 80)

  def run_epoch(self, train_initializer, dev_initializer=None, lr=1e-3,
                epoch_num=100, step_num=1000, dev_step=10,
                save_when_acc=0.94, save_when_loss=0.1, mode='train'):
    self.operate(lr=lr)

    with self.sess.as_default() as sess:
      if mode == 'train':
        sess.run(tf.initialize_all_variables())
      if mode == 'test':
        sess.run(train_initializer)

        X, pred, acc = sess.run([self.X, self.pred, self.accuracy])
        np.savetxt('X.txt', X, fmt='%.1f')
        np.savetxt('pred.txt', pred, fmt='%.1f')
        print('accuracy', acc)
        print('X', X)
        print('pred', pred)
        sys.exit(0)

      for epoch in range(epoch_num):
        # tf.train.global_step(sess, global_step_tensor=global_step)

        # Train
        sess.run(train_initializer)
        bar = tqdm(range(step_num), ncols=100)
        for step in bar:
          try:
            loss, acc, _ = sess.run([self.loss, self.accuracy, self.train_op])
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

        # modify03
        # if acc > save_when_acc:
        if loss < save_when_loss:
          self.saver.save(sess, CHECKPOINT, global_step=epoch)
          print('Saved model done.')


if __name__ == '__main__':
  import sys

  word_set = get_w2i_map('/home/lian/data/nlp/datagrand_info_extra/train.txt')
  print('word set', word_set)
  # sentences, tags = word2ind('/home/lian/data/nlp/datagrand_info_extra/train.txt', word_set)
  sentences, tags, seq_lens = word2ind_with_seqlen('/home/lian/data/nlp/datagrand_info_extra/train.txt', word_set)
  sentences = np.array(sentences)
  tags = np.array(tags)
  seq_lens = np.array(seq_lens)
  print('Batch sequence length', seq_lens)

  # print('train sentences\n', [word_set[item] for line in sentences for item in line][:10])
  # print('train sentences index\n', sentences[:10])
  test_sentences = word2ind_without_seg('/home/lian/data/nlp/datagrand_info_extra/test.txt', word_set)
  # print('test sentences\n', [word_set[item] for line in test_sentences for item in line][:10])
  # print('test sentences index\n', test_sentences[:10])


  train_initializer, dev_initializer, _, iterator = data_iterate_with_seqlen(sentences, tags, seq_lens, 100,
                                                                             mode='test')
  test_initializer, test_iterator = test_data_iterate(sentences, 100)

  X, y, seq_len = iterator.get_next()
  test = test_iterator.get_next()

  model = SequenceLabelling(num_classes=5, vocab_length=4550, word_dim=128, units=128, rnn_layer_num=1, keep_prob=0.88)

  # model.build(X, y, mode='train')

  model.build(X, y, seq_len, mode='train')

  model.run_epoch(train_initializer, dev_initializer, 1e-3, mode='train', save_when_acc=0.99)

  # model.inference(test_initializer, word_set)

  print('All done.')
