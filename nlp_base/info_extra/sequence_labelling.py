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

    # Defind model saver.
    self.saver = tf.train.Saver()

  def restore(self):
    saver = tf.train.Saver()
    with self.sess.as_default() as sess:
      saver.restore(sess, tf.train.latest_checkpoint(self.FLAGS.checkpoint[:self.FLAGS.checkpoint.rfind('/')]))
    print('Load model successfully.')

  def train(self, iterator, train_initializer, dev_initializer=None,
            epoch_num=100, step_num=1000, save_when_acc=None, save_when_loss=None, dev_step=10):
    X_ten, y_ten, seq_len_ten = iterator.get_next()
    with self.sess.as_default() as sess:
      sess.run(tf.initialize_all_variables())
      for epoch in range(epoch_num):
        sess.run(train_initializer)
        bar = tqdm(range(step_num), ncols=100)
        for step in bar:
          try:
            X, y, seq_len = sess.run([X_ten, y_ten, seq_len_ten])
            # print(X[0, 0], y[0, : 50])
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
            X, y, seq_len = sess.run([X_ten, y_ten, seq_len_ten])
            acc = sess.run(self.accuracy,
                           feed_dict={self.X: X,
                                      self.y: y,
                                      self.seq_len: seq_len}
                           )
          except tf.errors.OutOfRangeError:
            sess.run(dev_initializer)
          bar.set_description_str("Step:{}\tAcc:{}".format(step, acc))

        if save_when_loss:
          if loss <= save_when_loss:
            self.saver.save(sess, self.FLAGS.check_point, global_step=epoch)
            print('Saved model done.')
        if save_when_acc:
          if acc >= save_when_acc:
            self.saver.save(sess, self.FLAGS.check_point, global_step=epoch)
            print('Saved model done.')

  def inference(self, iterator, test_initializer, word_set):
    with self.sess.as_default() as sess:
      sess.run(tf.initialize_all_variables())
      sess.run(test_initializer)

      X_ten, y_ten, seq_len_ten = iterator.get_next()

      for step in range(1):
        X, y, seq_len = sess.run([X_ten, seq_len_ten])
        y_predict_results, acc = sess.run([self.pred, self.accuracy])
        y_predict_results = np.reshape(y_predict_results, X.shape)
        for i in range(len(X)):
          x_result, y_predict_result = list(filter(lambda x: x, X[i])), list(
            filter(lambda x: x, y_predict_results[i]))
          # x_text, y_predict_text = ''.join(id2word[x_result].values), ''.join(id2tag[y_predict_result].values)
          x_text, y_predict_text = x_result, y_predict_result
          print([word_set[idx] for idx in x_text], '\n', y_predict_text)
          print(x_text, '\n', y_predict_text)
          print('-' * 80)


if __name__ == '__main__':
  tf.app.flags.DEFINE_integer("num_classes", 5, "Number of class")
  tf.app.flags.DEFINE_integer("vocab_length", 4550, "Number of vocabular size")
  tf.app.flags.DEFINE_integer("word_dim", 128, "Dimensionality of character embedding (default: 128)")
  tf.app.flags.DEFINE_integer('rnn_units', 128, "Number of rnn output units")
  tf.app.flags.DEFINE_integer('rnn_layer_num', 1, "Number of rnn layer")
  tf.app.flags.DEFINE_integer('max_seq_len', 128, "Number of max sequence length")
  tf.app.flags.DEFINE_float("keep_prob", 0.8, "Dropout keep probability (default: 0.8)")
  tf.app.flags.DEFINE_float("lr", 0.01, "Learning rate")
  tf.app.flags.DEFINE_string("checkpoint",
                             '/home/lian/PycharmProjects/NLP-base/model/checkpoint/seqtag_ckpt',
                             "Model checkpoint")

  # word_set = get_w2i_map('/home/lian/data/nlp/datagrand_info_extra/train.txt')
  # print('word set', word_set)
  # # sentences, tags = word2ind('/home/lian/data/nlp/datagrand_info_extra/train.txt', word_set)
  # sentences, tags, seq_lens = word2ind_with_seqlen('/home/lian/data/nlp/datagrand_info_extra/train.txt', word_set)
  # sentences = np.array(sentences)
  # tags = np.array(tags)
  # seq_lens = np.array(seq_lens)
  # print('Batch sequence length', seq_lens)
  #
  # test_sentences = word2ind_without_seg('/home/lian/data/nlp/datagrand_info_extra/test.txt', word_set)
  #
  # np.savetxt('/home/lian/PycharmProjects/NLP-base/swap/word_set.txt', np.array(word_set), fmt='%s')
  # np.savetxt('/home/lian/PycharmProjects/NLP-base/swap/sentences.txt', np.array(sentences), fmt='%d')
  # np.savetxt('/home/lian/PycharmProjects/NLP-base/swap/tags.txt', np.array(tags), fmt='%d')
  # np.savetxt('/home/lian/PycharmProjects/NLP-base/swap/seq_lens.txt', np.array(seq_lens), fmt='%d')
  # np.savetxt('/home/lian/PycharmProjects/NLP-base/swap/test_sentences.txt', np.array(test_sentences), fmt='%d')
  # print('Saved done.')
  # import sys; sys.exit(0)

  word_set = np.loadtxt('/home/lian/PycharmProjects/NLP-base/swap/word_set.txt', dtype=np.object)
  sentences = np.loadtxt('/home/lian/PycharmProjects/NLP-base/swap/sentences.txt', dtype=np.int)
  tags = np.loadtxt('/home/lian/PycharmProjects/NLP-base/swap/tags.txt', dtype=np.int)
  seq_lens = np.loadtxt('/home/lian/PycharmProjects/NLP-base/swap/seq_lens.txt', dtype=np.int)

  train_initializer, dev_initializer, _, iterator = data_iterate_with_seqlen(sentences, tags, seq_lens,
                                                                             batch_size=100, mode='train')
  test_initializer, test_iterator = test_data_iterate(sentences, 100)

  model = SequenceLabelling()

  model.build_net(compile=True)

  model.train(iterator, train_initializer, dev_initializer, epoch_num=100, step_num=1000,
              save_when_loss=2e-1, dev_step=10)

  # model.inference(test_initializer, word_set)

  print('All done.')
