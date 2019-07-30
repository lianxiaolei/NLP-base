# coding:utf8

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.contrib import rnn
from tqdm import tqdm

MAX_SEQ_LEN = 128
t2i_dict = {'c': 1, 'o': 2, 'b': 3, 'a': 4}
CHECKPOINT = '/home/lian/PycharmProjects/NLP-base/model/checkpoint/seqtag_ckpt'


def word2ind(fname, word_set):
  """
  Convert word to index.
  Args:
    fname: Source file name.
    word_set:  A fixed word set.

  Returns:
    sentences: A 2D list with word index.
    tags:      A 2D list with word tags.
  """
  sentences = []
  tags = []
  with open(fname, 'r', encoding='utf8') as fin:
    content = fin.read().replace('\n\n', '')
    content_list = content.split('\n')
    for line in content_list:
      line_word_seg = []
      line_tag_seg = []
      for phrase in line.split(' '):
        if len(phrase) < 3: continue
        words, tag = phrase.split('/')
        words = words.split('_')
        tag = [t2i_dict[tag]] * len(words)
        words = [word_set.index(word) for word in words]
        line_word_seg.extend(words)
        line_tag_seg.extend(tag)
      if len(line_word_seg) < MAX_SEQ_LEN:
        line_word_seg.extend([0] * (MAX_SEQ_LEN - len(line_word_seg)))
        line_tag_seg.extend([0] * (MAX_SEQ_LEN - len(line_tag_seg)))
      elif len(line_word_seg) > MAX_SEQ_LEN:
        line_word_seg = line_word_seg[: MAX_SEQ_LEN]
        line_tag_seg = line_tag_seg[: MAX_SEQ_LEN]
      sentences.append(line_word_seg)
      tags.append(line_tag_seg)
  return sentences, tags


def get_w2i_map(fname, preserve_zero=True):
  """
  Generate word set.
  Args:
    fname:         Source file name.
    preserve_zero: Preserve the place for '<UNK>'.

  Returns:
    word_set:      A set contains word index.

  """
  word_list = []

  with open(fname, 'r', encoding='utf8') as fin:
    content = fin.read().replace('\n\n', '')
    content_list = content.split('\n')
    for line in content_list:
      for phrase in line.split(' '):
        if len(phrase) < 3: continue
        words, tag = phrase.split('/')
        words = words.split('_')
        word_list.extend(words)
  word_set = set(word_list)
  word_set = sorted(word_set)

  if preserve_zero:
    word_set.insert(0, '<UNK>')

  return word_set


def data_iterate(X, y, batch_size):
  """
  Generate iterator for train dataset, dev dataset and test dataset.
  Args:
    X:          Features
    y:          Labels
    batch_size: Batch size.

  Returns:
    Return three initializer and a iterator.
  """
  # Split dataset to train and test.
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
  train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  train_dataset = train_dataset.shuffle(17000).batch(batch_size)

  dev_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  dev_dataset = dev_dataset.shuffle(17000).batch(batch_size)

  test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  test_dataset = test_dataset.shuffle(17000).batch(batch_size)

  # A reinitializable iterator
  iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

  # Initialize train dev and test dataset.
  # This method can switch dataset flow, it's more flexible than single iterator.
  train_initializer = iterator.make_initializer(train_dataset)
  dev_initializer = iterator.make_initializer(dev_dataset)
  test_initializer = iterator.make_initializer(test_dataset)

  return train_initializer, dev_initializer, test_initializer, iterator


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
    return tf.reshape(output, [-1, self.units * 2])

  def _tail(self, X):
    logits = tf.keras.layers.Dense(self.num_classes)(X)
    return logits

  def _compile(self, logits, labels):
    # Reshape
    label = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    # Loss
    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=label, logits=tf.cast(logits, tf.float32)))

  def build(self, X, y, mode='train'):
    self._init_embedding()
    lookup = self._lookup(X)
    rnn_outputs = self._rnn_units(lookup)
    logits = self._tail(rnn_outputs)

    self._compile(logits, y)

    # Prediction
    self.pred = tf.cast(tf.argmax(logits, axis=1), tf.int64)
    correct_prediction = tf.equal(self.pred, tf.reshape(y, [-1]))
    print('pred shape {}, label shape {}'.format(self.pred.shape, y.shape))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if mode == 'test':
      self.X = X
      self.y = y

  def operate(self, lr=1e-3):
    # Train
    self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
    # Defind model saver.
    self.saver = tf.train.Saver()

  def restore(self):
    saver = tf.train.Saver()
    with tf.Session() as sess:
      self.model = saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT[:CHECKPOINT.rfind('/')]))
    print('Load model successfully.')

  def inference(self, test_initializer):
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      sess.run(test_initializer)
      for step in range(1):
        x_results, y_predict_results, acc = sess.run([self.X, self.pred, self.accuracy])
        y_predict_results = np.reshape(y_predict_results, x_results.shape)
        for i in range(len(x_results)):
          x_result, y_predict_result = list(filter(lambda x: x, x_results[i])), list(
            filter(lambda x: x, y_predict_results[i]))
          # x_text, y_predict_text = ''.join(id2word[x_result].values), ''.join(id2tag[y_predict_result].values)
          x_text, y_predict_text = x_result, y_predict_result
          print(x_text, y_predict_text)

  def run_epoch(self, train_initializer, dev_initializer=None, lr=1e-3,
                epoch_num=100, step_num=200, dev_step=10, save_when_acc=0.94):
    self.operate(lr=lr)

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
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
          acc = sess.run(self.accuracy)

          bar.set_description_str("Step:{}\tAcc:{}".format(step, acc))

        if acc > save_when_acc:
          self.saver.save(sess, CHECKPOINT,
                          global_step=epoch)
          print('Saved model done.')


if __name__ == '__main__':
  word_set = get_w2i_map('/home/lian/data/nlp/datagrand_info_extra/train.txt')
  sentences, tags = word2ind('/home/lian/data/nlp/datagrand_info_extra/train.txt', word_set)
  sentences = np.array(sentences)
  tags = np.array(tags)

  train_initializer, dev_initializer, test_initializer, iterator = data_iterate(sentences, tags, 100)

  X, y = iterator.get_next()

  model = SequenceLabelling(5, 4550, 20, 128, 1, 1.)

  model.build(X, y, mode='test')

  # model.run_epoch(train_initializer, dev_initializer, 1e-3)
  model.inference(test_initializer)

  print('Model trained done.')
