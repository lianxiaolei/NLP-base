# coding:utf8

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.contrib import rnn
from tqdm import tqdm

MAX_SEQ_LEN = 64
t2i_dict = {'c': 1, 'o': 2, 'b': 3, 'a': 4}


def word2ind_with_w2v(fname, word_set):
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


def get_t2i_map(fname, preserve_zero=True):
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

  Args:
    X:
    y:
    batch_size:

  Returns:

  """
  # Split dataset to train and test.
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
  train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  train_dataset = train_dataset.batch(batch_size)

  dev_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  dev_dataset = dev_dataset.batch(batch_size)

  test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  test_dataset = test_dataset.batch(batch_size)

  # A reinitializable iterator
  iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

  # Initialize train dev and test dataset.
  # This method can switch dataset flow, it's more flexible than single iterator.
  train_initializer = iterator.make_initializer(train_dataset)
  dev_initializer = iterator.make_initializer(dev_dataset)
  test_initializer = iterator.make_initializer(test_dataset)

  return train_initializer, dev_initializer, test_initializer, iterator


class SequenceLabelling(object):
  def __init__(self, num_classes, vocab_length, word_dim, units, rnn_layer_num, keep_prob):
    self.num_classes = num_classes
    self.vocab_length = vocab_length
    self.word_dim = word_dim
    self.units = units
    self.rnn_layer_num = rnn_layer_num
    self.keep_prob = keep_prob

  def lstm_cell(self, reuse=False):
    if reuse:
      tf.get_variable_scope.reuse_variables()
    cell = rnn.LSTMCell(self.units)
    return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

  def _init_embedding(self):
    self.embedding = tf.Variable(tf.random_normal([self.vocab_length, self.word_dim]), dtype=tf.float32)

  def lookup(self, X):
    self.inputs = tf.nn.embedding_lookup(self.embedding, X + 1)

  def rnn_units(self):
    # Defind rnn base units.
    cell_fw = [self.lstm_cell() for _ in range(self.rnn_layer_num)]
    cell_bw = [self.lstm_cell() for _ in range(self.rnn_layer_num)]

    # Unstack the inputs to list with step items.
    # TODO Why we need to unstack the timestep?
    inputs = tf.unstack(self.inputs, 64, axis=1)

    output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw,
                                                          inputs=inputs, dtype=tf.float32)

    # Stack the outputs.
    output = tf.stack(output, axis=1)

    # Reshape output to [batch * times, units * 2]
    return tf.reshape(output, [-1, self.units * 2])

  def compile(self, X):
    logits = tf.keras.layers.Dense(self.num_classes)(X)
    y_predict = tf.cast(tf.argmax(logits, axis=1), tf.int32)

    y_label_reshape = tf.cast(tf.reshape(self.y, [-1]), tf.int32)
    # Prediction
    correct_prediction = tf.equal(y_predict, y_label_reshape)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Loss
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y_label_reshape, logits=tf.cast(logits, tf.float32)))

    # Train
    train = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    saver = tf.train.Saver()



if __name__ == '__main__':
  word_set = get_t2i_map('/home/lian/data/nlp/datagrand_info_extra/train.txt')
  sentences, tags = word2ind_with_w2v('/home/lian/data/nlp/datagrand_info_extra/train.txt', word_set)
  sentences = np.array(sentences)
  tags = np.array(tags)

  train_initializer, dev_initializer, test_initializer, iterator = data_iterate(sentences, tags, 10)

  x, y = iterator.get_next()

  embedding = tf.Variable(tf.random_normal([4550, 20]), dtype=tf.float32)
  inputs = tf.nn.embedding_lookup(embedding, x + 1)

  cell_fw = [lstm_cell(128, 1) for _ in range(1)]
  cell_bw = [lstm_cell(128, 1) for _ in range(1)]
  inputs = tf.unstack(inputs, 64, axis=1)
  output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw,
                                                        inputs=inputs, dtype=tf.float32)

  output = tf.stack(output, axis=1)
  print(output.shape)
  output = tf.reshape(output, [-1, 128 * 2])

  logits = tf.keras.layers.Dense(5)(output)
  y_predict = tf.cast(tf.argmax(logits, axis=1), tf.int32)
  print('Output Y', y_predict)

  # Reshape y_label
  y_label_reshape = tf.cast(tf.reshape(y, [-1]), tf.int32)
  # Prediction
  correct_prediction = tf.equal(y_predict, y_label_reshape)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # Loss
  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y_label_reshape, logits=tf.cast(logits, tf.float32)))

  # Train
  train = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

  saver = tf.train.Saver()

  acc_thres = 0.94
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(100):
      # tf.train.global_step(sess, global_step_tensor=global_step)

      # Train
      sess.run(train_initializer)
      bar = tqdm(range(200), ncols=100)
      for step in bar:
        loss, acc, _ = sess.run([cross_entropy, accuracy, train])

        bar.set_description_str("Step:{}\t  Loss:{}\t  Acc:{}".format(step, str(loss)[:5], str(acc)[:5]))
        # if step % 10 == 0:
        #   print('Step', step, 'Train Loss', loss, 'Accuracy', acc)

      # Dev
      sess.run(dev_initializer)
      bar = tqdm(range(10), ncols=10)
      for step in bar:
        acc = sess.run(accuracy)

        bar.set_description_str("Step:{}\tAcc:{}".format(step, acc))
        # if step % 20 == 0:
        #   print('Dev Accuracy', acc)

      if acc > acc_thres:
        saver.save(sess, '/home/lian/PycharmProjects/NLP-base/model/checkpoint/seqtag_ckpt', global_step=step)
        print('Saved model done.')
