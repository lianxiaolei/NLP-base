# coding:utf8

import tensorflow as tf
from sklearn.model_selection import train_test_split

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

  train_initializer = iterator.make_initializer(train_dataset)
  dev_initializer = iterator.make_initializer(dev_dataset)
  test_initializer = iterator.make_initializer(test_dataset)

  return train_initializer, dev_initializer, test_initializer




if __name__ == '__main__':
  word_set = get_t2i_map('/home/lian/data/nlp/datagrand_info_extra/train.txt')
  sentences, tags = word2ind_with_w2v('/home/lian/data/nlp/datagrand_info_extra/train.txt', word_set)
