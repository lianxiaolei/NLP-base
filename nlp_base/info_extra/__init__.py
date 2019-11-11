# coding:utf8

import tensorflow as tf
from sklearn.model_selection import train_test_split

MAX_SEQ_LEN = 128

t2i_dict = {'c': 1, 'o': 2, 'b': 3, 'a': 4}


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
      if len(line) < 3: continue
      for phrase in line.split(' '):
        if len(phrase) < 3: continue
        words, tag = phrase.split('/')
        words = words.split('_')
        tag = [t2i_dict[tag]] * len(words)
        words = [word_set.index(word) if word in word_set else 0 for word in words]
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


def word2ind_without_seg(fname, word_set):
  """
    Convert word to index.
    Args:
      fname: Source file name.
      word_set:  A fixed word set.

    Returns:
      sentences: A 2D list with word index.
    """
  sentences = []
  with open(fname, 'r', encoding='utf8') as fin:
    content = fin.read().replace('\n\n', '')
    content_list = content.split('\n')
    for line in content_list:
      line_word_seg = []
      if len(line) < 3: continue
      for phrase in line.split(' '):
        if len(phrase) < 3: continue
        words = phrase
        words = words.split('_')
        words = [word_set.index(word) if word in word_set else 0 for word in words]
        line_word_seg.extend(words)

      if len(line_word_seg) < MAX_SEQ_LEN:
        line_word_seg.extend([0] * (MAX_SEQ_LEN - len(line_word_seg)))
      elif len(line_word_seg) > MAX_SEQ_LEN:
        line_word_seg = line_word_seg[: MAX_SEQ_LEN]

      sentences.append(line_word_seg)
  return sentences


def word2ind_with_seqlen(fname, word_set):
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
  seq_lens = []
  tags = []
  length_count = {}
  with open(fname, 'r', encoding='utf8') as fin:
    content = fin.read().replace('\n\n', '')
    content_list = content.split('\n')
    for line in content_list:
      line_word_seg = []
      line_tag_seg = []
      if len(line) < 3: continue
      for phrase in line.split(' '):
        if len(phrase) < 3: continue
        words, tag = phrase.split('/')
        words = words.split('_')
        tag = [t2i_dict[tag]] * len(words)
        words = [word_set.index(word) if word in word_set else 0 for word in words]
        line_word_seg.extend(words)
        line_tag_seg.extend(tag)
      # loop end
      length_count.setdefault(len(line_word_seg), 0)
      length_count[len(line_word_seg)] += 1
      if len(line_word_seg) < MAX_SEQ_LEN:
        seq_lens.append(len(line_word_seg))
        line_word_seg.extend([0] * (MAX_SEQ_LEN - len(line_word_seg)))
        line_tag_seg.extend([0] * (MAX_SEQ_LEN - len(line_tag_seg)))
      elif len(line_word_seg) >= MAX_SEQ_LEN:
        seq_lens.append(MAX_SEQ_LEN)
        line_word_seg = line_word_seg[: MAX_SEQ_LEN]
        line_tag_seg = line_tag_seg[: MAX_SEQ_LEN]

      sentences.append(line_word_seg)
      tags.append(line_tag_seg)
    # loop end
    print('Sentence length', length_count)
  return sentences, tags, seq_lens


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
        if '/' not in phrase:
          words = phrase
        else:
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


def data_iterate_with_seqlen(X, y, seq_len, batch_size, mode='train'):
  """
  Generate iterator for train dataset, dev dataset and test dataset.
  Args:
    X:          Features
    y:          Labels
    batch_size: Batch size.
    mode:

  Returns:
    Return three initializer and a iterator.
  """
  # Split dataset to train and test.
  if mode == 'train':
    shuffle = True
  else:
    shuffle = False
  X_train, X_test, y_train, y_test, sl_train, sl_test = train_test_split(X, y, seq_len, test_size=0.1, shuffle=shuffle)
  train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, sl_train))
  train_dataset = train_dataset.batch(batch_size)

  dev_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test, sl_test))
  dev_dataset = dev_dataset.batch(batch_size)

  test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test, sl_test))
  test_dataset = test_dataset.batch(batch_size)

  # A reinitializable iterator
  iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

  # Initialize train dev and test dataset.
  # This method can switch dataset flow, it's more flexible than single iterator.
  train_initializer = iterator.make_initializer(train_dataset)
  dev_initializer = iterator.make_initializer(dev_dataset)
  test_initializer = iterator.make_initializer(test_dataset)

  return train_initializer, dev_initializer, test_initializer, iterator


def test_data_iterate(X, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices(X)
  dataset = dataset.batch(batch_size)

  # A reinitializable iterator
  iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

  # Initialize train dev and test dataset.
  # This method can switch dataset flow, it's more flexible than single iterator.
  initializer = iterator.make_initializer(dataset)

  return initializer, iterator
