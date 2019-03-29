# coding:utf8

import numpy as np
import re
import jieba
import os
import itertools
from collections import Counter
from tensorflow.contrib import learn
from gensim.models import Word2Vec
import re
import jieba.analyse
import pandas as pd


class DataHelper(object):
  """

  """

  def __init__(self):
    self.jb = jieba
    self.jb_ana = jieba.analyse

  def load_stoped_word(self, fname):
    self.jb_ana.set_stop_words('/Users/imperatore/PycharmProjects/nlp-base/conf/stopwords.txt')

  def clean_str(self, string):
    """

    Args:
      string:

    Returns:

    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

  def load_data_and_label(self, data_path, mode='train'):
    """
    加载数据集
    Args:
      data_path:
      mode:

    Returns:

    """

    def load_data_label(file_path, need_cut=True):
      """
      处理特征和标签数据
      Args:
        file_path:
        need_cut:

      Returns:

      """
      data = pd.read_csv(file_path, delimiter=',', encoding='utf8', lineterminator='\n', dtype=object)
      print('Length of data', data.shape)
      data.replace(r'\n', ' ', inplace=True, regex=True)
      print('data head', data.head(10))
      data = data.values

      if mode == 'train':
        return data[:, 1].tolist(), data[:, 2].tolist()
      elif mode == 'test':
        return data[:, 1].tolist()
      else:
        raise ValueError('Argument "mode" has wrong value. It only be "train" or "test".')

    # Reading training data
    if mode == 'train':
      x_train, y_train_data = load_data_label(data_path, need_cut=False)

      y_labels = list(set(y_train_data))

      # y_labels = list(set(y_labels))
      label_len = len(y_labels)

      # Building training y
      y_train = np.zeros([len(y_train_data), label_len], dtype=np.int)  # (数据条数, 总标签个数) one-hot
      for index in range(len(y_train_data)):
        y_train[index][y_labels.index(y_train_data[index])] = 1
      return [x_train, y_train, y_labels]
    elif mode == 'test':
      x_test = load_data_label(data_path, need_cut=False)
      return x_test
    else:
      raise ValueError('Argument "mode" has wrong value. It only be "train" or "test".')

  def load_train_dev_data(self, train_file_path):
    """

    Args:
      train_file_path:

    Returns:

    """
    # x_train (数据条数, 1) 分词字符串
    # y_train (数据条数, 总标签个数) one-hot
    x_train_text, y_train, _ = self.load_data_and_label(train_file_path, mode='train')
    print('训练集大小：', y_train.shape[0])

    # 求最长分词字符串的词数
    max_train_document_length = max([len(x.split(' '))
                                     for x in x_train_text])
    print('最长文本', max_train_document_length)

    # Processing input data with VocabularyProcessor
    self.vocab_processor = learn.preprocessing.VocabularyProcessor(
      max_train_document_length)

    # The fit_transform function has the ability to slice
    _ = self.vocab_processor.fit_transform(x_train_text)
    self.vocabulary = self.vocab_processor.vocabulary_

    x_train = np.array(list(self.vocab_processor.transform(x_train_text)))
    # print(x_train_text[0])

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    print("Vocabulary Size: {:d}".format(len(self.vocab_processor.vocabulary_)))
    # print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))
    return x_train, y_train

  def load_test_data(self, test_file_path):
    """

    Args:
      test_file_path:

    Returns:

    """
    x_test_text = self.load_data_and_label(test_file_path, mode='test')

    x_test = np.array(list(self.vocab_processor.transform(x_test_text)))

    return x_test

  def load_embedding_vectors_word2vec(self, filename, encoding='utf8'):
    """

    Args:
      filename:
      encoding:

    Returns:

    """
    word2vec_model = Word2Vec.load(filename)

    # Initialize embedding vectors.
    # For idx=0, the vector wouldn't be initialized by word2vec model vector.
    embedding_vectors = np.random.uniform(-0.25, 0.25,
                                          (len(self.vocabulary), 200))

    # With the word2vec model vectors to initialize embedding vectors.
    for word in word2vec_model.wv.vocab:
      idx = self.vocabulary.get(word)
      if idx != 0:
        embedding_vectors[idx] = word2vec_model[word]
    return embedding_vectors

  def batch_iter(self, data, batch_size, num_epoch, shuffle=True):
    """

    Args:
      data:
      batch_size:
      num_epoch:
      shuffle:

    Returns:

    """
    data = np.array(data)
    data_len = len(data)
    num_batch_per_epoch = int((data_len - 1) / batch_size) + 1

    for epoch in range(num_epoch):
      if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_len))
        shuffle_data = data[shuffle_indices]
      else:
        shuffle_data = data

      for batch_num in range(num_batch_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_len)

        yield shuffle_data[start_index: end_index]


if __name__ == '__main__':
  dh = DataHelper()
  x_train, y_train = dh.load_train_dev_data('../../dataset/train.csv')
  x_test = dh.load_test_data('../../dataset/test.csv')
  print(x_train.shape)
  print(x_test.shape)
