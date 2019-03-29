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
import sklearn.model_selection

jieba.analyse.set_stop_words('/Users/imperatore/PycharmProjects/nlp-base/conf/stopwords.txt')


def clean_str(string):
  """

  :param string:
  :return:
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


def load_data_and_labels(train_file_path, test_file_path):
  """

  :param train_file_path:
  :param test_file_path:
  :return:
  """

  # Load data from files
  # x_train = list()  # 训练数据
  # x_test = list()  # 测试数据
  # y_train_data = list()  # y训练分类数据
  # y_test_data = list()  # y测试分类数据
  # y_labels = list()  # 分类集

  # def load_data_label(file_path, need_cut=False, fmt='id,feature,label'):
  #   data = pd.read_csv(file_path, names=fmt.split(','), skiprows=1, dtype=object,
  #                      delimiter=',', lineterminator='\n')
  #
  #   feature = data['feature'].values
  #   label = data['label'].values
  #   if need_cut:
  #     feature = [' '.join(jieba.analyse.extract_tags(' '.join(jieba.cut(text)), topK=20)) for text in feature]
  #   return feature.tolist(), label.tolist()

  def load_data_label(file_path, need_cut=True):
    x_data = list()
    y_data = list()

    with open(file_path, 'r', encoding='utf8') as train_file:
      for line in train_file.read().split('\n'):
        if len(line) < 10: continue
        line = re.sub("[' ']+", ' ', line)
        data = line.split('\t')
        text = data[1]
        if need_cut:
          text = ' '.join(jieba.cut(text))
          text = ' '.join(jieba.analyse.extract_tags(text, topK=20))
        label = data[0].replace('__label__', '')
        x_data.append(text)
        y_data.append(label)
    return x_data, y_data

  # Reading training data
  x_train, y_train_data = load_data_label(train_file_path)

  y_labels = list(set(y_train_data))

  # Reading test data
  x_test, y_test_data = load_data_label(test_file_path)
  print('Test data:', y_test_data)

  y_labels += list(set(y_test_data))
  y_labels = list(set(y_labels))
  label_len = len(y_labels)

  # Building training y
  y_train = np.zeros([len(y_train_data), label_len], dtype=np.int)  # (数据条数, 总标签个数) one-hot
  for index in range(len(y_train_data)):
    try:
      y_train[index][y_labels.index(y_train_data[index])] = 1
    except Exception as e:
      print(index)
      raise ValueError('baocao')

  # 构建测试y
  y_test = np.zeros((len(y_test_data), label_len), dtype=np.int)
  for index in range(len(y_test_data)):
    y_test[index][y_labels.index(y_test_data[index])] = 1

  return [x_train, y_train, x_test, y_test, y_labels]


def load_train_dev_data(train_file_path, test_file_path):
  """

  :param train_file_path:
  :param test_file_path:
  :return:
  """
  # x_train (数据条数, 1) 分词字符串
  # y_train (数据条数, 总标签个数) one-hot
  x_train_text, y_train, x_test_text, y_test, _ = load_data_and_labels(train_file_path, test_file_path)

  # 求最长分词字符串的词数
  max_train_document_length = max([len(x.split(' '))
                                   for x in x_train_text])

  max_test_document_length = max([len(x.split(' '))
                                  for x in x_test_text])

  max_document_length = max(max_train_document_length,
                            max_test_document_length)
  print('最长文本', max_document_length)

  # Processing input data with VocabularyProcessor
  vocab_processor = learn.preprocessing.VocabularyProcessor(
    max_document_length)

  # The fit_transform function has the ability to slice
  print(type(x_train_text), type(x_test_text))
  _ = vocab_processor.fit_transform(x_train_text + x_test_text)
  x_train = np.array(list(vocab_processor.transform(x_train_text)))
  # print(x_train_text[0])
  x_test = np.array(list(vocab_processor.transform(x_test_text)))

  shuffle_indices = np.random.permutation(np.arange(len(y_train)))
  x_train = x_train[shuffle_indices]
  y_train = y_train[shuffle_indices]

  print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
  # print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))
  return x_train, y_train, x_test, y_test, vocab_processor


def load_embedding_vectors_word2vec(vocabulary, filename, encoding='utf8'):
  """

  :param vocabulary:
  :param filename:
  :param binary:
  :return:
  """
  word2vec_model = Word2Vec.load(filename)

  embedding_vectors = np.random.uniform(-0.25, 0.25,
                                        (len(vocabulary), 200))

  for word in word2vec_model.wv.vocab:
    idx = vocabulary.get(word)
    if idx != 0:
      embedding_vectors[idx] = word2vec_model[word]
  return embedding_vectors


def batch_iter(data, batch_size, num_epoch, shuffle=True):
  """

  :param data:
  :param batch_size:
  :param num_epoch:
  :param shuffle:
  :return:
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
  x_train, y_train, x_test, y_test, vocab_processor = \
    load_train_dev_data('../../dataset/cnews.train.txt', '../../dataset/cnews.test.txt')
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)
  print('😒😒😒😒')
