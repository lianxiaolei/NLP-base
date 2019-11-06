# coding:utf8

import collections
import math
import os
import random
import zipfile
import matplotlib.pyplot as plt

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

# 第一步: 在下面这个地址下载语料库
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
  """
  这个函数的功能是：
      如果filename不存在，就在上面的地址下载它。
      如果filename存在，就跳过下载。
      最终会检查文字的字节数是否和expected_bytes相同。
  """
  if not os.path.exists(filename):
    print('start downloading...')
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename


# 将语料库解压，并转换成一个word的list
def read_data(filename):
  """
  这个函数的功能是：
      将下载好的zip文件解压并读取为word的list
  """
  # with zipfile.ZipFile(filename) as f:
  #   data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  # return data

  with open(filename, 'r') as fin:
    return fin.readlines()[0].split()


def build_dataset(words, n_words):
  """
  函数功能：将原始的单词表示变成index
  """
  count = [['UNK', -1]]
  # Calculate every word's frequence and preserve the top n_words-1.
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:  # word, freq
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # UNK的index为0
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary


def generate_batch(batch_size, cbow_window):
  global data_index
  assert cbow_window % 2 == 1
  span = 2 * cbow_window + 1
  # 去除中心word: span - 1
  batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

  buffer = collections.deque(maxlen=span)

  for _ in range(span):
    buffer.append(data[data_index])
    # 循环选取 data中数据，到尾部则从头开始
    data_index = (data_index + 1) % len(data)

  for i in range(batch_size):
    # target at the center of span
    target = cbow_window
    # 仅仅需要知道context(word)而不需要word
    target_to_avoid = [cbow_window]

    col_idx = 0
    for j in range(span):
      # 略过中心元素 word
      if j == span // 2:
        continue
      batch[i, col_idx] = buffer[j]
      col_idx += 1
    labels[i, 0] = buffer[target]
    # 更新 buffer
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)

  return batch, labels


def plot_with_labels(low_dim_embs, labels, filename='tsne1.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)


vocabulary = read_data('text8.data')
print('Data size', len(vocabulary))  # 总长度为1700万左右
# 输出前100个词。
print(vocabulary[0:100])

# 第二步: 制作一个词表，将不常见的词变成一个UNK标识符
# 词表的大小为5万（即我们只考虑最常出现的5万个词）
vocabulary_size = 50000

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # 删除已节省内存

data_index = 0

if __name__ == '__main__':
  # 下载语料库text8.zip并验证下载
  # filename = maybe_download('text8.zip', 31344016)

  vocabulary = read_data('text8.data')
  print('Data size', len(vocabulary))  # 总长度为1700万左右
  # 输出前100个词。
  print(vocabulary[0:100])

  # 第二步: 制作一个词表，将不常见的词变成一个UNK标识符
  # 词表的大小为5万（即我们只考虑最常出现的5万个词）
  vocabulary_size = 50000

  data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                              vocabulary_size)
  del vocabulary  # 删除已节省内存
  # 输出最常出现的5个单词
  print('Most common words (+UNK)', count[:5])
  # 输出转换后的数据库data，和原来的单词（前10个）
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
  # 我们下面就使用data来制作训练集
  print("+++++++++++++++++")
  data_index = 0
