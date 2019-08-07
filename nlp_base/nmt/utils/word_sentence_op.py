# coding: utf8

import jieba

print(' '.join(jieba.cut('初始化操作...')))
from seq2seq.utils import reduce
from tensorflow.contrib import learn
import numpy as np
import os


def load_corpus(fname, min_max_len=(0, 1e4), need_cut=False):
  """

  Args:
    fnames:
    min_max_len:
    need_cut:

  Returns:

  """

  data = list()
  with open(fname, 'r', encoding='utf8') as f:
    for line in f.read().split('\n'):
      line = line.replace('\n', '')
      if len(line) < min_max_len[0] or len(line) > min_max_len[1]: continue
      if need_cut:
        words = list(jieba.cut(line))
      else:
        words = line.split(' ')

      data.append(words)

  uni_data = list(set(reduce(data)))

  len_data = len(uni_data)

  return data, uni_data, len_data


def generate_vocab(words, **kwargs):
  """
  Generate dict with type{ind, w} and {w, ind}, and save to file.
  Args:
    words:
    **kwargs:

  Returns:

  """
  dic = {}
  rev_dic = {}
  if kwargs.get('vocab_fname'):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'dataset')
    with open(os.path.join(data_path, kwargs.get('vocab_fname')), 'w', encoding='utf8') as f:
      for i, w in enumerate(words):
        dic['%s' % i] = w
        rev_dic[w] = '%s' % i
        f.writelines('{}\n'.format(w))
  else:
    for i, w in enumerate(words):
      print(i)
      dic['%s' % i] = w
      rev_dic[w] = '%s' % i

  return dic, rev_dic


def word2index(corpus, dictionary):
  """

  Args:
    corpus:
    dictionary:
    **kwargs:

  Returns:

  """
  result = []
  for sentence in corpus:
    sen_ind = [dictionary[w] for w in sentence]
    result.append(sen_ind)
  return result


def load_padding_data(corpus):
  """

  Args:
    corpus:

  Returns:

  """
  max_document_length = max([len(sentence) for sentence in corpus])
  vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
  vocab_processor.fit(corpus)

  corpus = vocab_processor.transform(corpus)

  return corpus, vocab_processor


def shuffle(data):
  """

  Args:
    data:

  Returns:

  """
  shuffle_index = np.random.permutation(np.arange(len(data[0])))

  result = []
  for d in data:
    d = d[shuffle_index]
    result.append(d)

  return result


if __name__ == '__main__':
  base_path = '/Users/lianxiaohua/Learning/deepshare/attention_model/TED_data'
  data, uni_data, len_data = load_corpus(os.path.join(base_path, 'train.raw.zh'), need_cut=True)
  print('Load done.')
  dic, rev_dic = generate_vocab(uni_data)
  print('Generated done.')
  index_corpus = word2index(data, rev_dic)

  with open('train_self.zh', 'w', encoding='utf8') as f:
    for index_sen in index_corpus:
      f.writelines(' '.join(index_sen) + '\n')
  print('Written done.')
