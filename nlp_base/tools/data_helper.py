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
import tensorflow as tf

# jieba.analyse.set_stop_words('/Users/imperatore/PycharmProjects/nlp-base/conf/stopwords.txt')
MAX_LEN = 64  # 限定句子的最大单词数量。
SOS_ID = 1  # 目标语言词汇表中<sos>的ID。


def make_dataset(fname):
  """

  Args:
    fname:

  Returns:

  """
  dataset = tf.data.TextLineDataset(fname)

  # Add '.values' to convert sparseTensor into tensor
  # After using the '.values', the data is also sparseTensor,
  #   we need to padding in the following steps
  dataset = dataset.map(lambda s: tf.string_split([s]).values)

  # Convert string to number
  dataset = dataset.map(lambda s: tf.string_to_number(s, tf.int32))

  # Count length of each sentence
  dataset = dataset.map(lambda s: (s, tf.size(s)))

  return dataset


def filter_by_length(src_tuple, trg_tuple):
  """

  Args:
    src_tuple:
    trg_tuple:

  Returns:

  """
  ((src_input, src_len), trg_label) = (src_tuple, trg_tuple)
  src_len_ok = tf.logical_and(
    tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
  return tf.logical_and(src_len_ok, True)


def gen_target_input(src_tuple, trg_tuple):
  """

  Args:
    src_tuple:
    trg_tuple:

  Returns:

  """
  # ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
  # # src_input += 1
  # return (src_input, src_len), trg_label
  ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
  # src_input += 1
  return (src_input, src_len), trg_label


def gen_src_tar_dataset(src_path, tar_path, batch_size):
  """

  Args:
    src_path:
    trg_path:
    batch_size:

  Returns:

  """
  src_data = make_dataset(src_path)
  trg_data = make_dataset(tar_path)

  dataset = tf.data.Dataset.zip((src_data, trg_data))

  dataset = dataset.filter(filter_by_length)

  # Decoder needs two types sentence：
  # 1.decoder's input like "<sos> X Y Z"
  # 2.decoder's label like "X Y Z <eos>"
  # The sentences we read from method is as type "X Y Z <eos>",
  # we need to generate decoder's input sentence as type "<sos> X Y Z".
  # And add the generated labels to dataset.
  dataset = dataset.map(gen_target_input)

  dataset = dataset.shuffle(10000)

  # Define the output size of the padding data。
  padded_shapes = (
    (tf.TensorShape([None]),  # src sentence is a dynamic length vector.
     tf.TensorShape([])),  # src sentence's length is a int value.
    tf.TensorShape([None]),  # tar sentence(output) is a dynamic length vector.
  )

  # Call padded_batch to make batch.
  batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
  return batched_dataset


def load_embedding_vectors_word2vec(filename, emb_dim):
  """

  :param vocabulary:
  :param filename:
  :param binary:
  :return:
  """
  word2vec_model = Word2Vec.load(filename)
  embedding_vectors = np.random.uniform(-0.25, 0.25,
                                        (len(word2vec_model.wv.vocab) + 1, emb_dim))

  for word in word2vec_model.wv.vocab:
    idx = word2vec_model.wv.vocab[word].index
    embedding_vectors[idx + 1] = word2vec_model[word]
  return embedding_vectors


def split_sentence(fname_in, fname_out, has_tag=False):
  fout = open(fname_out, 'a', encoding='utf-8')

  with open(fname_in, 'r', encoding='utf8') as fin:
    lines = fin.readlines()
    i = 0
    for line in lines:
      i += 1
      line = line.replace('\n', '')
      if len(line) == 0: continue
      phrases = line.split(' ')
      sentence = []
      for phrase in phrases:
        if phrase == '': continue
        if has_tag:
          words = phrase.split('/')[0]
        else:
          words = phrase
        sentence.append(' '.join(words.split('_')))
      # print(' '.join(sentence))
      fout.write(' '.join(sentence) + '\n')  # 词汇用空格分开
      if i % 10000 == 0:
        print('Current %s lines.' % i)
  fout.close()


def merge_corpus(fnames, target_fname):
  print('start merge')
  fout = open(target_fname, 'a', encoding='utf8')
  for i in range(0, len(fnames)):
    with open(fnames[i], 'r', encoding='utf8') as fin:
      lines = fin.readlines()[:-1]
      fout.writelines(lines)
  fout.close()


def gen_target_data(fname, target_fname):
  fout = open(target_fname, 'a', encoding='utf8')
  with open(fname, 'r', encoding='utf8') as fin:
    lines = fin.readlines()
    for line in lines:
      line = line.replace('\n', '')
      line_tags = []
      phrases = line.split(' ')
      for phrase in phrases:
        if len(phrase) == 0: continue
        words, tag = phrase.split('/')
        words = words.split('_')
        for word in words:
          line_tags.append(tag)
      # print(' '.join(line_tags))
      fout.write(' '.join(line_tags) + '\n')
  fout.close()
  print('Generate tags done.')


def get_w2v_vocab(md_fname):
  word2vec_model = Word2Vec.load(md_fname)
  print(sorted(word2vec_model.wv.vocab.items()))
  print(len(word2vec_model.wv.vocab.keys()))

  # for word in word2vec_model.wv.vocab.keys():
  #   print(word, word2vec_model.wv.vocab[word].index)


def gen_train_index(fname, target_fname, md_fname):
  word2vec_model = Word2Vec.load(md_fname)
  fout = open(target_fname, 'a', encoding='utf8')
  with open(fname, 'r', encoding='utf8') as fin:
    lines = fin.readlines()
    lines = lines[: -1]
    for line in lines:
      line = line.replace('\n', '')
      words = line.split(' ')
      word_ids = []
      for word in words:
        try:
          word_id = word2vec_model.wv.vocab[word]
          word_ids.append(str(word_id.index))
        except Exception as e:
          print('Model cannot recognized', word)
          word_id = 0
          word_ids.append(str(word_id))
      fout.write(' '.join(word_ids) + '\n')
  print('Convert word to index done.')


def tag2num(fname, out_fname):
  w2i_dict = {'c': 1, 'o': 2, 'b': 3, 'a': 4}
  fout = open(out_fname, 'a', encoding='utf8')
  with open(fname, 'r', encoding='utf8') as fin:
    lines = fin.readlines()
    lines = lines[: -1]
    for line in lines:
      line_tag2num = []
      line = line.replace('\n', '')
      words = line.split(' ')
      for word in words:
        line_tag2num.append(str(w2i_dict[word]))
      fout.write(' '.join(line_tag2num) + '\n')
  fout.close()


def count_label_num(fname):
  tag_dict = {}
  with open(fname, 'r', encoding='utf8') as fin:
    lines = fin.readlines()
    index = 1
    for line in lines:
      line = line.replace('\n', '')
      words = line.split(' ')
      for word in words:
        if not tag_dict.get(word):
          tag_dict[word] = index - 1
          index += 1
    print(tag_dict)


def compare_index_tag(index_fname, tag_fname):
  fid = open(index_fname, 'r', encoding='utf8')
  ftg = open(tag_fname, 'r', encoding='utf8')
  ids = fid.readlines()
  tgs = ftg.readlines()
  print(len(ids), len(tgs))
  assert len(ids) == len(tgs), "行长度不一样"
  for i in range(len(ids)):
    if len(ids[i].split(' ')) != len(tgs[i].split(' ')):
      print('在%s行长度不一致%s:%s' % (i, len(ids[i].split(' ')), len(tgs[i].split(' '))))
  fid.close()
  ftg.close()


def check_index_corr(fname):
  with open(fname, 'r', encoding='utf8') as fin:
    lines = fin.readlines()
    total = []
    for line in lines:
      total.extend(line[:-1].split(' '))
    total = sorted(set(total))
    print(total)
    print('Length of vocabulary', len(total))
    for i in range(len(total)):
      if str(i) not in total:
        print('%s is not in vocabulary index.' % i)


def merge_word_tag(word_fname, tag_fname, target_fname):
  fword = open(word_fname, 'r', encoding='utf8')
  ftg = open(tag_fname, 'r', encoding='utf8')
  words = fword.readlines()
  tags = ftg.readlines()


def get_low_freq_word(trg_word_fname, freq=5):
  freq_dic = {}
  removable = []
  with open(trg_word_fname, 'r', encoding='utf8') as fin:
    lines = fin.readlines()
    for line in lines:
      line = line.replace('\n', '')
      words = line.split(' ')
      for word in words:
        if len(word) < 1: continue
        freq_dic.setdefault(word, 0)
        if (freq_dic[word] == 0) and (word not in removable):
          removable.append(word)
        freq_dic[word] += 1
        if (freq_dic[word] >= 5) and (word in removable):
          removable.remove(word)
  freq_dic = sorted(freq_dic.items(), key=lambda d: d[1])
  print('frequence dictionary', freq_dic)
  print('remoable list', removable)

  with open(trg_word_fname, 'r', encoding='utf8') as fin:
    lines = fin.readlines()
    loc = []
    row = 0
    for line in lines:
      line = line.replace('\n', '')
      col = 0
      for word in line.split(' '):
        if word in removable:
          loc.append([row, col])
        col += 1
      row += 1

  return loc


def remove_low_freq_word(locs, trg_word_fname, trg_word_cled_fname,
                         trg_index_fname, trg_index_cled_fname):
  f_word = open(trg_word_fname, 'r', encoding='utf8')
  f_word_cled = open(trg_word_cled_fname, 'w', encoding='utf8')

  lines = f_word.readlines()
  words = [line.replace('\n', '').split(' ') for line in lines]

  for loc in locs:
    current_words = words[loc[0]]
    words[loc[0]].remove(current_words[loc[1]])


if __name__ == '__main__':
  # split_sentence('/home/lian/data/nlp/datagrand_info_extra/corpus.txt',
  #                '/home/lian/data/nlp/datagrand_info_extra/corpus_sliced.txt')
  # split_sentence('/home/lian/data/nlp/datagrand_info_extra/train.txt',
  #                '/home/lian/data/nlp/datagrand_info_extra/train_sliced.txt', has_tag=True)
  # split_sentence('/home/lian/data/nlp/datagrand_info_extra/test.txt',
  #                '/home/lian/data/nlp/datagrand_info_extra/test_sliced.txt', has_tag=False)

  # split_sentence('/Users/lianxiaohua/Data/datagrand/corpus.txt',
  #                '/Users/lianxiaohua/Data/datagrand/corpus_pretr.txt')
  # split_sentence('/Users/lianxiaohua/Data/datagrand/train.txt',
  #                '/Users/lianxiaohua/Data/datagrand/train_pretr.txt', has_tag=True)
  # split_sentence('/Users/lianxiaohua/Data/datagrand/test.txt',
  #                '/Users/lianxiaohua/Data/datagrand/test_pretr.txt', has_tag=False)
  # print('Split sentence done.')

  # merge_corpus(['/home/lian/data/nlp/datagrand_info_extra/corpus_sliced.txt',
  #               '/home/lian/data/nlp/datagrand_info_extra/train_sliced.txt',
  #               '/home/lian/data/nlp/datagrand_info_extra/test_sliced.txt'],
  #              '/home/lian/data/nlp/datagrand_info_extra/corpus_pretr.txt')

  # gen_target_data('/home/lian/data/nlp/datagrand_info_extra/train.txt',
  #                 '/home/lian/data/nlp/datagrand_info_extra/target.txt')

  print(get_w2v_vocab('../../model/datagrand_corpus_pretrain.bin'))

  gen_train_index('/home/lian/data/nlp/datagrand_info_extra/train_sliced.txt',
                  '/home/lian/data/nlp/datagrand_info_extra/train_index.txt',
                  '../../model/datagrand_corpus_pretrain.bin')

  tag2num('/home/lian/data/nlp/datagrand_info_extra/target.txt',
          '/home/lian/data/nlp/datagrand_info_extra/target_index.txt')

  count_label_num('/home/lian/data/nlp/datagrand_info_extra/target.txt')

  compare_index_tag('/home/lian/data/nlp/datagrand_info_extra/train_index.txt',
                    '/home/lian/data/nlp/datagrand_info_extra/target_index.txt')

  # remove_low_freq_word('/home/lian/data/nlp/datagrand_info_extra/train_sliced.txt',
  #                      '/home/lian/data/nlp/datagrand_info_extra/train_index.txt')
  print('done')
