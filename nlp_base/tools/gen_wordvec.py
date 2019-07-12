# coding:utf8

from gensim.models import word2vec
import numpy as np
from tensorflow.contrib import learn
import sys


def gen_wordvec(corp_name, md_name, vproc_md_name=None, sg=1):
  print('Start generating...')
  sentences = []
  with open(corp_name, 'r', encoding='utf8') as fin:
    lines = fin.readlines()
    for line in lines:
      sentences.append(line.split(' '))

  # 求最长分词字符串的词数
  max_document_length = max([len(item) for item in sentences])
  print('Max sequence length', max_document_length)

  for i in range(len(sentences)):
    if len(sentences[i]) > max_document_length:
      sentences[i] = sentences[i][:max_document_length]

  # Processing input data with VocabularyProcessor
  vocab_processor = learn.preprocessing.VocabularyProcessor(
    max_document_length)
  print('Word to index done.')

  # # The fit_transform function has the ability to slice
  # corp_idxs = vocab_processor.fit_transform([' '.join(line) for line in sentences])
  # vocabulary = vocab_processor.vocabulary_
  # print(list(corp_idxs)[:10])
  #
  # if vproc_md_name:
  #   vocab_processor.save(vproc_md_name)
  #   print('Save vocabulary processor done.')

  # sentences = word2vec.Text8Corpus(corp_name)
  model = word2vec.Word2Vec(sentences, sg=sg)
  model.save(md_name)

  # get word index from model
  # id = model.wv.vocabulary[word].index


if __name__ == '__main__':
  gen_wordvec('/home/lian/data/nlp/datagrand_info_extra/total_corpus_pretr.txt',
              '../../datagrand_corpus_pretrain.bin', '../../datagrand_vocab_processor.bin')

# model = word2vec.Word2Vec.load('../../model/datagrand_corpus_pretrain.bin')
# idx = sorted(model.wv.index2word)[:100]
# print(len(model.wv.index2word))

# sentences = [["平行线", "直线", "这条"], ["曲线", "弧长", "曲率"], ['<UNK>']]
# vocab_processor = learn.preprocessing.VocabularyProcessor(2)
# print([' '.join(line) for line in sentences])
# print(list(vocab_processor.fit_transform([' '.join(line) for line in sentences])))
# print(vocab_processor.vocabulary_._mapping)
