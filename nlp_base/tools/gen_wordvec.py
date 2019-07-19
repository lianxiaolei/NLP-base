# coding:utf8

from gensim.models import word2vec
import numpy as np
from tensorflow.contrib import learn
import sys
import time


def gen_wordvec(corp_name, md_name, sg=1):
  print('Start generating...')
  sentences = []
  with open(corp_name, 'r', encoding='utf8') as fin:
    lines = fin.readlines()
    nan_count = 0
    for line in lines:
      line = line.replace('\n', '')
      if len(line) < 1:
        nan_count += 1
      sentences.append(line.split(' '))
    print('num of nan row', nan_count)
  # 求最长分词字符串的词数
  max_document_length = max([len(item) for item in sentences])
  print('Max sequence length', max_document_length)

  for i in range(len(sentences)):
    if len(sentences[i]) > max_document_length:
      sentences[i] = sentences[i][:max_document_length]

  # Processing input data with VocabularyProcessor
  # vocab_processor = learn.preprocessing.VocabularyProcessor(
  #   max_document_length)
  # print('Word to index done.')

  print('Start training...', time.asctime())
  sentences = word2vec.Text8Corpus(corp_name)
  model = word2vec.Word2Vec(sentences, sg=sg)
  model.save(md_name)
  print('Saved model done.', time.asctime())
  # get word index from model
  # id = model.wv.vocabulary[word].index


if __name__ == '__main__':
  # gen_wordvec('/home/lian/data/nlp/datagrand_info_extra/corpus_pretr.txt',
  #             '../../model/datagrand_corpus_pretrain.bin')

# model = word2vec.Word2Vec.load('../../model/datagrand_corpus_pretrain.bin')
# idx = sorted(model.wv.index2word)[:100]
# print(len(model.wv.index2word))

# sentences = [["平行线", "直线", "这条"], ["曲线", "弧长", "曲率"], ['<UNK>']]
# vocab_processor = learn.preprocessing.VocabularyProcessor(2)
# print([' '.join(line) for line in sentences])
# print(list(vocab_processor.fit_transform([' '.join(line) for line in sentences])))
# print(vocab_processor.vocabulary_._mapping)
