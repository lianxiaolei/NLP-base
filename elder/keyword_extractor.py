# coding:utf8


import jieba
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
import pandas as pd


def word_cut(doc):
  """
  切词
  :param doc:
  :return:2D list
  """
  result = []
  for line in doc:
    result.append(line.split())
  dictionary = corpora.Dictionary(result)
  return result, dictionary


def lda_extractor(corpus, dictionary, num_topics=1):
  lda = LdaModel(corpus=corpus,
                 id2word=dictionary,
                 num_topics=num_topics,
                 )
  return lda


if __name__ == '__main__':
  doc = pd.read_csv('../../dataset/020_7_shuffle_train.csv',
                    delimiter='\t',
                    header=None,
                    names=['content', 'label'])

  doc = doc['content']

  rubbish = [r'\[ / LaTeXI ]', r'\[', r'LaTeXI', r'\]', r'\{', r'\}',
             r'& gt ;', r'\\', r'frac', r'matrix', r'\(', r'\)', r'[0-9]']

  for rub in rubbish:
    doc.replace(rub, '', inplace=True, regex=True)

  doc = doc.values

  corpus, dictionary = word_cut(doc)
  print(dictionary)
