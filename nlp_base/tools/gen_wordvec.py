# coding:utf8

from gensim.models import word2vec


def gen_wordvec(corp_name, md_name, sg=1):
  sentences = word2vec.Text8Corpus(corp_name)
  model = word2vec.Word2Vec(sentences, sg=sg)
  model.save(md_name)


if __name__ == '__main__':
  # gen_wordvec('/home/lian/data/nlp/datagrand_info_extra/corpus_sliced.txt',
  #             'datagrand_corpus_pretrain.bin')

  model = word2vec.Word2Vec.load('../../model/datagrand_corpus_pretrain.bin')
  idx = sorted(model.wv.index2word)[:100]
  print(len(model.wv.index2word))
