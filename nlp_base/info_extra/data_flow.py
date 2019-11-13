# coding:utf8

import numpy as np


def data_pre_flow():
  from info_extra.data_helper import word2ind_with_seqlen, get_w2i_map, word2ind_without_seg

  word_set = get_w2i_map('/home/lian/data/nlp/datagrand_info_extra/train.txt')
  print('word set', word_set)
  # sentences, tags = word2ind('/home/lian/data/nlp/datagrand_info_extra/train.txt', word_set)
  sentences, tags, seq_lens = word2ind_with_seqlen('/home/lian/data/nlp/datagrand_info_extra/train.txt', word_set)
  sentences = np.array(sentences)
  tags = np.array(tags)
  seq_lens = np.array(seq_lens)
  print('Batch sequence length', seq_lens)

  test_sentences = word2ind_without_seg('/home/lian/data/nlp/datagrand_info_extra/test.txt', word_set)

  np.savetxt('/home/lian/data/nlp/datagrand_info_extra/swap/word_set.txt', np.array(word_set), fmt='%s')
  np.savetxt('/home/lian/data/nlp/datagrand_info_extra/swap/sentences.txt', np.array(sentences), fmt='%d')
  np.savetxt('/home/lian/data/nlp/datagrand_info_extra/swap/tags.txt', np.array(tags), fmt='%d')
  np.savetxt('/home/lian/data/nlp/datagrand_info_extra/swap/seq_lens.txt', np.array(seq_lens), fmt='%d')
  np.savetxt('/home/lian/data/nlp/datagrand_info_extra/swap/test_sentences.txt', np.array(test_sentences), fmt='%d')
  print('Saved done.')


def model_flow(infer=False):
  from info_extra.data_helper import test_data_iterate, data_iterate_with_seqlen

  word_set = np.loadtxt('/home/lian/data/nlp/datagrand_info_extra/swap/word_set.txt', dtype=np.object)
  sentences = np.loadtxt('/home/lian/data/nlp/datagrand_info_extra/swap/sentences.txt', dtype=np.int)
  tags = np.loadtxt('/home/lian/data/nlp/datagrand_info_extra/swap/tags.txt', dtype=np.int)
  seq_lens = np.loadtxt('/home/lian/data/nlp/datagrand_info_extra/swap/seq_lens.txt', dtype=np.int)

  train_initializer, dev_initializer, _, iterator = data_iterate_with_seqlen(sentences, tags, seq_lens,
                                                                             batch_size=100, mode='train')
  test_initializer, test_iterator = test_data_iterate(sentences, 100)

  from info_extra.sequence_labelling import ModelAssist
  model = ModelAssist()

  if not infer:
    model.train(iterator, train_initializer, dev_initializer, epoch_num=100, step_num=1000,
                save_when_loss=2e-1, dev_step=10)
  else:
    model.inference(test_initializer, word_set)

  print('All done.')


if __name__ == '__main__':
  # data_pre_flow()
  model_flow()
