# coding:utf8

import tensorflow as tf

SRC_TRAIN_DATA = "/home/lian/PycharmProjects/seq2seq/dataset/train.en"  # 源语言输入文件。
TRG_TRAIN_DATA = "/home/lian/PycharmProjects/seq2seq/dataset/train.zh"  # 目标语言输入文件。
CHECKPOINT_PATH = "/home/lian/PycharmProjects/seq2seq/checkpoint/seq2seq_ckpt"  # checkpoint保存路径。

HIDDEN_SIZE = 1024  # LSTM的隐藏层规模。
NUM_LAYERS = 2  # 深层循环神经网络中LSTM结构的层数。
SRC_VOCAB_SIZE = 10000  # 源语言词汇表大小。
TRG_VOCAB_SIZE = 4000  # 目标语言词汇表大小。
BATCH_SIZE = 100  # 训练数据batch的大小。
NUM_EPOCH = 5  # 使用训练数据的轮数。
KEEP_PROB = 0.8  # 节点不被dropout的概率。
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的梯度大小上限。
SHARE_EMB_AND_SOFTMAX = True  # 在Softmax层和词向量层之间共享参数。

MAX_LEN = 50  # 限定句子的最大单词数量。
# SOS_ID = 1  # 目标语言词汇表中<sos>的ID。


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
  ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
  src_len_ok = tf.logical_and(
    tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
  trg_len_ok = tf.logical_and(
    tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
  return tf.logical_and(src_len_ok, trg_len_ok)


def gen_target_input(src_tuple, trg_tuple):
  """

  Args:
    src_tuple:
    trg_tuple:

  Returns:

  """
  ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
  trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
  return (src_input, src_len), (trg_input, trg_label, trg_len)


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
    (tf.TensorShape([None]),  # tar sentence(input) is a dynamic length vector.
     tf.TensorShape([None]),  # tar sentence(output) is a dynamic length vector.
     tf.TensorShape([])))  # tar sentence's length is a int value.

  # Call padded_batch to make batch.
  batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
  return batched_dataset
