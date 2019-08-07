# coding:utf8

import tensorflow as tf
from seq2seq.utils.rnn_data_helper import *

# 假设输入数据已经用9.2.1小节中的方法转换成了单词编号的格式。
SRC_TRAIN_DATA = "/home/lian/PycharmProjects/seq2seq/dataset/train.en"  # 源语言输入文件。
TRG_TRAIN_DATA = "/home/lian/PycharmProjects/seq2seq/dataset/train.zh"  # 目标语言输入文件。
CHECKPOINT_PATH = "/home/lian/PycharmProjects/seq2seq/checkpoint/seq2seq_ckpt"  # checkpoint保存路径。

WORD_DIM = 1024  # 词向量维度。
NUM_LAYERS = 2  # 深层循环神经网络中LSTM结构的层数。
SRC_VOCAB_SIZE = 10000  # 源语言词汇表大小。
TRG_VOCAB_SIZE = 4000  # 目标语言词汇表大小。
BATCH_SIZE = 100  # 训练数据batch的大小。
NUM_EPOCH = 5  # 使用训练数据的轮数。
KEEP_PROB = 0.8  # 节点不被dropout的概率。
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的梯度大小上限。
SHARE_EMB_AND_SOFTMAX = True  # 在Softmax层和词向量层之间共享参数。

MAX_LEN = 50  # 限定句子的最大单词数量。
SOS_ID = 1  # 目标语言词汇表中<sos>的ID。


class NMTModel(object):
  """

  """

  def __init__(self):
    # self.graph = tf.Graph()

    self.initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # with self.graph.as_default() as graph:
    # print('Initlized graph:', graph)
    with tf.name_scope('Embedding'):
      self.src_emb = tf.get_variable('src_emb', (SRC_VOCAB_SIZE, WORD_DIM), initializer=initializer)
      self.trg_emb = tf.get_variable('trg_emb', (TRG_VOCAB_SIZE, WORD_DIM), initializer=initializer)

    if SHARE_EMB_AND_SOFTMAX:
      self.softmax_weight = tf.transpose(self.trg_emb)
    else:
      self.softmax_weight = tf.get_variable(
        "softmax_weight", [WORD_DIM, TRG_VOCAB_SIZE], initializer=initializer)
    self.softmax_bias = tf.get_variable(
      "softmax_bias", [TRG_VOCAB_SIZE], initializer=initializer)

    # 定义编码器和解码器所使用的LSTM结构。
    self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
      [tf.nn.rnn_cell.BasicLSTMCell(WORD_DIM)
       for _ in range(NUM_LAYERS)])
    self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
      [tf.nn.rnn_cell.BasicLSTMCell(WORD_DIM)
       for _ in range(NUM_LAYERS)])

  def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
    """
    在forward函数中定义模型的前向计算图。
    src_input, src_size, trg_input, trg_label, trg_size分别是上面
    MakeSrcTrgDataset函数产生的五种张量。
    Args:
      src_input:
      src_size:
      trg_input:
      trg_label:
      trg_size:

    Returns:

    """
    batch_size = tf.shape(src_input)[0]

    # with self.graph.as_default() as graph:
    # print('Forward graph', graph)
    with tf.name_scope('lookup'):
      src_emb = tf.nn.embedding_lookup(self.src_emb, src_input, name='src_lookup')
      trg_emb = tf.nn.embedding_lookup(self.trg_emb, trg_input, name='trg_lookup')

    src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
    trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

    with tf.variable_scope("encoder"):
      _, enc_state = tf.nn.dynamic_rnn(
        self.enc_cell, src_emb, src_size, dtype=tf.float32)

    with tf.variable_scope("decoder"):
      dec_outputs, _ = tf.nn.dynamic_rnn(
        self.dec_cell, trg_emb, trg_size, initial_state=enc_state)

    # Calculate every time step's log perplexity.
    # Flatten the shape (batch, maxtime, units) to (batch * maxtime, units).
    output = tf.reshape(dec_outputs, [-1, WORD_DIM])
    # Make a project between every time step and every word in vocabulary.
    # The size of result is (batch * maxtime, len_vocab).
    logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias

    labels = tf.reshape(trg_label, [-1])

    with tf.name_scope('loss'):
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

      weight_mask = tf.sequence_mask(trg_size, tf.shape(trg_label)[1], dtype=tf.float32)

      weight_mask = tf.reshape(weight_mask, [-1])

      cost = tf.reduce_sum(loss * weight_mask)
      cost_per_token = cost / tf.reduce_sum(weight_mask)

    trainable_variables = tf.trainable_variables()

    with tf.name_scope('compile'):
      grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
      # clip gradients, return gradients,
      #   using this method, we cann't need compute_gradient method.
      grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

    return cost_per_token, train_op

  def run_epoch(self, session, cost_op, train_op, saver, step):
    """
      使用给定的模型model上训练一个epoch，并返回全局步数。
      每训练200步便保存一个checkpoint。
      训练一个epoch。
      重复训练步骤直至遍历完Dataset中所有数据。
    Args:
      session:
      cost_op:
      train_op:
      saver:
      step:

    Returns:

    """
    while True:
      try:
        cost, _ = session.run([cost_op, train_op])
        if step % 10 == 0:
          print("After %d steps, per token cost is %.3f" % (step, cost))
        # 每200步保存一个checkpoint。
        if step % 200 == 0:
          saver.save(session, CHECKPOINT_PATH, global_step=step)
        step += 1
      except tf.errors.OutOfRangeError:
        print('All data has been used.')
        break
    return step

  def run(self):
    # with self.graph.as_default() as graph:
    #   print('Running graph:', graph)
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    # 定义训练用的循环神经网络模型。
    # with tf.variable_scope("nmt_model", reuse=None, initializer=initializer):
      # 定义输入数据。
    data = gen_src_tar_dataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    # 定义前向计算图。输入数据以张量形式提供给forward函数。
    cost_op, train_op = self.forward(src, src_size, trg_input,
                                     trg_label, trg_size)

    # 训练模型。
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      for i in range(NUM_EPOCH):
        print("In iteration: %d" % (i + 1))
        sess.run(iterator.initializer)
        step = self.run_epoch(sess, cost_op, train_op, saver, step)


if __name__ == '__main__':
  initializer = tf.random_uniform_initializer(-0.05, 0.05)
  with tf.variable_scope("nmt_model", reuse=None, initializer=initializer):
    NMTModel().run()
