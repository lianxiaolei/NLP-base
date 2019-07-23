# coding:utf8

import datetime
import time
from tensorflow.python.framework import graph_util
from tensorflow.contrib import rnn
from nlp_base.tools.data_helper import *
from nlp_base.tools import get_tensorflow_conf


# CHECKPOINT_PATH = "/home/lian/PycharmProjects/seq2seq/checkpoint/seq2seq_ckpt"  # checkpoint保存路径。

class SequenceTagging(object):
  def __init__(self, vocab_size, embedding_size, rnn_units, max_seq_length=64, rnn_layer_num=2):
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.rnn_units = rnn_units
    self.max_seq_length = max_seq_length
    self.rnn_layer_num = 2
    self.x_test = None

    # TODO Initialize embedding matrix.
    self.embedding = tf.Variable(
      tf.random_uniform([self.vocab_size, self.embedding_size], -1., 1.),
      name='embedding', trainable=False)

    self.FLAGS = tf.flags.FLAGS

    config = tf.ConfigProto(
      allow_soft_placement=self.FLAGS.allow_soft_placement,
      log_device_placement=self.FLAGS.log_device_placement)

    self.sess = tf.Session(config=config)

  def init_wemb(self, value):
    """
    Initialize global embedding matrix.
    Args:
      value: A ndarray with shape [vocab_size, word_dim],
        suggest train word vector with gensim-skipgram.

    """
    # self.sess.run(tf.assign(self.embedding, value, name='init_wemb'))
    self.embedding = tf.assign(self.embedding, value, name='inited_wemb')

  def _build_lookup(self, src):
    """
    Get lookup embedding tensor from input ids.
    Args:
      src: A Tensor contains words' ids.

    """
    with tf.name_scope('lookup'):
      self.lookup = tf.nn.embedding_lookup(self.embedding, src)

  def _build_rnn_with_keras(self):
    """
    Build a BiLSTM to capture the sequence information.

    """
    gru_1 = tf.keras.layers.GRU(self.rnn_units, return_sequences=True,
                                kernel_initializer='he_normal',
                                name='gru1')(self.lookup)
    gru_1b = tf.keras.layers.GRU(self.rnn_units, return_sequences=True, go_backwards=True,
                                 kernel_initializer='he_normal',
                                 name='gru1_b')(self.lookup)
    # self.outputs = tf.keras.layers.add([gru_1, tf.reverse(gru_1b, axis=[1])])  # [batch, words, units]
    self.merged_gru = tf.keras.layers.add([gru_1, gru_1b])  # [batch, words, units]

    gru_2 = tf.keras.layers.GRU(self.rnn_units, return_sequences=True,
                                kernel_initializer='he_normal',
                                name='gru2')(self.merged_gru)
    gru_2b = tf.keras.layers.GRU(self.rnn_units, return_sequences=True, go_backwards=True,
                                 kernel_initializer='he_normal',
                                 name='gru2_b')(self.merged_gru)

    self.outputs = tf.keras.layers.concatenate([gru_2, gru_2b])  # [batch, height, units * 2]

  def _build_rnn(self):
    """
    Useless
    Returns:

    """

    def lstm_cell():
      cell = rnn.LSTMCell(self.rnn_units, reuse=tf.get_variable_scope().reuse)
      return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    # ** 1.构建前向后向多层 LSTM
    cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(self.rnn_layer_num)], state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_cell() for _ in range(self.rnn_layer_num)], state_is_tuple=True)

    # ** 2.初始状态
    initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

    # ** 3. bi-lstm 计算（展开）
    with tf.variable_scope('bidirectional_rnn'):
      # *** 下面，两个网络是分别计算 output 和 state
      # Forward direction
      outputs_fw = list()
      state_fw = initial_state_fw
      with tf.variable_scope('fw'):
        for timestep in range(self.max_seq_length):
          if timestep > 0:
            tf.get_variable_scope().reuse_variables()
          (output_fw, state_fw) = cell_fw(self.outputs[:, timestep, :], state_fw)  # The outputs is lookup.
          outputs_fw.append(output_fw)

      # backward direction
      outputs_bw = list()
      state_bw = initial_state_bw
      with tf.variable_scope('bw') as bw_scope:
        inputs = tf.reverse(self.src, [1])
        for timestep in range(self.max_seq_length):
          if timestep > 0:
            tf.get_variable_scope().reuse_variables()
          (output_bw, state_bw) = cell_bw(self.outputs[:, timestep, :], state_bw)
          outputs_bw.append(output_bw)
      # *** 然后把 output_bw 在 timestep 维度进行翻转
      # outputs_bw.shape = [timestep_size, batch_size, hidden_size]
      outputs_bw = tf.reverse(outputs_bw, [0])
      # 把两个oupputs 拼成 [timestep_size, batch_size, hidden_size*2]
      output = tf.concat([outputs_fw, outputs_bw], 2)
      output = tf.transpose(output, perm=[1, 0, 2])
      # output = tf.reshape(output, [-1, self.rnn_units * 2])
      self.outputs = output

  def build_net(self, src, src_size, target):
    """
    Build network architecture and compile model with softmax_cross_entropy loss.
    Args:
      src: Input Tensor from TFRecord.
      src_size: Every record's length.
      target: A tensor of record tagging.

    """
    self.batch_size = self.FLAGS.batch_size
    self.src = src
    self.src_size = src_size
    self.target = target

    self.keep_prob = self.FLAGS.dropout_keep_prob

    with self.sess.as_default():
      # embedding
      self._build_lookup(src)

      # RNN
      self._build_rnn_with_keras()

      # Tail to logits
      self.outputs = tf.keras.layers.Dense(self.rnn_units * 4, activation='relu')(self.outputs)
      self.score = tf.keras.layers.Dense(self.num_tag, activation='softmax')(self.outputs)
      self.logits = self.score
      # self.logits = tf.nn.softmax(self.score, name='logits')

      self.argmax = tf.argmax(self.logits, axis=-1)

      # Define the sequence mask.
      self.seq_mask = tf.sequence_mask(self.src_size, tf.shape(self.target)[1], dtype=tf.float32)

  def compile(self):
    """
    Compile model, define loss and optimizer to take the gradient descent.

    """
    # Define the loss operator
    with tf.name_scope('loss'):
      print('logits shape', self.logits.shape, 'target shape', self.target.shape)
      self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=self.target,
                                                                 name='sparse_cross_entropy')
      # Mean of loss.
      # self.cost = tf.reduce_mean(self.loss, name='reduce_mean')

      self.cost = tf.reduce_sum(self.loss * self.seq_mask)
      self.cost = self.cost / tf.reduce_sum(self.seq_mask)

      self.optimize()

  def compile_with_crf(self):
    with tf.name_scope('crf_loss'):
      log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.target,
                                                                            self.src_size)
      self.cost = tf.reduce_mean(-log_likelihood)

      self.optimize()

  def optimize(self):
    """
    Optimizer
    Returns:

    """
    # Defind the optimizer
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    # self.learning_rate = tf.train.exponential_decay(self.FLAGS.lr, self.global_step,
    #                                                 decay_steps=256, decay_rate=0.86)
    self.learning_rate = self.FLAGS.lr
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.grads_and_vars = self.optimizer.compute_gradients(self.cost)
    self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, self.global_step)

  def gd(self):
    """
    Useless
    Returns:

    """
    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost / tf.to_float(self.batch_size), trainable_variables)
    # clip gradients, return gradients,
    #   using this method, we cann't need compute_gradient method.
    grads, _ = tf.clip_by_global_norm(grads, 5.)

    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

    self.train_op = self.optimizer.apply_gradients(zip(grads, trainable_variables))

  def inference(self):
    """
    Make a inference.
    Returns:
      A ndarray, the most likely tags for a given document.
    """
    return self.sess.run([self.argmax, self.label])

  def summary(self):
    """
    Summary

    """
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, 'run', timestamp))

    # Summary for loss and accuracy
    # loss_summary = tf.summary.scalar('loss', self.loss)
    # acc_summary = tf.summary.scalar('accuracy', self.accuracy)

    # Train summaries
    train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
    self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

    # Dev summaries
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)

  def current_step(self):
    """
    Get current training steps.
    Returns:

    """
    return tf.train.global_step(self.sess, self.global_step)

  def train_step(self, saver, step):
    while True:
      try:
        _, cost = self.sess.run([self.train_op, self.cost])
        if step % 10 == 0:
          print("After %d steps, per token cost is %.3f" % (step, cost))

        # Generate checkpoint every 200 train steps.
        if step % 200 == 0:
          saver.save(self.sess, self.FLAGS.checkpoint_path, global_step=step)
        step += 1
      except tf.errors.OutOfRangeError:
        print('All data has been used.')
        break
    return step

  def restore(self, checkpoint_path=None):
    if not checkpoint_path:
      checkpoint_path = self.FLAGS.checkpoint_path

    # saver = tf.train.import_meta_graph('/home/lian/PycharmProjects/NLP-base/model/checkpoint/seqtag_ckpt-1000.meta')
    saver = tf.train.Saver()

    self.model = saver.restore(self.sess,
                               tf.train.latest_checkpoint('/home/lian/PycharmProjects/NLP-base/model/checkpoint'))
    print('Load model successfully.')

  def run(self, inference=False):
    self.num_tag = self.FLAGS.num_tag
    data = gen_src_tar_dataset(self.FLAGS.train_file, self.FLAGS.target_file, self.FLAGS.batch_size)
    iterator = data.make_initializable_iterator()
    (src, src_size), trg_label = iterator.get_next()

    # 定义前向计算图。输入数据以张量形式提供给forward函数。
    self.build_net(src, src_size, trg_label)

    if not inference:
      self.compile()
      saver = tf.train.Saver()
      step = 0
      with self.sess.as_default() as sess:
        tf.global_variables_initializer().run()
        for i in range(self.FLAGS.num_epochs):
          print("In iteration: %d" % (i + 1))
          sess.run(iterator.initializer)
          step = self.train_step(saver, step)
        return None
    else:
      self.label = trg_label
      with self.sess.as_default() as sess:
        self.restore()
        tf.global_variables_initializer().run()
        sess.run(iterator.initializer)
        print('Interator has been initialized.')
        pred, label = self.inference()
        return pred, label

  def experiment(self):
    data = gen_src_tar_dataset(self.FLAGS.train_file, self.FLAGS.target_file, self.FLAGS.batch_size)
    iterator = data.make_initializable_iterator()
    (src, src_size), trg_label = iterator.get_next()
    mask = tf.sequence_mask(src_size, dtype=tf.int32)
    masked_trg = trg_label * mask
    with self.sess.as_default() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(iterator.initializer)
      src_arr, src_size_arr, trg_label_arr, masked_trg_arr = sess.run([src, src_size, trg_label, masked_trg])
    print('src_arr', src_arr)
    print('src_size_arr', src_size_arr)
    print('trg_label_arr', trg_label_arr)
    print('mask_trg', masked_trg_arr)
    np.savetxt('masked.txt', masked_trg_arr, fmt='%.0f')


if __name__ == '__main__':
  FLAGS = get_tensorflow_conf(tf)

  print('Building model...')
  network = SequenceTagging(
    vocab_size=21350,
    embedding_size=100,
    rnn_units=128
  )

  init_w = load_embedding_vectors_word2vec('../../model/datagrand_corpus_pretrain.bin', FLAGS.embedding_dim)

  network.init_wemb(init_w)

  # network.experiment()

  # Build and compile network
  pred, label = network.run(inference=False)
  # Add summaries to graphy
  # network.summary()
  # print(pred.shape, label.shape)
  # print("pred", pred)
  # print("labl", label)
  # np.savetxt('pred.txt', pred.astype(np.int), fmt='%.0f')
  # np.savetxt('labl.txt', label.astype(np.int), fmt='%.0f')
  # print("accu", np.sum(pred[pred > 0] == label[label > 0]) / (np.sum(label > 0)))
