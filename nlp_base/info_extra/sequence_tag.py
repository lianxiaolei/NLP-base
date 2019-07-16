# coding:utf8

import datetime
import time
from tensorflow.python.framework import graph_util
from tensorflow.contrib import rnn
from nlp_base.tools.data_helper import *
from nlp_base.tools import get_tensorflow_conf

# CHECKPOINT_PATH = "/home/lian/PycharmProjects/seq2seq/checkpoint/seq2seq_ckpt"  # checkpoint保存路径。
CHECKPOINT_PATH = "/Users/lianxiaohua/PycharmProjects/NLP-base/model/checkpoint/seq2seq_ckpt"  # checkpoint保存路径。


class SequenceTagging(object):
  def __init__(self, sequence_length,
               vocab_size, embedding_size,
               rnn_units, l2_reg_lambda=0.0):
    self.vocab_size = vocab_size
    self.sequence_length = sequence_length
    self.embedding_size = embedding_size
    self.rnn_units = rnn_units
    self.l2_reg_lambda = l2_reg_lambda
    self.x_test = None

    # TODO Initialize embedding matrix.
    self.embedding = tf.Variable(
      tf.random_uniform([self.vocab_size, self.embedding_size], -1., 1.),
      name='embedding')

  def init_wemb(self, value):
    """
    Initialize global embedding matrix.
    Args:
      value: A ndarray with shape [vocab_size, word_dim],
        suggest train word vector with gensim-skipgram.

    """
    self.sess.run(tf.assign(self.embedding, value, name='init_wemb'))

  def _build_lookup(self, src):
    """
    Get lookup embedding tensor from input ids.
    Args:
      src: A Tensor contains words' ids.

    """
    with tf.name_scope('lookup'):
      self.embedding_chars = tf.nn.embedding_lookup(self.embedding, src)

  def _build_rnn(self):
    """
    Build a BiLSTM to capture the sequence information.

    """
    rnn_cell_fw = rnn.LSTMCell(self.rnn_units)
    rnn_cell_bw = rnn.LSTMCell(self.rnn_units)
    initial_state_fw = rnn_cell_fw.zero_state(self.batch_size, dtype=tf.float32)
    initial_state_bw = rnn_cell_bw.zero_state(self.batch_size, dtype=tf.float32)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,
                                                      rnn_cell_bw,
                                                      self.embedding_chars,
                                                      self.src_size,
                                                      initial_state_fw,
                                                      initial_state_bw)

    self.rnn_outputs = tf.add(outputs[0], outputs[1])

  def build_net(self, src, src_size, target):
    """
    Build network architecture and compile model with softmax_cross_entropy loss.
    Args:
      src: Input Tensor from TFRecord.
      src_size: Every record's length.
      target: A tensor of record tagging.

    """
    self.batch_size = self.FLAGS.batch_size
    self.src_size = src_size

    self.keep_prob = self.FLAGS.dropout_keep_prob
    self.learning_rate = self.FLAGS.lr

    self.l2_loss = tf.constant(0.0)

    config = tf.ConfigProto(
      allow_soft_placement=self.FLAGS.allow_soft_placement,
      log_device_placement=self.FLAGS.log_device_placement)

    self.sess = tf.Session(config=config)

    with self.sess.as_default():
      # embedding
      self._build_lookup(src)

      # RNN
      self._build_rnn()

      # Tail to logits
      self.score = tf.keras.layers.Dense(self.num_tag, use_bias=True)(self.rnn_outputs)
      self.logits = tf.nn.softmax(self.score, name='logits')

      # Reshape label from shape [batch_size,] to shape [batch_size, 1]
      self.target = target
      self.argmax = tf.argmax(self.logits, axis=-1)

      # Define the loss operator
      with tf.name_scope('loss'):
        print('logits shape', self.logits.shape, 'target shape', target.shape)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                   labels=target,
                                                                   name='sparse_cross_entropy')
        # Mean of loss.
        self.cost = tf.reduce_mean(self.loss, name='reduce_mean')

      # Define the accuracy operator
      with tf.name_scope('accuracy'):
        correct = tf.equal(tf.cast(self.argmax, tf.int32), self.target)
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'),
                                       name='accuracy')
      # Defind the optimizer
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      self.optimizer = tf.train.AdamOptimizer(self.FLAGS.lr)
      self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
      self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, self.global_step)

      self.sess.run(tf.global_variables_initializer())

  def inference(self, documents):
    """

    Args:
      documents:

    Returns:

    """
    pass

  def summary(self):
    """
    Summary

    """
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, 'run', timestamp))

    # Summary for loss and accuracy
    loss_summary = tf.summary.scalar('loss', self.loss)
    acc_summary = tf.summary.scalar('accuracy', self.accuracy)

    # Train summaries
    self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
    self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

    # Dev summaries
    self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)

  def checkpoint(self, out_dir, verbose=True):
    """
    Generate model checkpoint.
    Args:
      out_dir: A string, output path.
      verbose: A bool, whether to print information.
    """
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
    self.checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

    if not os.path.exists(checkpoint_dir):
      os.mkdir(checkpoint_dir)
    self.saver = tf.train.Saver(tf.all_variables())
    path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=self.global_step)
    if verbose: print("Saved model checkpoint to {}\n".format(path))

  def save_pb(self, pb_path):
    """
    Save model as a serialized pb file.
    Args:
      pb_path: Pb file path.

    """
    # Method convert_variables_to_constants need to fix output_node_names with type list.
    constant_graph = graph_util.convert_variables_to_constants(
      self.sess, self.sess.graph_def, ['op_to_store'])

    # Write to serialized PB file
    with tf.gfile.FastGFile(pb_path + 'model.pb', mode='wb') as f:
      f.write(constant_graph.SerializeToString())

    # Save model builder
    builder = tf.saved_model.builder.SavedModelBuilder(pb_path + 'savemodel')
    # 构造模型保存的内容，指定要保存的 session，特定的 tag,
    # 输入输出信息字典，额外的信息
    builder.add_meta_graph_and_variables(self.sess, ['cpu_server_1'])

    # Add the second MetaGraphDef
    # with tf.Session(graph=tf.Graph()) as sess:
    # ...
    # builder.add_meta_graph([tag_constants.SERVING])
    # ...

    builder.save()

  def load_pb(self, pb_path):
    with tf.Session(graph=tf.Graph()) as sess:
      tf.saved_model.loader.load(sess, ['cpu_1'], pb_path + 'savemodel')
      sess.run(tf.global_variables_initializer())

      input_x = sess.graph.get_tensor_by_name('x:0')
      input_y = sess.graph.get_tensor_by_name('y:0')

      op = sess.graph.get_tensor_by_name('op_to_store:0')

  def current_step(self):
    """
    Get current training steps.
    Returns:

    """
    return tf.train.global_step(self.sess, self.global_step)

  def for_each_epoch(self, session, cost_op, train_op, saver, step):
    while True:
      try:
        cost, _, acc = session.run([cost_op, train_op, self.accuracy])
        if step % 10 == 0:
          print("After %d steps, per token cost is %.3f, acc is %.3f" % (step, cost, acc))

        # 每200步保存一个checkpoint。
        if step % 200 == 0:
          saver.save(session, self.FLAGS.checkpoint_path, global_step=step)
        step += 1
      except tf.errors.OutOfRangeError:
        print('All data has been used.')
        break
    return step

  def run(self):
    self.FLAGS = tf.flags.FLAGS
    self.num_tag = self.FLAGS.num_tag
    data = gen_src_tar_dataset(self.FLAGS.train_file, self.FLAGS.target_file, self.FLAGS.batch_size)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()
    print('src shape', src.shape)
    # 定义前向计算图。输入数据以张量形式提供给forward函数。
    self.build_net(src, src_size, trg_label)

    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      for i in range(self.FLAGS.num_epochs):
        print("In iteration: %d" % (i + 1))
        sess.run(iterator.initializer)
        step = self.for_each_epoch(sess, self.cost, self.train_op, saver, step)


if __name__ == '__main__':
  FLAGS = get_tensorflow_conf(tf)

  print('Building model...')
  network = SequenceTagging(
    vocab_size=10600,
    sequence_length=800,
    embedding_size=200,
    rnn_units=128
  )

  init_w = load_embedding_vectors_word2vec('../../model/datagrand_corpus_pretrain.bin', FLAGS.embedding_dim)

  network.init_wemb(init_w)

  # Build and compile network
  network.run()
  # Add summaries to graph
  network.summary()

  # Save model
  # network.checkpoint(out_dir)
