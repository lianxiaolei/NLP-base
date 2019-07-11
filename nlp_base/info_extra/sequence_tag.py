# coding:utf8

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time
import os
import datetime


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
    self.flag = 0

  def init_wemb(self, value):
    self.sess.run(tf.assign(self.w_emb, value, name='init_wemb'))

  def _build_embedding(self, inputs, wordvecs=None):
    with tf.name_scope('embedding'):
      self.w_emb = tf.Variable(
        tf.random_uniform([self.vocab_size, self.embedding_size], -1., 1.),
        name='w_emb')
      if wordvecs is not None:
        self.init_wemb(wordvecs)
      # 获取下标为x的w_emb中的内容
      self.embedded_chars = tf.nn.embedding_lookup(self.w_emb, inputs)

      self.embedding_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

  def _build_rnn(self, inputs):
    rnn_cell_fw = rnn.LSTMCell(self.rnn_units)
    rnn_cell_bw = rnn.LSTMCell(self.rnn_units)
    initial_state_fw = rnn_cell_fw.zero_state(self.batch_size, dtype=tf.float32)
    initial_state_bw = rnn_cell_bw.zero_state(self.batch_size, dtype=tf.float32)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,
                                                      rnn_cell_bw,
                                                      inputs,
                                                      self.sequence_length,
                                                      initial_state_fw,
                                                      initial_state_bw)

    return outputs

  def build_net(self, inputs, batch_size=128, wordvecs=None):
    self.FLAGS = tf.flags.FLAGS
    self.batch_size = batch_size

    with tf.Graph().as_default():
      with tf.name_scope(name='ph'):
        self.x = tf.placeholder(tf.int32, [None, self.sequence_length], name='x')
        self.y = tf.placeholder(tf.int32, [None, self.sequence_length], name='y')

        self.keep_prob = tf.placeholder(tf.float32, name='kp')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

      self.l2_loss = tf.constant(0.0)

      config = tf.ConfigProto(
        allow_soft_placement=self.FLAGS.allow_soft_placement,  # 设置让程序自动选择设备运行
        log_device_placement=self.FLAGS.log_device_placement)

      self.sess = tf.Session(config=config)

      with self.sess.as_default():
        self._build_embedding(inputs, wordvecs)
        rnn_outputs = self._build_rnn(self.embedding_chars_expanded)

        self.score = tf.python.keras.layers.Dense(self.sequence_length,
                                                  use_bias=True)(rnn_outputs)
        self.prediction = tf.argmax(self.score, axis=1, name='pred')

        # Define the loss operator
        with tf.name_scope('loss'):
          loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.score,
                                                         labels=self.y,
                                                         name='cross_entropy')
          self.loss = tf.reduce_mean(loss, name='reduce_mean')

        # Define the accuracy operator
        with tf.name_scope('accuracy'):
          correct = tf.equal(self.prediction, tf.argmax(self.y, 1))
          self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'),
                                         name='accuracy')
        # Defind the optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, self.global_step)

        self.sess.run(tf.global_variables_initializer())

  def summary(self):
    # Summary
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, 'run', timestamp))
    print('Writing to {}\n'.format(out_dir))

    # Summary for loss and accuracy
    loss_summary = tf.summary.scalar('loss', self.loss)
    acc_summary = tf.summary.scalar('accuracy', self.accuracy)
    # replace with tf.summary.scalar

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in self.grads_and_vars:
      if g is not None:
        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Train summaries
    # self.train_summary_op = tf.contrib.deprecated.merge_summary([loss_summary, acc_summary])
    self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
    # self.train_summary_writer = tf.contrib.summary.SummaryWriter(train_summary_dir, self.sess.graph_def)
    self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

    # Dev summaries
    # self.dev_summary_op = tf.contrib.deprecated.merge_summary([loss_summary, acc_summary])
    self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    # self.dev_summary_writer = tf.contrib.summary.SummaryWriter(dev_summary_dir, self.sess.graph_def)
    self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)

  def checkpoint(self, out_dir):
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
    self.checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

    if not os.path.exists(checkpoint_dir):
      os.mkdir(checkpoint_dir)
    self.saver = tf.train.Saver(tf.all_variables())
    path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=self.global_step)
    print("Saved model checkpoint to {}\n".format(path))

  def train_step(self, x_batch, y_batch):
    feed_dict = {
      self.x: x_batch,  # placeholder
      self.y: y_batch,  # placeholder
      self.keep_prob: self.FLAGS.dropout_keep_prob
    }

    _, step, summaries, loss, accuracy = self.sess.run(
      [self.train_op, self.global_step,
       self.train_summary_op, self.loss, self.accuracy],
      feed_dict=feed_dict
    )

    time_str = datetime.datetime.now().isoformat()
    print("{}:step:{},loss:{:g},acc:{:g}".format(time_str, step, loss, accuracy))

    self.train_summary_writer.add_summary(summaries, step)
    if step % self.FLAGS.batch_size == 0:
      print('epoch:{}'.format(step // self.FLAGS.batch_size))

  def dev_step(self, x_batch, y_batch):
    feed_dict = {
      self.x: x_batch,  # placeholder
      self.y: y_batch,  # placeholder
      self.keep_prob: 1.
    }

    _, step, summaries, loss, accuracy = self.sess.run(
      [self.train_op, self.global_step,
       self.dev_summary_op, self.loss, self.accuracy],
      feed_dict=feed_dict
    )

    time_str = datetime.datetime.now().isoformat()
    print("{}:step:{},loss:{:g},acc:{:g}".format(time_str, step, loss, accuracy))

    self.dev_summary_writer.add_summary(summaries, step)
    # if step % FLAGS.batch_size == 0:
    #     print('epoch:{}'.format(step // FLAGS.batch_size))

    if accuracy > 0.8 and self.flag == 0 or accuracy > 0.9:
      pred = self.sess.run(self.prediction,
                           feed_dict={self.x: self.x_test[:, :self.sequence_length],
                                      self.keep_prob: 1.})
      np.savetxt('result.txt', pred)
      print('save done')

  def current_step(self):
    return tf.train.global_step(self.sess, self.global_step)

  def save_checkpoint(self, step):
    return self.saver.save(self.sess, self.checkpoint_prefix, global_step=step)

  def run(self, x_train, y_train, x_dev, y_dev, dh):
    for epoch in range(100):
      # Generate batches
      batches = dh.batch_iter(
        list(zip(x_train, y_train)), self.FLAGS.batch_size, self.FLAGS.num_epochs)
      # print('x_train:', x_train[0: 1])
      # print('x_dev:', x_dev[0: 1])
      # Training loop. For each batch...
      for batch in batches:
        try:
          x_batch, y_batch = zip(*batch)
        except Exception as e:
          continue
        self.train_step(x_batch, y_batch)

        current_step = tf.train.global_step(self.sess, self.global_step)
        if current_step % self.FLAGS.evaluate_every == 0:
          print("\nEvaluation:")
          start = np.random.randint(0, y_dev.shape[0] - self.FLAGS.batch_size + 1)
          end = start + self.FLAGS.batch_size
          self.dev_step(x_dev[start: end], y_dev[start: end])
          print('ev done')
        # if current_step % self.FLAGS.checkpoint_every == 0:
        #     self.checkpoint(out_dir)
        # path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
        # print("Saved model checkpoint to {}\n".format(path))


if __name__ == '__main__':
  # Data loading params
  tf.app.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
  # tf.app.flags.DEFINE_string("train_file", "../../dataset/cnews.train.txt", "Train file source.")
  tf.app.flags.DEFINE_string("train_file", "../../dataset/train.csv", "Train file source.")
  # tf.app.flags.DEFINE_string("test_file", "../../dataset/cnews.val.txt", "Test file source.")
  tf.app.flags.DEFINE_string("test_file", "../../dataset/test.csv", "Test file source.")

  # Model Hyperparameters
  tf.app.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
  tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
  tf.app.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
  tf.app.flags.DEFINE_float("dropout_keep_prob", 0.1, "Dropout keep probability (default: 0.5)")
  tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

  # Training parameters
  tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
  tf.app.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
  tf.app.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
  tf.app.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
  tf.app.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
  # Misc Parameters
  tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
  tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

  FLAGS = tf.app.flags.FLAGS

  dh = DataHelper()

  print('Loading data...')
  x_train, y_train = dh.load_train_dev_data(FLAGS.train_file)
  print('Data shape:', x_train.shape, y_train.shape)

  # 大家伙，不能输出
  # print('vocabulary:', vocab_processor.vocabulary_._mapping)

  network = SequenceTagging(
    vocab_size=9999,
    sequence_length=x_train.shape[1],
    embedding_size=200,
    rnn_units=128
  )

  init_w = dh.load_embedding_vectors_word2vec('../../model/datagrand_corpus.bin')

  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False)

  # Build and compile network
  network.build_net()
  # Add summaries to graph
  network.summary()

  # 加载测试集
  network.x_test = dh.load_test_data(FLAGS.test_file)
  network.run(x_train, y_train, x_val, y_val, dh)
  # Save model
  # network.checkpoint(out_dir)
