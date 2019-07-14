# coding:utf8

import datetime
import time

from tensorflow.contrib import rnn

from nlp_base.tools.data_helper import *

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
      # target = tf.expand_dims(self.targettarget, axis=-1)

      self.argmax = tf.argmax(self.logits, axis=-1)

      # Define the loss operator
      with tf.name_scope('loss'):
        print('logits shape', self.logits.shape, 'target shape', target.shape)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                   labels=target,
                                                                   name='sparse_cross_entropy')

        self.cost = tf.reduce_mean(self.loss, name='reduce_mean')

      # Define the accuracy operator
      with tf.name_scope('accuracy'):
        correct = tf.equal(tf.cast(self.argmax, tf.int32), target)
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
    self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
    self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

    # Dev summaries
    self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
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

    if accuracy > 0.8:
      pred = self.sess.run(self.logits,
                           feed_dict={self.x: self.x_test[:, :self.sequence_length],
                                      self.keep_prob: 1.})
      np.savetxt('result.txt', pred)
      print('save done')

  def current_step(self):
    return tf.train.global_step(self.sess, self.global_step)

  def save_checkpoint(self, step):
    return self.saver.save(self.sess, self.checkpoint_prefix, global_step=step)

  def run_epoch(self, session, cost_op, train_op, saver, step):
    while True:
      try:
        cost, _, acc = session.run([cost_op, train_op, self.accuracy])
        if step % 10 == 0:
          print("After %d steps, per token cost is %.3f, acc is %.3f" % (step, cost, acc))
          # print('loss', loss)
        # 每200步保存一个checkpoint。
        if step % 200 == 0:
          saver.save(session, CHECKPOINT_PATH, global_step=step)
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

    # 训练模型。
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      for i in range(self.FLAGS.num_epochs):
        print("In iteration: %d" % (i + 1))
        sess.run(iterator.initializer)
        step = self.run_epoch(sess, self.cost, self.train_op, saver, step)


if __name__ == '__main__':
  # Data loading params
  tf.app.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
  # tf.app.flags.DEFINE_string("train_file", "/home/lian/data/nlp/datagrand_info_extra/train_index.txt", "Train file source.")
  tf.app.flags.DEFINE_string("train_file", "/Users/lianxiaohua/Data/datagrand/train_index.txt", "Train file source.")
  # tf.app.flags.DEFINE_string("target_file", "/home/lian/data/nlp/datagrand_info_extra/target_index.txt", "Train file source.")
  tf.app.flags.DEFINE_string("target_file", "/Users/lianxiaohua/Data/datagrand/target_index.txt", "Train file source.")
  tf.app.flags.DEFINE_integer("num_tag", 4,
                              "Train file source.")

  # Model Hyperparameters
  tf.app.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
  tf.app.flags.DEFINE_integer("rnn_units", 128, "Number of filters per filter size (default: 128)")
  tf.app.flags.DEFINE_float("dropout_keep_prob", 0.1, "Dropout keep probability (default: 0.5)")
  tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
  tf.app.flags.DEFINE_float("lr", 0.001, "Learning rate")

  # Training parameters
  tf.app.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
  tf.app.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
  tf.app.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
  tf.app.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
  tf.app.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

  # Misc Parameters
  tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
  tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

  FLAGS = tf.app.flags.FLAGS

  print('Building model...')
  network = SequenceTagging(
    vocab_size=10600,
    sequence_length=800,
    embedding_size=200,
    rnn_units=128
  )

  init_w = load_embedding_vectors_word2vec('../../model/datagrand_corpus_pretrain.bin', FLAGS.embedding_dim)

  # Build and compile network
  network.run()
  # Add summaries to graph
  network.summary()

  # Save model
  # network.checkpoint(out_dir)
