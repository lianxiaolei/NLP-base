# coding:utf8

import datetime
import time
from tensorflow.python.framework import graph_util
from tensorflow.contrib import rnn
from nlp_base.tools.data_helper import *
from nlp_base.tools import get_tensorflow_conf


# CHECKPOINT_PATH = "/home/lian/PycharmProjects/seq2seq/checkpoint/seq2seq_ckpt"  # checkpoint保存路径。

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
      name='embedding', trainable=True)

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
    self.sess.run(tf.assign(self.embedding, value, name='init_wemb'))

  def _build_lookup(self, src):
    """
    Get lookup embedding tensor from input ids.
    Args:
      src: A Tensor contains words' ids.

    """
    with tf.name_scope('lookup'):
      self.outputs = tf.nn.embedding_lookup(self.embedding, src)

  def _build_rnn(self):
    """
    Build a BiLSTM to capture the sequence information.

    """
    gru_1 = tf.keras.layers.GRU(self.rnn_units, return_sequences=True,
                                kernel_initializer='he_normal',
                                name='gru1')(self.outputs)
    gru_1b = tf.keras.layers.GRU(self.rnn_units, return_sequences=True, go_backwards=True,
                                 kernel_initializer='he_normal',
                                 name='gru1_b')(self.outputs)
    gru1_merged = tf.keras.layers.add([gru_1, gru_1b])  # [batch, height, units]

    # gru_2 = tf.keras.layers.GRU(self.rnn_units, return_sequences=True,
    #                             kernel_initializer='he_normal',
    #                             name='gru2')(self.outputs)
    # gru_2b = tf.keras.layers.GRU(self.rnn_units, return_sequences=True, go_backwards=True,
    #                              kernel_initializer='he_normal',
    #                              name='gru2_b')(self.outputs)
    #
    # outputs = tf.keras.layers.concatenate([gru_2, gru_2b])  # [batch, height, units * 2]

    self.outputs = gru1_merged

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

    with self.sess.as_default():
      # embedding
      self._build_lookup(src)

      # RNN
      self._build_rnn()

      # Tail to logits
      self.outputs = tf.keras.layers.Dense(self.rnn_units, activation='relu')(self.outputs)
      self.score = tf.keras.layers.Dense(self.num_tag, activation='relu', use_bias=True)(self.outputs)
      self.logits = tf.nn.softmax(self.score, name='logits')

      self.argmax = tf.argmax(self.logits, axis=-1)

      self.target = target

      # Define the sequence mask.
      self.seq_mask = tf.sequence_mask(self.src_size, dtype=tf.float32)

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
    # Defind the optimizer
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    # self.learning_rate = tf.train.exponential_decay(self.FLAGS.lr, self.global_step,
    #                                                 decay_steps=256, decay_rate=0.86)
    self.learning_rate = self.FLAGS.lr
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.grads_and_vars = self.optimizer.compute_gradients(self.cost)
    self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, self.global_step)

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
    loss_summary = tf.summary.scalar('loss', self.loss)
    # acc_summary = tf.summary.scalar('accuracy', self.accuracy)

    # Train summaries
    # self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
    self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

    # Dev summaries
    # self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)

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
    """
    Load serialized model file.
    Args:
      pb_path: A string, model file path.

    """
    with tf.Session(graph=tf.Graph()) as sess:
      tf.saved_model.loader.load(sess, ['cpu_1'], pb_path + 'savemodel')
      sess.run(tf.global_variables_initializer())

      x = sess.graph.get_tensor_by_name('x:0')
      y = sess.graph.get_tensor_by_name('y:0')

  def current_step(self):
    """
    Get current training steps.
    Returns:

    """
    return tf.train.global_step(self.sess, self.global_step)

  def train_step(self, saver, step):
    while True:
      try:
        cost, _ = self.sess.run([self.cost, self.train_op])
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
    print('src shape', src.shape)
    # 定义前向计算图。输入数据以张量形式提供给forward函数。
    self.build_net(src, src_size, trg_label)
    self.label = trg_label
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
    with self.sess.as_default() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(iterator.initializer)
      src_arr, src_size_arr, trg_label_arr = sess.run([src, src_size, trg_label])
    print('src_arr', src_arr)
    print('src_size_arr', src_size_arr)
    print('trg_label_arr', trg_label_arr)


if __name__ == '__main__':
  FLAGS = get_tensorflow_conf(tf)

  print('Building model...')
  network = SequenceTagging(
    vocab_size=21350,
    sequence_length=200,
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
