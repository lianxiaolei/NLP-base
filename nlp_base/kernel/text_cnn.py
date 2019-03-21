# coding:utf8

import tensorflow as tf
import time
import os
import datetime
from nlp_base.tools.data_helpers import *


class TextCNN(object):
    def __init__(self, sequence_length, num_classes,
                 vocab_size, embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda

    def init_wemb(self, value):
        self.sess.run(tf.assign(self.w_emb, value, name='init_wemb'))

    def _build_embedding(self):
        print('Building network')
        with tf.name_scope('embedding'):
            w_emb = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1., 1.),
                name='w_emb')
            self.w_emb = w_emb
            # 获取下标为x的w_emb中的内容
            self.embedded_chars = tf.nn.embedding_lookup(w_emb, self.x)

            self.embedding_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv_%s' % filter_size):
                # filter_size: 一次窗口包含的词数
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]

                w = tf.Variable(
                    tf.truncated_normal(filter_shape, stddev=0.1),
                    name='w')

                b = tf.Variable(
                    tf.constant(0.1, shape=[self.num_filters]), name='b'
                )
                conv = tf.nn.conv2d(self.embedding_chars_expanded,
                                    w,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                pool = tf.nn.max_pool(
                    # 第二维是词窗口大小
                    h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')

                pooled_outputs.append(pool)

        num_filters_flatten = self.num_filters * len(self.filter_sizes)

        self.h_pool = tf.concat(pooled_outputs, 3, )
        # flatten
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_flatten])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob, name='dropout')

        # Dense + Softmax
        with tf.name_scope('output'):
            w = tf.Variable(
                tf.truncated_normal(shape=[num_filters_flatten, self.num_classes]),
                name='w')
            b = tf.Variable(
                tf.constant(value=0.1, shape=(self.num_classes,), name='b'))
            print(self.h_drop.shape, w.shape)
            self.score = tf.nn.xw_plus_b(self.h_drop, w, b, name='dense')
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

    def architecture(self):
        FLAGS = tf.flags.FLAGS
        self.FLAGS = FLAGS

        with tf.Graph().as_default():
            with tf.name_scope(name='ph'):
                self.x = tf.placeholder(tf.int32, [None, self.sequence_length], name='x')
                self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='y')

                self.keep_prob = tf.placeholder(tf.float32, name='kp')
                self.learning_rate = tf.placeholder(tf.float32,
                                                    name='learning_rate')
            self.l2_loss = tf.constant(0.0)

            config = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,  # 设置让程序自动选择设备运行
                log_device_placement=FLAGS.log_device_placement)

            self.sess = tf.Session(config=config)

            with self.sess.as_default():
                self._build_embedding()

                self.learning_rate = 1e-3

                # Defind the optimizer
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(1e-3)
                self.grads_and_vars = self.optimizer.compute_gradients(network.loss)
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, self.global_step)

                self.sess.run(tf.global_variables_initializer())

    def summary(self):
        # Summary
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'run', timestamp))
        print('Writing to {}\n'.format(out_dir))

        # Summary for loss and accuracy
        loss_summary = tf.summary.scalar('loss', network.loss)
        acc_summary = tf.summary.scalar('accuracy', network.accuracy)
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
        print("{}:step{},loss{:g},acc{:g}".format(time_str, step, loss, accuracy))

        self.train_summary_writer.add_summary(summaries, step)
        if step % FLAGS.batch_size == 0:
            print('epoch:{}'.format(step // FLAGS.batch_size))

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
        print("{}:step{},loss{:g},acc{:g}".format(time_str, step, loss, accuracy))

        self.dev_summary_writer.add_summary(summaries, step)
        # if step % FLAGS.batch_size == 0:
        #     print('epoch:{}'.format(step // FLAGS.batch_size))

    def current_step(self):
        return tf.train.global_step(self.sess, self.global_step)

    def save_checkpoint(self, step):
        return self.saver.save(self.sess, self.checkpoint_prefix, global_step=step)

    def run(self, x_train, y_train, x_dev, y_dev):
        for epoch in range(100):
            # Generate batches
            batches = batch_iter(
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
    #tf.app.flags.DEFINE_string("train_file", "../../dataset/020_7_shuffle_train.csv", "Train file source.")
    tf.app.flags.DEFINE_string("train_file", "../../dataset/cnews.train.txt", "Train file source.")
    #tf.app.flags.DEFINE_string("test_file", "../../dataset/020_7_shuffle_test.csv", "Test file source.")
    tf.app.flags.DEFINE_string("test_file", "../../dataset/cnews.test.txt", "Test file source.")

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
    # print("\nParameters:")
    # for attr, value in sorted(FLAGS.__flags.items()):
    #     print("{}={}".format(attr.upper(), value))
    # print("")

    # with open('config.yml', 'r') as yml:
    #     cfg = yaml.load(yml)

    print('Loading data...')
    x_train, y_train, x_test, y_test, vocab_processor = \
        load_train_dev_data(FLAGS.train_file, FLAGS.test_file)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # 大家伙，不能输出
    # print('vocabulary:', vocab_processor.vocabulary_._mapping)

    network = TextCNN(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=200,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
        num_filters=FLAGS.num_filters
    )

    out_dir = '../model'

    vocab_processor.save(os.path.join(out_dir, 'vocab'))
    vocabulary = vocab_processor.vocabulary_

    init_w = load_embedding_vectors_word2vec(vocabulary,
                                             '../model_w2v/cnews.bin')

    # Build and compile network
    network.architecture()
    # Initialize embedding layer weights
    network.init_wemb(init_w)
    # Add summaries to graph
    network.summary()
    # Train
    network.run(x_train, y_train, x_test, y_test)
    # Save model
    # network.checkpoint(out_dir)
