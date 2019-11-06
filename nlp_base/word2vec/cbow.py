# coding:utf8

import numpy as np
import random
import math
import tensorflow as tf
from word2vec.data_helper import *

batch_size = 128
word_dim = 128  # Dimension of the embedding vector.
cbow_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
# pick 16 samples from 100
valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size // 2))
num_sampled = 64  # Number of negative examples to sample.
vocabulary_size = 50000

num_steps = 100001


class CBOW(object):
  """
  CBOW model
  """

  def __init__(self):
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)

    with self.graph.as_default():
      with tf.variable_scope('initialized_vars'):
        self.input = tf.placeholder(tf.int32, [None, cbow_window * 2])
        self.y = tf.placeholder(tf.int32, [None, 1])

        self.valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        self.embs = tf.get_variable('embs', shape=[vocabulary_size, word_dim],
                                    dtype=tf.float32, initializer=tf.random_uniform_initializer)

        self.nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, word_dim],
                                                           stddev=1.0 / math.sqrt(word_dim)))
        self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  def build_net(self):
    with self.graph.as_default():
      with tf.name_scope('operations'):
        lookup = tf.nn.embedding_lookup(self.embs, self.input)

        means = tf.reduce_mean(lookup, axis=1, keepdims=False)
        print('emb emb shape', self.input.get_shape().as_list())
        print('avg emb shape', means.get_shape().as_list())

        self.loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases, self.y, means,
                                                  num_sampled, vocabulary_size))

        self.train_op = tf.train.AdagradOptimizer(1.0).minimize(loss=self.loss)

        norm = tf.sqrt(tf.reduce_mean(self.embs, 1, keepdims=True))

        self.normed_embs = self.embs / norm

        valid_embs = tf.nn.embedding_lookup(self.normed_embs, self.valid_dataset)

        self.sim = tf.matmul(valid_embs, self.normed_embs, transpose_b=True)

  def train(self):
    with self.graph.as_default():
      self.build_net()
      with self.sess.as_default() as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        average_loss = 0
        for step in range(num_steps):
          batch_data, batch_labels = generate_batch(batch_size, cbow_window)
          feed_dict = {self.input: batch_data, self.y: batch_labels}
          _, l = session.run([self.train_op, self.loss], feed_dict=feed_dict)
          average_loss += l
          if step % 2000 == 0:
            if step > 0:
              average_loss = average_loss / 2000
              # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
          # note that this is expensive (~20% slowdown if computed every 500 steps)
          if step % 10000 == 0:
            sim = self.sim.eval()
            for i in range(valid_size):
              valid_word = reverse_dictionary[valid_examples[i]]
              top_k = 8  # number of nearest neighbors
              nearest = (-sim[i, :]).argsort()[1:top_k + 1]
              log = 'Nearest to %s:' % valid_word
              for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log = '%s %s,' % (log, close_word)
              print(log)
        final_embeddings = self.normed_embs.eval()

      try:
        # pylint: disable=g-import-not-at-top
        from sklearn.manifold import TSNE
        import matplotlib

        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        # 因为我们的embedding的大小为128维，没有办法直接可视化
        # 所以我们用t-SNE方法进行降维
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        # 只画出500个词的位置
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels)

      except ImportError:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')


if __name__ == '__main__':


  cbow = CBOW()
  cbow.train()
