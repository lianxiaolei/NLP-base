# coding:utf8

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tqdm import tqdm
from info_extra.metrics import gen_metrics
from info_extra.model import SequenceLabelling

tf.app.flags.DEFINE_integer("num_classes", 5, "Number of class")
tf.app.flags.DEFINE_integer("vocab_length", 4550, "Number of vocabular size")
tf.app.flags.DEFINE_integer("word_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.app.flags.DEFINE_integer('rnn_units', 128, "Number of rnn output units")
tf.app.flags.DEFINE_integer('rnn_layer_num', 1, "Number of rnn layer")
tf.app.flags.DEFINE_integer('max_seq_len', 128, "Number of max sequence length")
tf.app.flags.DEFINE_float("keep_prob", 0.8, "Dropout keep probability (default: 0.8)")
tf.app.flags.DEFINE_float("lr", 0.01, "Learning rate")
tf.app.flags.DEFINE_string("checkpoint",
                           '/home/lian/PycharmProjects/NLP-base/model/seqtag_checkpoint/seqtag_ckpt',
                           "Model checkpoint")


class ModelAssist(object):
  def __init__(self):
    self.FLAGS = tf.flags.FLAGS
    self.model = SequenceLabelling(self.FLAGS)
    self.model.build_net(compile=True)
    # Defind model saver.
    self.saver = tf.train.Saver()

  def restore(self):
    saver = tf.train.Saver()
    with self.model.sess.as_default() as sess:
      saver.restore(sess, tf.train.latest_checkpoint(self.FLAGS.checkpoint[:self.FLAGS.checkpoint.rfind('/')]))
    print('Load model successfully.')

  def train(self, iterator, train_initializer, dev_initializer=None,
            epoch_num=100, step_num=1000, save_when_acc=None, save_when_loss=None, dev_step=10):
    X_ten, y_ten, seq_len_ten = iterator.get_next()
    with self.model.sess.as_default() as sess:
      sess.run(tf.initialize_all_variables())
      for epoch in range(epoch_num):
        sess.run(train_initializer)
        bar = tqdm(range(step_num), ncols=100)
        for step in bar:
          try:
            X, y, seq_len = sess.run([X_ten, y_ten, seq_len_ten])
            # print(X[0, 0], y[0, : 50])
            loss, logits, _ = sess.run([self.model.loss, self.model.logits, self.model.train_op],
                                       feed_dict={self.model.X: X,
                                                  self.model.y: y,
                                                  self.model.seq_len: seq_len})
          except tf.errors.OutOfRangeError:
            sess.run(train_initializer)

          auc, acc = gen_metrics(seq_len, logits, y, 128, 100)
          bar.set_description_str("Step:{}\t  Loss:{}\t  Acc:{}".format(step, str(loss)[:5], str(acc)[:5]))

        # Dev
        sess.run(dev_initializer)
        bar = tqdm(range(dev_step), ncols=100)
        for step in bar:
          try:
            X, y, seq_len = sess.run([X_ten, y_ten, seq_len_ten])
            logits = sess.run(self.model.logits,
                              feed_dict={self.model.X: X,
                                         self.model.y: y,
                                         self.model.seq_len: seq_len}
                              )
            acc = gen_metrics(seq_len, logits, y, 128, 100)
          except tf.errors.OutOfRangeError:
            sess.run(dev_initializer)
          bar.set_description_str("Step:{}\tAcc:{}".format(step, acc))

        if save_when_loss:
          if loss <= save_when_loss:
            self.saver.save(sess, self.FLAGS.check_point, global_step=epoch)
            print('Saved model done.')
        if save_when_acc:
          if acc >= save_when_acc:
            self.saver.save(sess, self.FLAGS.check_point, global_step=epoch)
            print('Saved model done.')

  def inference(self, iterator, test_initializer, word_set):
    with self.model.sess.as_default() as sess:
      sess.run(tf.initialize_all_variables())
      sess.run(test_initializer)

      X_ten, y_ten, seq_len_ten = iterator.get_next()

      for step in range(1):
        X, y, seq_len = sess.run([X_ten, seq_len_ten])
        y_predict_results, acc = sess.run([self.model.pred, self.model.accuracy])
        y_predict_results = np.reshape(y_predict_results, X.shape)
        for i in range(len(X)):
          x_result, y_predict_result = list(filter(lambda x: x, X[i])), list(
            filter(lambda x: x, y_predict_results[i]))
          # x_text, y_predict_text = ''.join(id2word[x_result].values), ''.join(id2tag[y_predict_result].values)
          x_text, y_predict_text = x_result, y_predict_result
          print([word_set[idx] for idx in x_text], '\n', y_predict_text)
          print(x_text, '\n', y_predict_text)
          print('-' * 80)


if __name__ == '__main__':
  model = ModelAssist()
