import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.contrib import rnn
from tqdm import tqdm


if __name__ == '__main__':
  train_initializer, dev_initializer, test_initializer, iterator = data_iterate(sentences, tags, 10)

  x, y = iterator.get_next()

  embedding = tf.Variable(tf.random_normal([4550, 20]), dtype=tf.float32)
  inputs = tf.nn.embedding_lookup(embedding, x + 1)

  cell_fw = [lstm_cell(128, 1) for _ in range(1)]
  cell_bw = [lstm_cell(128, 1) for _ in range(1)]
  inputs = tf.unstack(inputs, 64, axis=1)
  output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw,
                                                        inputs=inputs, dtype=tf.float32)

  output = tf.stack(output, axis=1)
  print(output.shape)
  output = tf.reshape(output, [-1, 128 * 2])

  logits = tf.keras.layers.Dense(5)(output)
  y_predict = tf.cast(tf.argmax(logits, axis=1), tf.int32)
  print('Output Y', y_predict)

  # Reshape y_label
  y_label_reshape = tf.cast(tf.reshape(y, [-1]), tf.int32)
  # Prediction
  correct_prediction = tf.equal(y_predict, y_label_reshape)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # Loss
  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y_label_reshape, logits=tf.cast(logits, tf.float32)))

  # Train
  train = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

  saver = tf.train.Saver()

  acc_thres = 0.94
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(100):
      # tf.train.global_step(sess, global_step_tensor=global_step)

      # Train
      sess.run(train_initializer)
      bar = tqdm(range(200), ncols=100)
      for step in bar:
        loss, acc, _ = sess.run([cross_entropy, accuracy, train])

        bar.set_description_str("Step:{}\t  Loss:{}\t  Acc:{}".format(step, str(loss)[:5], str(acc)[:5]))
        # if step % 10 == 0:
        #   print('Step', step, 'Train Loss', loss, 'Accuracy', acc)

      # Dev
      sess.run(dev_initializer)
      bar = tqdm(range(10), ncols=10)
      for step in bar:
        acc = sess.run(accuracy)

        bar.set_description_str("Step:{}\tAcc:{}".format(step, acc))
        # if step % 20 == 0:
        #   print('Dev Accuracy', acc)

      if acc > acc_thres:
        saver.save(sess, '/home/lian/PycharmProjects/NLP-base/model/checkpoint/seqtag_ckpt', global_step=step)
        print('Saved model done.')