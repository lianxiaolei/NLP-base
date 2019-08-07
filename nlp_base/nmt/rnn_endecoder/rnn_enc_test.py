import tensorflow as tf
import codecs
import sys

# 读取checkpoint的路径。9000表示是训练程序在第9000步保存的checkpoint。
CHECKPOINT_PATH = "/home/lian/PycharmProjects/seq2seq/checkpoint/seq2seq_ckpt-9000"

# 模型参数。必须与训练时的模型参数保持一致。
HIDDEN_SIZE = 1024  # LSTM的隐藏层规模。
NUM_LAYERS = 2  # 深层循环神经网络中LSTM结构的层数。
SRC_VOCAB_SIZE = 10000  # 源语言词汇表大小。
TRG_VOCAB_SIZE = 4000  # 目标语言词汇表大小。
SHARE_EMB_AND_SOFTMAX = True  # 在Softmax层和词向量层之间共享参数。

# 词汇表文件
SRC_VOCAB = "/home/lian/PycharmProjects/seq2seq/dataset/en.vocab"
TRG_VOCAB = "/home/lian/PycharmProjects/seq2seq/dataset/zh.vocab"

# 词汇表中<sos>和<eos>的ID。在解码过程中需要用<sos>作为第一步的输入，并将检查
# 是否是<eos>，因此需要知道这两个符号的ID。
SOS_ID = 1
EOS_ID = 2

MAX_DEC_LEN = 100


class NMTModel(object):
  def __init__(self):
    # 定义编码器和解码器所使用的LSTM结构。
    self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
      [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
       for _ in range(NUM_LAYERS)])
    self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
      [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
       for _ in range(NUM_LAYERS)])

    # 为源语言和目标语言分别定义词向量。
    self.src_embedding = tf.get_variable(
      "src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
    self.trg_embedding = tf.get_variable(
      "trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

    # 定义softmax层的变量
    if SHARE_EMB_AND_SOFTMAX:
      self.softmax_weight = tf.transpose(self.trg_embedding)
    else:
      self.softmax_weight = tf.get_variable(
        "weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
    self.softmax_bias = tf.get_variable(
      "softmax_bias", [TRG_VOCAB_SIZE])

  def inference(self, src_input):
    """

    Args:
      src_input:

    Returns:

    """
    src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
    src_input = tf.convert_to_tensor([src_input])
    src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

    with tf.variable_scope("nmt_model/encoder"):
      enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb, src_size, dtype=tf.float32)

    with tf.variable_scope("nmt_model/decoder/rnn/multi_rnn_cell"):
      # Initialize a dynamic tensorArray to cache the generating sentence.
      init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)

      # Initialize sentence with a start signal.
      init_array = init_array.write(0, SOS_ID)

      # Construct the loop status, contains encoder hidden state, tensorArray, step number.
      init_loop_var = (enc_state, init_array, 0)

      def continue_loop_condition(state, trg_ids, step):
        """
        condition function
        Args:
          state: The last step hidden state of encoder RNN.
          trg_ids: A TensorArray var, init array which used by sentence saving.
          step: Time step.

        Returns:

        """
        # reduce_all is a logical operation.
        return tf.reduce_all(
          tf.logical_and(tf.not_equal(trg_ids.read(step), EOS_ID),
                         tf.less(step, MAX_DEC_LEN - 1)))

      def loop_body(state, trg_ids, step):
        """

        Args:
          state: The last step hidden state of encoder RNN.
          trg_ids: A TensorArray var, init array which used by sentence saving.
          step: Time step.

        Returns:

        """
        trg_input = [trg_ids.read(step)]
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # Don't use dynamic_rnn to forward, instead by dec_cell.call to run a time step.
        dec_outputs, next_state = self.dec_cell.call(inputs=trg_emb, state=state)

        # Calculate the most probably word's logit,
        #   select the max of word's logit as output of this step.
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias

        # Get the maximum prob as the current step's output.
        next_id = tf.argmax(logits, axis=1, output_type=tf.int32)

        # Write the current word to dynamic TensorArray var.
        trg_ids = trg_ids.write(step + 1, next_id[0])

        return next_state, trg_ids, step + 1

      # Run tf.while_loop
      state, trg_ids, step = tf.while_loop(continue_loop_condition, loop_body, init_loop_var)

      return trg_ids.stack()


def main():
  # 定义训练用的循环神经网络模型。
  with tf.variable_scope("nmt_model", reuse=None):
    model = NMTModel()

  # 定义个测试句子。
  test_en_text = "I use my hands to make your dreams . <eos>"
  print(test_en_text)

  # 根据英文词汇表，将测试句子转为单词ID。
  with codecs.open(SRC_VOCAB, "r", "utf-8") as f_vocab:
    src_vocab = [w.strip() for w in f_vocab.readlines()]
    src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
  test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                 for token in test_en_text.split()]
  print(test_en_ids)

  output_op = model.inference(test_en_ids)
  sess = tf.Session()

  from tensorflow.python import pywrap_tensorflow
  reader = pywrap_tensorflow.NewCheckpointReader(CHECKPOINT_PATH)
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in var_to_shape_map:
    print('tensor name', key)
    # print(reader.get_tensor(key))

  saver = tf.train.Saver()
  saver.restore(sess, CHECKPOINT_PATH)

  # Get the translation result.
  output_ids = sess.run(output_op)
  print(output_ids)

  with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
    trg_vocab = [w.strip() for w in f_vocab.readlines()]
  output_text = ''.join([trg_vocab[x] for x in output_ids])

  print(output_text.encode('utf8').decode(sys.stdout.encoding))
  sess.close()


if __name__ == '__main__':
  main()
