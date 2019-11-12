# coding:utf8

"""
最大概率分词法（即1-gram），假设每个词都是独立的，根据整体序列分割后最大化似然来获取分割结果。
"""
import math

DELIMITER = " "


class DNASegment:
  def __init__(self):
    self.word_dict = {}  # 记录概率
    self.word_dict_count = {}  # 记录词频
    self.gmax_word_length = 20  # 最大词长
    self.all_freq = 0  # 所有词的词频总和

  def get_unknown_word_prob(self, word):
    return math.log(10. / self.all_freq * 10 ** len(word)) * 100

  def get_best_pre_node(self, sequence, node, node_state_list):
    """
    寻找node的最佳前驱节点
    方法为寻找所有可能的前驱片段
    Args:
      sequence:
      node:
      node_state_list:

    Returns:

    """
    # 求最大分割长度：即如果node值大于最大分词长度，则只向前回溯最大分词长度，
    # 否则sequence长度肯定不够最大分词长度，故回溯node值。
    max_seg_len = min(node, self.gmax_word_length)

    node_prob_list = {}
    for seg_idx in range(1, max_seg_len + 1):
      start = node - seg_idx
      tmp_wd = sequence[start: node]

      if self.word_dict.get(tmp_wd):
        node_prob_list[start] = self.word_dict[tmp_wd] + node_state_list[start]['prob_sum']
      else:
        node_prob_list[start] = self.get_unknown_word_prob(tmp_wd) + node_state_list[start]['prob_sum']
      # print('start:', start, 'tmp_wd:', tmp_wd, 'prob:', self.word_dict.get(tmp_wd),
      #       self.get_unknown_word_prob(tmp_wd), 'seg_idx:', seg_idx, 'node:', node,
      #       'start:', start, 'start val:', node_prob_list[start])

    sorted_dic = dict(sorted(node_prob_list.items(), key=lambda x: x[1], reverse=True))

    max_idx, max_prob = list(sorted_dic.items())[0]
    return max_idx, max_prob

  def mp_seg(self, sequence):
    """
    最大概率分词
    Args:
      sequence:

    Returns:

    """
    # TODO 1 初始化
    # TODO 2 逐个节点寻找最佳前驱节点，字符串概率为1元概率,#P(ab c) = P(ab)P(c)
    # TODO 3 获取最优路径，从后向前
    # TODO 4 构建切分
    sequence = sequence.strip()

    # step1 初始化
    node_state_list = []  # 记录节点的状态，该数组下标对应节点位置
    # 初始节点，也就是0节点信息
    ini_state = {}
    ini_state["pre_node"] = -1  # 前一个节点
    ini_state["prob_sum"] = 0  # 当前的概率总和
    node_state_list.append(ini_state)

    # step2 逐个节点寻找最佳前驱节点
    for node in range(1, len(sequence) + 1):
      best_pre_node, best_prob_sum = self.get_best_pre_node(sequence, node, node_state_list)

      cur_node = {}
      cur_node["pre_node"] = best_pre_node
      cur_node["prob_sum"] = best_prob_sum
      node_state_list.append(cur_node)
      # print('node_state_list', node_state_list)
      # print('-' * 40)

    # step3 获取最优路径，从后向前
    best_path = []
    node = len(sequence)  # 最后一个点
    best_path.append(node)

    while True:
      pre_node = node_state_list[node]["pre_node"]
      if pre_node == -1:
        break
      node = pre_node
      best_path.append(node)
    best_path.reverse()
    print('The best path is', best_path)

    # step 3, 构建切分
    word_list = []
    for i in range(len(best_path) - 1):
      left = best_path[i]
      right = best_path[i + 1]
      word = sequence[left:right]
      word_list.append(word)

    seg_sequence = DELIMITER.join(word_list)
    return seg_sequence

  def initial_dict(self, filename):
    """
    加载词典，为词\t词频的格式
    计算all_values
    计算频数
    计算频率
    Args:
      filename:

    Returns:

    """
    with open(filename, 'r') as fin:
      for line in fin.readlines():
        line = line.strip()
        k, v = line.split('\t')
        self.word_dict_count[k] = int(v)

      self.all_freq = sum(self.word_dict_count.values())

      for k in self.word_dict_count:
        v = self.word_dict_count[k]
        self.word_dict[k] = math.log(v / self.all_freq)
    print(self.word_dict)


if __name__ == '__main__':
  myseg = DNASegment()
  myseg.initial_dict("count_1w.txt")

  sequences = ['itisatest', 'tositdown', 'ihaveaterm', 'itoseefilm']

  for seq in sequences:
    seg_seq = myseg.mp_seg(seq)
    print("original sequence: " + seq)
    print("segment result: " + seg_seq)
