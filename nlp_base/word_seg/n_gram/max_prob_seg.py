# coding:utf8

import sys
import math


class DNASegment:
  def __init__(self):
    self.word_dict = {}  # 记录概率
    self.word_dict_count = {}  # 记录词频
    self.gmax_word_length = 0  # 最大词长
    self.all_freq = 0  # 所有词的词频总和

  def get_unknown_word_prob(self, word):
    return math.log(10. / self.all_freq * 10 ** len(word))

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
    # step2
    for node in range(1, len(sequence) + 1):
      best_pre_node, best_prob_sum = self.get_best_pre_node(sequence, node, node_state_list)

      cur_node = {}
      cur_node["pre_node"] = best_pre_node
      cur_node["prob_sum"] = best_prob_sum
      node_state_list.append(cur_node)

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

        self.word_dict_count[k] = v

      self.all_freq = sum(self.word_dict_count.values())

      for k in self.word_dict_count:
        v = self.word_dict_count[k]
        self.word_dict[k] = math.log(v / self.all_freq)
