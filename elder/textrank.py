#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals
import sys
from operator import itemgetter
from collections import defaultdict
import jieba.posseg
from jieba.analyse.tfidf import KeywordExtractor
from hqyx_dbs import _compat


# 用于表示文本的无向带权图
class UndirectWeightedGraph:
    d = 0.85  # 衰减系数

    def __init__(self):
        # 用一条边的起始结点作为关键字的字典来表示整个图的信息
        self.graph = defaultdict(list)

    def addEdge(self, start, end, weight):
        # use a tuple (start, end, weight) instead of a Edge object
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self):
        ws = defaultdict(float)  # 表示每一个结点的权重
        outSum = defaultdict(float)  # 结点所关联边的权值之和

        wsdef = 1.0 / (len(self.graph) or 1.0)  # 为每个结点初始化一个权值
        for n, out in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((e[2] for e in out), 0.0)

        # this line for build stable iteration
        sorted_keys = sorted(self.graph.keys())
        for x in xrange(10):  # 10 iters,textrank值的计算只进行10次循环
            # 进行textrank值的更新
            for n in sorted_keys:
                s = 0
                for e in self.graph[n]:
                    s += e[2] / outSum[e[1]] * ws[e[1]]
                ws[n] = (1 - self.d) + self.d * s

        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])  # 系统设定的最大和最小值

        for w in _compat.itervalues(ws):  # 返回值的迭代器
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        # 将权值进行归一化
        for n, w in ws.items():
            # to unify the weights, don't *100.
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return ws


class TextRank(KeywordExtractor):
    def __init__(self):
        self.tokenizer = self.postokenizer = jieba.posseg.dt  # 对输入文本进行分词的方法
        self.stop_words = self.STOP_WORDS.copy()  # 停用词列表
        self.pos_filt = frozenset(('ns', 'n', 'vn', 'v'))  # 词性的筛选集
        self.span = 5  # 表示词与词之间有联系的窗口的大小为5

    def pairfilter(self, wp):
        # 确保该词的词性是我们所需要的,且这个词包含两个字以上,且都是小写,并且没有出现在停用词列表中
        return (wp.flag in self.pos_filt and len(wp.word.strip()) >= 2
                and wp.word.lower() not in self.stop_words)

    def textrank(self, sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False):
        """
        Extract keywords from sentence using TextRank algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v'].
                        if the POS of w is not in this list, it will be filtered.
            - withFlag: if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        self.pos_filt = frozenset(allowPOS)
        g = UndirectWeightedGraph()
        cm = defaultdict(int)
        words = tuple(self.tokenizer.cut(sentence))  # 首先必须要对文本进行分词
        for i, wp in enumerate(words):  # i和wp分别代表words的下标与其对应的元素,元素内容包括词以及其对应的词性
            if self.pairfilter(wp):
                for j in xrange(i + 1, i + self.span):
                    if j >= len(words):
                        break
                    if not self.pairfilter(words[j]):
                        continue
                    if allowPOS and withFlag:
                        cm[(wp, words[j])] += 1
                    else:
                        cm[(wp.word, words[j].word)] += 1  # 如果两个词出现在同一个窗口内,则在两者间加上一条边

        for terms, w in cm.items():
            g.addEdge(terms[0], terms[1], w)  # 将共同出现的次数作为边的权值,表示边的关键字用词而非词的编号
        nodes_rank = g.rank()
        if withWeight:
            tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)

        if topK:
            return tags[:topK]
        else:
            return tags

    xtract_tags = textrank
