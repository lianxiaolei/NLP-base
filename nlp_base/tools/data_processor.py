# coding:utf8

import pandas as pd
import csv as csv
import sys
sys.path.append('../')
import os
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split


class DataWrapper(metaclass=ABCMeta):
    def __init__(self, inpath, outpath):
        self._outpath = outpath
        self._inpath = inpath

    @property
    def inpath(self):
        return self._inpath

    @property
    def outpath(self):
        return self._outpath

    @inpath.setter
    def inpath(self, inpath):
        self._inpath = inpath

    @outpath.setter
    def outpath(self, outpath):
        self._outpath = outpath

    @abstractmethod
    def run(self, names, delimiter='\t'):
        raise AssertionError('The subclass\' method "call" should be defined.')


class DataFilter(DataWrapper):
    def __init__(self, inpath, outpath):
        super(DataFilter, self).__init__(inpath, outpath)
        self.df = None

    def run(self, names, delimiter='\t'):
        df = pd.read_csv(self.inpath, delimiter=delimiter, names=names, header=None)
        self.df = df
        self.delimiter = delimiter

        self.filter([r'\[ / LaTeXI ]', r'\[', r'LaTeXI', r'\]', r'\{', r'\}',
                     r'& gt ;', r'\\', r'frac', r'matrix', '\n'])

        self.write()

    def write(self):
        csv.register_dialect('mydialect', delimiter=self.delimiter,
                             quoting=csv.QUOTE_NONE, escapechar=' ')

        csvfile = open(self.outpath, 'w', encoding='utf8')
        writer = csv.writer(csvfile, dialect='mydialect')
        writer.writerows(self.df.values)
        csvfile.close()

    def groupby(self, keys, values, group_keys=False):
        """

        :param keys:
        :param values:
        :param group_keys:
        :return:
        """
        return self.df[values].groupby(keys, group_keys=group_keys).count()

    def filter(self, rubbish, drop_nan=True):
        """

        :param drop_nan:
        :param rubbish:
        :return:
        """
        for rub in rubbish:
            self.df.replace(rub, '', inplace=True, regex=True)

        if drop_nan:
            self.df.replace('\\N', np.nan, inplace=True)
            self.df.dropna()


class DataSpliter(DataWrapper):
    def __init__(self, inpath, outpath):
        super(DataSpliter, self).__init__(inpath, outpath)
        self.df = None

    def run(self, names, delimiter='\t', test_size=0.15, random_state=0):
        df = pd.read_csv(self.inpath, delimiter=delimiter, names=names, header=None)
        if 'que_id' in names:
            df = df[[name for name in names if not name == 'que_id']]
        self.df = df
        self.delimiter = delimiter

        train, test = self.split(test_size=test_size, random_state=random_state)

        csv.register_dialect('mydialect',
                             delimiter='\t',
                             quoting=csv.QUOTE_NONE, escapechar=' ')

        self.write(train, 'train')
        self.write(test, 'test')

    def write(self, value, mode='train'):
        csv.register_dialect('mydialect', delimiter=self.delimiter,
                             quoting=csv.QUOTE_NONE, escapechar=' ')

        outpath = self.outpath[:self.outpath.rfind('.')] \
                  + '_%s' % mode \
                  + self.outpath[self.outpath.rfind('.'):]
        csvfile = open(outpath, 'w', encoding='utf8')
        writer = csv.writer(csvfile, dialect='mydialect')
        writer.writerows(value)
        csvfile.close()

    def split(self, test_size=0.15, random_state=0):
        """

        :param test_size:
        :param random_state:
        :return:
        """
        train, test = train_test_split(
            self.df.values, test_size=test_size, random_state=random_state)

        return train, test


class DataAggregator(DataWrapper):
    def __init__(self, inpath, outpath):
        super(DataAggregator, self).__init__(inpath, outpath)
        self.df = None

    def run(self, names, delimiter='\t', nums=5, subject='020', grade='7'):
        df = pd.read_csv(self.inpath,
                         delimiter=delimiter,
                         names=names,
                         header=None,
                         dtype='object')

        df.replace(r'\__label__', '', inplace=True, regex=True)

        gd = df.groupby('content').count()

        # 科目年级总题数
        nm = set(df['content'].values.tolist())
        self.data_len = len(nm)

        data = df.values

        self.write(data, gd, nm, nums=nums, subject='020', grade='7')

    def write(self, value, groupby, cont_set, nums, subject='020', grade='7'):
        f = list()
        for i in range(1, nums + 1):
            f.append(
                open(
                    os.path.join(self.outpath, '%s_%s_%s.txt' % (subject, grade, i)),
                    'w', encoding='utf8'))

        for n in cont_set:
            if groupby.loc[n]['kp_id'] > nums:
                continue

            st = '%s\t%s\t%s\n' % (n, groupby.loc[n]['kp_id'], ','.join(value[value[:, 0] == n][:, 1]))
            f[groupby.loc[n]['kp_id'] - 1].write(st)


if __name__ == '__main__':
    content = sys.argv[1]
    flt = sys.argv[2]
    split = sys.argv[3]
    # df = DataFilter('../../dataset/020_7_content.csv', '../../dataset/020_7_filtered.csv')
    # df = DataFilter(content, flt)
    # df.run(names=["que_id", "content", "kp_id"])

    # ds = DataSpliter('../../dataset/020_7_filtered.csv', '../../dataset/020_7_shuffle.csv')
    ds = DataSpliter(flt, split)
    ds.run(names=["que_id", "content", "kp_id"])
