# coding:utf8

import fasttext
from abc import ABCMeta, abstractmethod
import sys
sys.path.append('../')


class ModelWrapper(metaclass=ABCMeta):
    """
    Model Abstract Class
    """
    def __init__(self):
        """
        Initialization method
        """
        self.model = None

    @abstractmethod
    def fit(self, data_fname, model_name):
        """

        :param data_fname:
        :param model_name:
        :return:
        """
        pass

    @abstractmethod
    def predict(self, word_text):
        """

        :param word_text:
        :return:
        """
        pass


class FastTextModel(ModelWrapper):
    """
    Text Classification Class
    """
    def __init__(self,
                 label_prefix="__label__",
                 lr=1e-1,
                 lr_update_rate=100,
                 dim=16,
                 ws=3,
                 epoch=100,
                 minn=0,
                 min_count=5,
                 t=1e-4,
                 encoding='utf8',
                 thread=4
                 ):
        super(FastTextModel, self).__init__()
        self._encoding = encoding
        self._t = t
        self._label_prefix = label_prefix
        self._lr = lr
        self._lr_update_rate = lr_update_rate
        self._dim = dim
        self._ws = ws
        self._epoch = epoch
        self._minn = minn
        self._min_count = min_count
        self._thread = thread
        self.model_name = None
        print('__init__ done')

    @property
    def lr_update_rate(self):
        return self._lr_update_rate

    @property
    def label_prefix(self):
        return self._label_prefix

    @property
    def lr(self):
        return self._lr

    @property
    def dim(self):
        return self._dim

    @property
    def ws(self):
        return self._ws

    @property
    def epoch(self):
        return self._epoch

    @property
    def minn(self):
        return self._minn

    @property
    def min_count(self):
        return self._min_count

    @property
    def thread(self):
        return self._thread

    @property
    def t(self):
        return self._t

    @property
    def encoding(self):
        return self._encoding

    @lr_update_rate.setter
    def lr_update_rate(self, value):
        self._lr_update_rate = value

    @t.setter
    def t(self, value):
        self._t = value

    @encoding.setter
    def encoding(self, value):
        self._encoding = value

    @label_prefix.setter
    def label_prefix(self, value):
        self._label_prefix = value

    @lr.setter
    def lr(self, value):
        self._lr = value

    @dim.setter
    def dim(self, value):
        self._dim = value

    @ws.setter
    def ws(self, value):
        self._ws = value

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    @minn.setter
    def minn(self, value):
        self._minn = value

    @min_count.setter
    def min_count(self, value):
        self._min_count = value

    @thread.setter
    def thread(self, value):
        self._thread = value

    def fit(self, data_fname, model_name):
        """

        :param model_name:
        :param data_fname:
        :return:
        """
        self.model_name = model_name

        classifier = fasttext.supervised(
            data_fname,
            self.model_name,
            lr=self.lr,
            lr_update_rate=self.lr_update_rate,
            dim=self.dim,
            ws=self.ws,
            epoch=self.epoch,
            minn=self.minn,
            min_count=self.min_count,
            t=self.t,
            encoding=self.encoding,
        )
        print('fit done')
        self.model = classifier

    def test(self, data_fname, label_num=1):
        """

        :param data_fname:
        :param label_num:
        :return:
        """
        result = self.model.test(
            data_fname,
            label_num
        )
        precision = result.precision
        recall = result.recall
        if not self.precision:
            self.precision = dict()
        if not self.recall:
            self.recall = dict()
        self.precision[label_num] = precision
        self.recall[label_num] = recall

        return precision, recall

    def load_model(self, model_name, label_prefix='__label__'):
        """

        :param model_name:
        :param label_prefix:
        :return:
        """
        try:
            self.model = fasttext.load_model(model_name, label_prefix=label_prefix)
            self.model_name = model_name
        except Exception as e:
            # 无该模型
            self.model = None

    def predict(self, data, recommend_kp_num=4):
        """

        :param data:
        :param recommend_kp_num:
        :return:
        """
        result = self.model.predict(data, recommend_kp_num)

        return result


if __name__ == '__main__':
    ftm = FastTextModel(thread=1)

    mode = sys.argv[1]
    if mode == 'train':
        # ftm.fit('/opt/kp_estimation/kpest_train/data/fasttext_data_1207/020_7_fasttext_train.txt',
        #         './fasttext_020_7_model.bin')
        # ftm.test('/opt/kp_estimation/kpest_train/data/fasttext_data_1207/020_7_fasttext_train.txt',
        #         1)

        ftm.fit(sys.argv[3], sys.argv[2])
        precision, recall = ftm.test(sys.argv[4], 1)

        print('percision: %s' % precision)
        print('recall: %s' % recall)
    elif mode == 'predict':
        ftm.load_model(sys.argv[2])
        result = ftm.predict(sys.argv[3])

        print('test result: \n', result)
