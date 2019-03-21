# -*- coding: UTF-8 -*-

import jieba
import sys
import os

reload(sys)

sys.setdefaultencoding('utf-8')  # 设置默认编码方式为utf-8


def set_thesaurus_lib(source_file, target_file):
    if os.path.exists(source_file) and os.path.exists(target_file):
        file = open(source_file, "r")
        output = open(target_file, "a+")
        while True:
            line = file.readline().replace("\n", "")
            if line:
                line_list = line.split(" ")[1:]
                output.write(" ".join(line_list) + "\n")
            else:
                print "Read File End."
                break
        file.close()
        output.close()
        print "Write Done!"


def get_thesaurus_lib(source_words):
    print "Source words is: "
    print " ".join(source_words)
    thesaurus_dict = {}
    thesaurus_file = "D:\\word_split\\thesaurus.txt"
    if os.path.exists(thesaurus_file):
        file = open(thesaurus_file)
        while True:
            line = file.readline().decode("utf8").replace("\n", "")
            if not line:
                break
            words = line.split(" ")
            for i in range(1, len(words)):
                thesaurus_dict[words[i].encode("utf8")] = words[0].encode("utf8")

    source_words_tmp = [thesaurus_dict.get(x.encode("utf8")) if thesaurus_dict.get(x.encode("utf8")) else x for x in
                        source_words]
    print " ".join(source_words_tmp)

