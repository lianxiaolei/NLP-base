# encoding=utf-8

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


def clean_data_total(source_path, target_path):
    # grand_path = open(source_path)
    if os.path.exists(source_path) and os.path.isdir(source_path):
        for parent_path in os.listdir(source_path):
            parent_path_tmp = source_path + "\\" + parent_path
            for short_file_name in os.listdir(parent_path_tmp):
                file_name = parent_path_tmp + "\\" + short_file_name
                in_file = open(file_name)
                if not os.path.exists(target_path + "\\" + parent_path):
                    os.mkdir(target_path + "\\" + parent_path)
                out_file = open(target_path + "\\dict.txt", "a+")
                while True:
                    line = in_file.readline().replace("\n", "")
                    if not line:
                        break
                    word = line.replace("\n", "").split(" ")[-1]
                    out_file.write(word + "\n")
                    print word
                in_file.close()
                out_file.close()


def clean_data_every(source_path, target_path):
    # grand_path = open(source_path)
    if os.path.exists(source_path) and os.path.isdir(source_path):
        for parent_path in os.listdir(source_path):
            parent_path_tmp = source_path + "\\" + parent_path
            for short_file_name in os.listdir(parent_path_tmp):
                file_name = parent_path_tmp + "\\" + short_file_name
                in_file = open(file_name)
                print(target_path + "\\" + parent_path)
                if not os.path.exists(target_path + "\\" + parent_path):
                    os.mkdir(target_path + "\\" + parent_path)
                out_file = open(target_path + "\\" + parent_path + "\\dict.txt", "a+")
                while True:
                    line = in_file.readline().replace("\n", "")
                    if not line:
                        break
                    word = line.replace("\n", "").split(" ")[-1]
                    out_file.write(word + "\n")
                    print word
                in_file.close()
                out_file.close()


if __name__ == '__main__':
    clean_data_total("D:\sougou_pinyin", "D:\sougou_pinyin_target")
