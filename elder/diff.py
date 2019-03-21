# encoding=utf-8

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


# 加载词库文件
def loadJieba(filePath):
    my_jieba = {}
    try:
        file = open(filePath, "r")  # 只读方式打开文件
        while True:
            line = file.readline()  # 读取行
            if not line:
                break
            word = line.split(" ")  # 词、词频、词性之间用" "分开
            my_jieba[word[0]] = word[1:]
        file.close()
    except:
        print "File does not found"
    return my_jieba


# 找不同
def getDiff(my_jieba, target_path):
    new_word = {}
    list = os.listdir(target_path)  # 获取路径下所有文件
    for fileName in list:
        fileName = target_path + "\\" + fileName  # 拼接路径与文件名
        file = open(fileName, "r")  # 只读方式打开文件
        while True:
            line = file.readline().replace("\n", "")
            if not line:
                break
            word = line.split(" ")
            # 判断word[0]是否在my_jieba或new_word里，若没有则加入new_word
            if not my_jieba.get(word[0].strip()) and not new_word.get(word[0].strip()):
                try:
                    word[0].decode('utf8')  # 使用utf-8解码，若有异常则为乱码
                except:
                    continue
                new_word[word[0].strip()] = word[1:]
    return new_word


# 将新词写入文件
def writeToFile(new_word, fileName):
    # 追加方式打开文件
    file = open(fileName, "a+")
    for word in new_word.keys():
        attr = new_word.get(word)
        line = word + " " + " ".join(attr) + "\n"
        file.write(line)
    file.close()
    print "writing Done! Total:" + len(new_word) + "line."


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "The arguments you have typed are error"
        sys.exit()

    jiebaLib = sys.argv[1]  # 获取第一个参数
    wordDataPath = sys.argv[2]  # 获取第二个参数
    jiebaTmpLib = sys.argv[3]  # 获取第三个参数

    # jieba.analyse.set_stop_words("stop_words.txt")

    my_jieba = loadJieba(jiebaLib)

    new_word = getDiff(my_jieba, wordDataPath)

    print "The word lib length is:", len(my_jieba)
    print "The number of new words is:", len(new_word)

    writeToFile(new_word, jiebaTmpLib)
