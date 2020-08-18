# -*- coding: utf-8 -*-
'''
将训练数据使用jieba分词工具进行分词。并且剔除stopList中的词。
得到词表：
        词表的每一行的内容为：词 词的序号 词的频次
'''


import json
import jieba
from tqdm import tqdm

# trainFile = 'baike_qa2019/my_traindata.json'
allFile = 'baike_qa2019/all_data.json'
stopwordFile = 'stopword.txt'
wordLabelFile = 'wordLabel.txt'
lengthFile = 'length.txt'


def read_stopword(file):
    data = open(file, 'r', encoding='utf_8').read().split('\n')

    return data


def main():
    worddict = {}
    stoplist = read_stopword(stopwordFile)
    datas = open(allFile, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    data_num = len(datas)
    len_dic = {}
    for line in datas:
        line = json.loads(line)
        title = line['title']
        title_seg = jieba.cut(title, cut_all=False)
        length = 0
        for w in title_seg:
            if w in stoplist:
                continue
            length += 1
            if w in worddict:
                worddict[w] += 1
            else:
                worddict[w] = 1
        if length in len_dic:
            len_dic[length] += 1
        else:
            len_dic[length] = 1  # len_dic存放title分词为key个数的数量为value

    wordlist = sorted(worddict.items(), key=lambda item:item[1], reverse=True)  # 按词频从大到小排序
    f = open(wordLabelFile, 'w', encoding='utf_8')
    ind = 0
    for t in wordlist:
        d = t[0] + ' ' + str(ind) + ' ' + str(t[1]) + '\n'
        ind += 1
        f.write(d)

    for k, v in len_dic.items():
        len_dic[k] = round(v * 1.0 / data_num, 3)  # 归一化？
    len_list = sorted(len_dic.items(), key=lambda item:item[0], reverse=True)  # 按title分词个数从大到小排序
    f = open(lengthFile, 'w')
    for t in len_list:
        d = str(t[0]) + ' ' + str(t[1]) + '\n'
        f.write(d)

if __name__ == "__main__":
    main()