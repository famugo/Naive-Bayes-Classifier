
#coding: utf-8
import os
import time
import random
import jieba
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r',encoding = 'utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set

def TextProcessing(folder_path, test_size=0.2):
    folder_list = os.listdir(folder_path) # 查看folder_path下的文件
    data_list = []
    class_list = []

    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path) #获取子文件夹中的所有文件

        # 类内循环
        j = 1
        # 遍历每个文件
        for file in files:
            if j > 100: # 每类text样本数最多100
                break
            with open(os.path.join(new_folder_path, file), 'r',encoding = 'utf-8') as fp:
               raw = fp.read()
            # print raw
            ## --------------------------------------------------------------------------------
            ## jieba分词
            # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
            word_cut = jieba.cut(raw, cut_all=False) # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut) # genertor转化为list，每个词unicode格式
            # jieba.disable_parallel() # 关闭并行分词模式
            # print word_list
            ## --------------------------------------------------------------------------------
            data_list.append(word_list)
            class_list.append(folder)
            j += 1

    ## 划分训练集和测试集
    # train_data_list, test_data_list, train_class_list, test_class_list = sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)
    data_class_list = list(zip(data_list, class_list))  # 打包为list，元素为元组
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size)+1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True) # 内建函数sorted参数需为list
    all_words_list = [word for words_tuple in all_words_tuple_list for word in words_tuple]

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def words_dict(all_words_list, deleteN, stopwords_set, min_features=1):
    feature_words = []
    n = 0
    for t in range(len(all_words_list)):
        if isinstance(all_words_list[t], str) and not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and len(all_words_list[t]) >= 2:
            feature_words.append(all_words_list[t])
            n += 1
            if n >= min_features:
                break
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words, flag='nltk'):
    def text_features(text, feature_words):
        text_words = set(text)
        ## -----------------------------------------------------------------------------------
        if flag == 'nltk':
            ## nltk特征 dict
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ## sklearn特征 list
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag):
    if not train_feature_list or not test_feature_list:
        return 0
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy


if __name__ == '__main__':

    print("start")

    ## 文本预处理
    folder_path = r'D:\github项目\贝叶斯分类\Naive-Bayes-Classifier\Database\SogouC\Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)

    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    ## 文本特征提取和分类
    # flag = 'nltk'
    flag = 'sklearn'
    deleteNs = range(0, 1000, 20)
    test_accuracy_list = []
    for deleteN in deleteNs:
        # feature_words = words_dict(all_words_list, deleteN)
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words, flag)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
        test_accuracy_list.append(test_accuracy)
    print("test_accuracy_list:\n",test_accuracy_list)

    # 结果评价
    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.savefig('result.png')

    print("finished")