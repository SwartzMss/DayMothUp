#!/usr/bin/env python
# coding: utf-8


import jieba
import re
import os
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib


# 获取脚本文件的绝对路径
script_path = os.path.abspath(__file__)

# 获取脚本文件的目录
script_dir = os.path.dirname(script_path)

# 设置当前工作目录为脚本文件的目录
os.chdir(script_dir)

'''
jieba 是一个用于中文分词的 Python 库。它的主要作用是将连续的文本分解为独立的词语。
这是许多自然语言处理任务（如情感分析、文本分类、信息检索等）的一个基础步骤，尤其对于中文文本来说，因为中文文本没有像英文那样的空格分隔符。
1. 精确模式：试图将句子最精确地切开，适合文本分析。
2. 全模式：把句子中所有的可以成词的词语都扫描出来，速度非常快，但是不能解决歧义。
3. 搜索引擎模式：在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
'''
f = open("sanguo.txt", 'r',encoding='utf-8') #读入文本
lines = []
for line in f: #分别对每段分词
    temp = jieba.lcut(line)  #结巴分词 精确模式
    words = []
    for i in temp:
        #过滤掉所有的标点符号
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
        if len(i) > 0:
            words.append(i)
    if len(words) > 0:
        lines.append(words)
print(lines[0:5])#预览前5行分词结果

# 调用Word2Vec训练
# 参数：size: 词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
model = Word2Vec(lines,vector_size = 20, window = 2 , min_count = 3, epochs=7, negative=10,sg=1)
print("孔明的词向量：\n",model.wv.get_vector('孔明'))
print("\n和孔明相关性最高的前5个词语：")
print(model.wv.most_similar('孔明', topn = 5))# 与孔明最相关的前5个词语


# 将词向量投影到二维空间
rawWordVec = []
word2ind = {}
for i, w in enumerate(model.wv.index_to_key): #index_to_key 序号,词语
    rawWordVec.append(model.wv[w]) #词向量
    word2ind[w] = i #{词语:序号}
rawWordVec = np.array(rawWordVec)
'''
PCA: 一种常用的数据分析方法。主成分分析是一种在高维数据中提取关键信息（主成分）的方法，它的目标是减少数据的维度，同时尽量保留数据的重要信息
这边的话将每个词的词向量从20维空间降低到2维空间，使得你可以在一个二维图形中展示词向量，同时尽量保留了词向量原本的信息
'''
X_reduced = PCA(n_components=2).fit_transform(rawWordVec) 


print(f"降维之前20维 = {rawWordVec}") #降维之前20维

print(f"降维之后2维 = {X_reduced}")

# ## 类比关系实验

words = model.wv.most_similar(positive=['玄德', '曹操'], negative=['孔明'], topn = 3)
print(f"玄德－孔明＝？－曹操 = {words}")


words = model.wv.most_similar(positive=['曹操', '蜀'], negative=['魏'], topn = 3)
print(f"曹操－魏＝？－蜀 =  {words}")








