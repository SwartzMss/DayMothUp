#!/usr/bin/env python
# coding: utf-8

import os
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# 获取脚本文件的绝对路径
script_path = os.path.abspath(__file__)

# 获取脚本文件的目录
script_dir = os.path.dirname(script_path)

# 设置当前工作目录为脚本文件的目录
os.chdir(script_dir)


# 首先将GloVe模型文件转换为Word2Vec格式
glove_file = 'glove.6B.100d.txt' # GloVe预训练模型文件路径
word2vec_glove_file = 'glove.6B.100d.word2vec.txt' # Word2Vec格式模型输出文件路径
glove2word2vec(glove_file, word2vec_glove_file)

# 加载转换后的Word2Vec模型
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

print(f"model.most_similar('banana', topn=3)) = {model.most_similar('banana', topn=3)}")

# woman - king = man - queen 
result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print("woman - king = man - {}  and rate =  {:.4f}".format(*result[0]))
