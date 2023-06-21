from rouge import Rouge
import jieba
from nltk.tokenize import word_tokenize
from langdetect import detect

def is_chinese(s):
    """
    判断输入字符串是否为中文
    """
    return detect(s) == 'zh-cn'

def tokenize(s):
    """
    根据输入字符串的语言选择不同的分词工具
    """
    if is_chinese(s):
        return ' '.join(list(jieba.cut(s)))
    else:
        return ' '.join(word_tokenize(s))

# 假设我们有一个参考句子和一个模型生成的句子
reference_sentence = '这是一个测试'
candidate_sentence = '这是一个测试'

# 使用tokenize函数进行分词
reference = tokenize(reference_sentence)
candidate = tokenize(candidate_sentence)

# 初始化rouge类
rouge = Rouge()

# 计算ROUGE得分
scores = rouge.get_scores(candidate, reference)[0]

'''
每种得分都包含了精确率（precision）、召回率（recall）和F1得分。然后，我们分别打印了这些得分。

ROUGE-1得分是基于单个词的匹配。
ROUGE-2得分是基于两个连续词的匹配。
ROUGE-L得分是基于最长公共子序列（LCS）的匹配。
请注意，精确率、召回率和F1得分的定义如下：

精确率是指预测为正的样本中实际为正的比例。
召回率是指实际为正的样本中预测为正的比例。
F1得分是精确率和召回率的调和平均值，用于综合评价精确率和召回率。
'''
# 打印ROUGE得分
print("ROUGE-1: ", scores['rouge-1'])  # 包含了ROUGE-1的精确率、召回率和F1得分
print("ROUGE-2: ", scores['rouge-2'])  # 包含了ROUGE-2的精确率、召回率和F1得分
print("ROUGE-L: ", scores['rouge-l'])  # 包含了ROUGE-L的精确率、召回率和F1得分
'''
ROUGE-1:  {'r': 0.75, 'p': 1.0, 'f': 0.8571428522448981}
ROUGE-2:  {'r': 0.3333333333333333, 'p': 0.5, 'f': 0.39999999520000007}
ROUGE-L:  {'r': 0.75, 'p': 1.0, 'f': 0.8571428522448981}
'''