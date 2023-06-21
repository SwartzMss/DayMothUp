import jieba
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
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
        return list(jieba.cut(s))
    else:
        return word_tokenize(s)

# 假设我们有一个参考句子和一个模型生成的句子
reference_sentence = '这是一个小测试'
candidate_sentence = '这是一个测试'

# 使用tokenize函数进行分词
reference = [tokenize(reference_sentence)]
candidate = tokenize(candidate_sentence)

# 计算BLEU得分
score = sentence_bleu(reference, candidate)

print(score)
#8.987727354491445e-155
