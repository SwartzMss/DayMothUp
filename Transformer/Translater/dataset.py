'''
德语->英语翻译数据集
参考: https://pytorch.org/tutorials/beginner/translation_transformer.html
'''

import de_core_news_sm
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k

# 下载翻译数据集
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
train_dataset = list(Multi30k(split='train', language_pair=('de', 'en')))

# 创建分词器
de_tokenizer=get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer=get_tokenizer('spacy', language='en_core_web_sm')

# 生成词表
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3     # 特殊token
UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM = '<unk>', '<pad>', '<bos>', '<eos>'

de_tokens=[] # 德语token列表
en_tokens=[] # 英语token列表
for de,en in train_dataset:
    de_tokens.append(de_tokenizer(de))
    en_tokens.append(en_tokenizer(en))


#这边表示的词表大小，也就是词向量token的大小,会把一些重复的单子 词组啥的进行组合
de_vocab=build_vocab_from_iterator(de_tokens,specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM],special_first=True) # 德语token词表
de_vocab.set_default_index(UNK_IDX)
en_vocab=build_vocab_from_iterator(en_tokens,specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM],special_first=True) # 英语token词表
en_vocab.set_default_index(UNK_IDX)

# 句子特征预处理
def de_preprocess(de_sentence):
    tokens=de_tokenizer(de_sentence)
    tokens=[BOS_SYM]+tokens+[EOS_SYM]
    ids=de_vocab(tokens)
    return tokens,ids


'''
en_sentence = Two young, White males are outside near many bushes.
tokens = ['Two', 'young', ',', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.']
'''
def en_preprocess(en_sentence):
    #print(f"en_sentence = {en_sentence}")
    tokens=en_tokenizer(en_sentence)
    #print(f"tokens = {tokens}")
    tokens=[BOS_SYM]+tokens+[EOS_SYM]
    ids=en_vocab(tokens)
    return tokens,ids

if __name__ == '__main__':
    # 词表大小
    print('de vocab:', len(de_vocab))
    print('en vocab:', len(en_vocab))

    # 特征预处理
    de_sentence,en_sentence=train_dataset[0]
    print('de preprocess:',*de_preprocess(de_sentence))
    print('en preprocess:',*en_preprocess(en_sentence))


    '''
    de vocab: 19214
    en vocab: 10837
    de preprocess: ['<bos>', 'Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.', '<eos>'] [2, 21, 85, 257, 31, 87, 22, 94, 7, 16, 112, 7910, 3209, 4, 3]
    en preprocess: ['<bos>', 'Two', 'young', ',', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.', '<eos>'] [2, 19, 25, 15, 1169, 808, 17, 57, 84, 336, 1339, 5, 3]
    '''