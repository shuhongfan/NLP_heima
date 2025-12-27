# coding:utf-8
import jieba
import torch
# from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import jieba
import torch.nn as nn


def dm_embeding_show():

    # 1 对句子分词 word_list
    sentence1 = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    sentence2 = "我爱自然语言处理"
    sentences = [sentence1, sentence2]

    # 2.对所有的句子进行分词
    word_list = list() # []
    for s in sentences:
        word_list.append(jieba.lcut(s))
    print(f'word_list--》{word_list}')

    #3. 获取word_index,index_word
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(word_list)
    print(f'tokenizer.word_index{tokenizer.word_index}')
    print(f'tokenizer.index_word{tokenizer.index_word}')
    # 4.将文本序列转换成数字序列
    seq_ids = tokenizer.texts_to_sequences(word_list)
    print(f'seq_ids---》{seq_ids}')
    # 5.获取样本中所有单词
    words = tokenizer.word_index.keys()
    print(f'words--》{words}')

    # 6.实例化Embedding对象
    # 6.1 num_embeddings:代表需要进行词向量表示的单词总个数（一定是去重），
    # 6.2 embedding_dim:代表每个单词进行词嵌入的维度，
    embed = nn.Embedding(num_embeddings=len(words), embedding_dim=8)
    print(f'embed.weight:{embed.weight}')
    # 7.可视化embedding
    # summary = S
    # 8. 获取每个单词对应的词向量
    for idx in range(len(tokenizer.index_word)):
        output = embed(torch.tensor(idx))
        # print(f'output--》{output}')
        print('%4s'%(tokenizer.index_word[idx+1]), output)
        print('*'*80)



if __name__ == '__main__':
    dm_embeding_show()
