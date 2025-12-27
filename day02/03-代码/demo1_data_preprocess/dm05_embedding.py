import torch
from keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import jieba
import torch.nn as nn

# 实验：nn.Embedding层词向量可视化分析
# 1 对句子分词 word_list
# 2 对句子word2id求my_token_list，对句子文本数值化sentence2id
# 3 创建nn.Embedding层，查看每个token的词向量数据
# 4 创建SummaryWriter对象, 可视化词向量
#   词向量矩阵embd.weight.data 和 词向量单词列表my_token_list添加到SummaryWriter对象中
#   summarywriter.add_embedding(embd.weight.data, my_token_list)
# 5 通过tensorboard观察词向量相似性
# 6 也可通过程序，从nn.Embedding层中根据idx拿词向量

def dm_embeding_show():

    # 1 对句子分词 word_list
    sentence1 = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    sentence2 = "我爱自然语言处理"
    sentences = [sentence1, sentence2]

    word_list = [] # list()
    for s in sentences:
        word_list.append(jieba.lcut(s))
    # print(f'word_list--》{word_list}')

    # 2 对每次词进行词表映射
    my_tokenizer = Tokenizer()
    my_tokenizer.fit_on_texts(word_list)
    print(f'my_tokenizer.word_index-->{my_tokenizer.word_index}')
    # print(f'my_tokenizer.index_word-->{my_tokenizer.index_word}')

     #获取所有词汇（去重）
    my_token_list = my_tokenizer.index_word.values()
    print(f'my_token_list-->{my_token_list}')
    print(f'my_token_list的长度-->{len(my_token_list)}')

    # 将句子的单词用数字进行表示
    seq2id = my_tokenizer.texts_to_sequences(word_list)
    # print(f'seq2id--》{seq2id}')

    # 3 创建Embedding层
    embed = nn.Embedding(num_embeddings=len(my_token_list), embedding_dim=8)
    # print(f'embed-->{embed.weight}')
    print(f'embed-->{embed.weight.data}')
    # print(f'embed-->{embed.weight.data.shape}')

    # 4 可视化展示
    # my_summary = SummaryWriter()
    # my_summary.add_embedding(embed.weight.data, my_token_list)
    # my_summary.close()

    # 5 取出每个单词对应的向量表示
    for idx in range(len(my_tokenizer.index_word)):
        temp_vector = embed(torch.tensor(idx))
        print(f'temp_vector--》{temp_vector}')
        word = my_tokenizer.index_word[idx+1]
        print(f'当前单词--》--<{word}>--的词向量是--》{temp_vector.detach().numpy()}')


if __name__ == '__main__':
    dm_embeding_show()