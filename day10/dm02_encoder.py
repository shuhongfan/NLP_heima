# -*-encoding:utf-8-*-
import copy
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dm01_input import *


# todo:1. 生成一个下三角矩阵

def sub_mask(size):
    b = torch.ones((1, size, size),dtype=torch.long)
    return 1 - torch.triu(b, diagonal=1)


# todo:2. 实现注意力计算公式的代码表示
def attention(query, key, value, mask=None, dropout=None):
    # 自注意力 query=key=value -->[2, 4, 512]
    # mask如果用到编码器端就是padding_mask,如果用到解码器端就是sentence_mask
    # dropout随机失活的对象
    # dk--》词嵌入的维度：512
    dk = query.size(-1)
    # 根据公式去计算注意力的分数
    # Q-->[2, 4, 512];K的转置--》[2, 512, 4]-->matmul-->[2, 4, 4]
    scores = torch.matmul(query, torch.transpose(key, -1, -2)) / math.sqrt(dk)
    # 如果mask不为空，需要对scores记性掩码
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)

    # scores进行softmax归一化
    atten_weight = F.softmax(scores, dim=-1)
    # print(f'atten_weight--》{atten_weight}')

    # 对上述的结果进行随机失活
    if dropout is not None:
        atten_weight = dropout(atten_weight)

    return torch.matmul(atten_weight, value), atten_weight


#

# 测试注意力的结果
def test_attention():
    # 假设编码器的输入
    x0 = torch.tensor([[1, 2, 3, 10],
                       [2, 5, 28, 6]])
    # 经过Embedding层
    vocab_size = 1000
    d_model = 512
    my_embed = Embeddings(vocab_size=vocab_size, d_model=d_model)
    embed_x = my_embed(x0)
    print(f'embed之后的结果是：{embed_x.shape}')
    # 经过位置编码器层（在位置编码器内部，我们其实已经融合来embed_x）
    dropout_p = 0.1
    my_pe = PositionEncoding(d_model=d_model, dropout_p=dropout_p)
    # embed_x送入位置编码器
    position_x = my_pe(embed_x)
    print(f'position_x之后的结果--》{position_x.shape}')
    # 因为是自注意力机制，简化：query=key=value=position_x
    query = key = value = position_x
    # atten_result, atten_weight = attention(query, key, value)
    # print(f'atten_result---》{atten_result.shape}')
    # print(f'atten_weight---》{atten_weight.shape}')
    # 假如加入掩码
    mask = torch.zeros(2, 4, 4)
    atten_result, atten_weight = attention(query, key, value, mask)
    print(f'atten_result---》{atten_result.shape}')
    print(f'atten_weight---》{atten_weight.shape}')

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# todo:3.实现多头注意力机制的运算
class MutiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head, dropout_p):
        super().__init__()

        #embed_dim:代表词嵌入的维度：512
        # head：代表头的个数
        # 一定要确保embed_dim整除head
        assert embed_dim % head == 0
        # 1.获取每个头的词嵌入维度
        self.d_k = embed_dim // head
        # 2.定义head的属性
        self.head = head
        # 3. 定义4个全连接层
        self.linears = clones(module=nn.Linear(embed_dim, embed_dim), N=4)

        # 4.定义属性atten_weight
        self.atten_weight = None
        # 5. 定义dropout层
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, query, key, value, mask=None):
        #  # 自注意力 query=key=value -->[2, 4, 512]
        # mask-->[head, seq_len, seq_len]-->[8, 4, 4]
        if mask is not None:
            mask = mask.unsqueeze(0) # [1, 8, 4, 4]
        self.batch_size = query.size(0)
        # 将原始的query，key,value都要经过一个linear层，进行线性的变化，然后在切分成多个头的形式
        # model(x)-->[2, 4, 512]
        # result = model(x).view(self.batch_size, -1,self.head, self.d_k)-->[2, 4, 8, 64]
        # result进行转置--》[2, 8, 4, 64]
        query, key, value = [model(x).view(self.batch_size, -1,self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value ))]

        # 调用attention方式实现多头注意力机制的运算
        # atten_result-->[2, 8, 4, 64]
        # query--》[2, 8, 4, 64]，key-->[2, 8, 4, 64]--转置>矩阵相乘--》[2,8,4, 4]--》atten_weight
        # [2,8,4, 4]要和value:[2, 8, 4, 64]-->[2, 8, 4, 64]
        atten_result, atten_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 将多头进行合并:result-->[2, 4, 512]
        result = atten_result.transpose(1, 2).contiguous().view(self.batch_size, -1, self.head*self.d_k)

        return self.linears[-1](result)

def test_mutiheadatten():
    # 假设编码器的输入
    x0 = torch.tensor([[1, 2, 3, 10],
                       [2, 5, 28, 6]])
    # 经过Embedding层
    vocab_size = 1000
    d_model = 512
    my_embed = Embeddings(vocab_size=vocab_size, d_model=d_model)
    embed_x = my_embed(x0)
    print(f'embed之后的结果是：{embed_x.shape}')
    # 经过位置编码器层（在位置编码器内部，我们其实已经融合来embed_x）
    dropout_p = 0.1
    my_pe = PositionEncoding(d_model=d_model, dropout_p=dropout_p)
    # embed_x送入位置编码器
    position_x = my_pe(embed_x)
    print(f'position_x之后的结果--》{position_x.shape}')
    # 因为是自注意力机制，简化：query=key=value=position_x
    query = key = value = position_x
    # atten_result, atten_weight = attention(query, key, value)
    # print(f'atten_result---》{atten_result.shape}')
    # print(f'atten_weight---》{atten_weight.shape}')
    # 实例化多头注意力机制的对象
    mha = MutiHeadAttention(embed_dim=512, head=8, dropout_p=0.1)
    # 假如加入掩码
    mask = torch.zeros(8, 4, 4)
    atten_result = mha(query, key, value, mask=mask)
    # print(f'atten_result---》{atten_result.shape}')
    # print(f'atten_result---》{atten_result}')
    return atten_result

# todo:4.实现前馈全连接层的运算
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        super().__init__()
        # d_model:词嵌入维度
        # d_ff：前馈全连接内部特征维度
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # x-->来自于第一个子层连接结构的结果（多头注意力机制计算的结果）
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# todo:5.实现规范化层
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # features:词嵌入维度
        # eps为一个常数，防止分母为0
        # 可学习的参数a
        self.a = nn.Parameter(torch.ones(features))
        # 可学习的参数b-->bias
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # x-->可能来自于多头自主自注意力机制层，也可能来自于自前馈全连接层；x-->[2, 4, 512]
        # 获取张量x的均值
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        # 获取张量x的标准差
        x_std = torch.std(x, dim=-1, keepdim=True)

        return self.a*(x-x_mean)/(x_std+self.eps) + self.b


# todo:6.实现子层连接结构
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout_p):
        super().__init__()
        # size :词嵌入维度，是实例化LayerNorm使用的
        self.norm = LayerNorm(features=size)
        # 实例化dropout
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, sublayer):
        # x: 如果sublayer是多头注意力机制，x代表原始的输入部分，如果sublayer是前馈全连接层，x代表第一层的输出结果
        # x-->[2, 4, 512]
        # sublayer:代表的是一个函数对象或者是前馈全连接层对象
        # 举例：sublayer-->:lambda x: atten(x, x, x, mask)
        # 第一种方式：post_norm
        result = x + self.dropout(self.norm(sublayer(x)))
        # 第二种方式：pre_norm
        # result = x + self.dropout(sublayer(self.norm(x)))
        return result

def test_sublayer():
    # 假设编码器的输入
    x0 = torch.tensor([[1, 2, 3, 10],
                       [2, 5, 28, 6]])
    # 经过Embedding层
    vocab_size = 1000
    d_model = 512
    my_embed = Embeddings(vocab_size=vocab_size, d_model=d_model)
    embed_x = my_embed(x0)
    print(f'embed之后的结果是：{embed_x.shape}')
    # 经过位置编码器层（在位置编码器内部，我们其实已经融合来embed_x）
    dropout_p = 0.1
    my_pe = PositionEncoding(d_model=d_model, dropout_p=dropout_p)
    # embed_x送入位置编码器
    position_x = my_pe(embed_x)
    print(f'position_x之后的结果--》{position_x.shape}')
    # 因为是自注意力机制，简化：query=key=value=position_x
    # 实例化多头注意力机制的对象
    mha = MutiHeadAttention(embed_dim=512, head=8, dropout_p=0.1)
    # 假如加入掩码
    mask = torch.zeros(8, 4, 4)
    # 定义一个匿名函数
    sub_layer = lambda x: mha(x, x, x, mask)
    #实例化子层连接结构
    my_sublayer_connect = SublayerConnection(size=512, dropout_p=0.1)
    result = my_sublayer_connect(x=position_x, sublayer=sub_layer)
    print(f'第一个子层包含多头自注意力机制的结果-》{result}')
    print(f'第一个子层包含多头自注意力机制的结果-》{result.shape}')


# todo:7 定义编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_atten, ff, dropout_p):
        super().__init__()
        # size:词嵌入维度
        self.size = size
        # self_atten:代表：多头自注意力机制的对象
        self.self_atten = self_atten
        # ff:代表：前馈全连接层的对象
        self.ff = ff
        # 因为一个编码器层由两个子层连接结构堆叠而成，所以要克隆两个子层连接结构的对象
        self.sublayers = clones(SublayerConnection(size, dropout_p), 2)

    def forward(self, x, mask):
        # x--》来自于输入部分：wordEmbedding+positionEncoding:[2, 4, 512]
        # mask--》是否需要进行掩码，多头注意力形状--》[head, seq_len,seq_len]-->[8, 4, 4,]
        # 1。将x和mask送入第一个子层连接结构
        x1 = self.sublayers[0].forward(x, lambda x: self.self_atten(x, x, x, mask))
        # x1 = self.sublayers[0](x, lambda x: self.self_atten(x, x, x, mask))
        # 2。将第一个子层连接结构的结果送入第二个子层连接结构
        x2 = self.sublayers[1].forward(x1, self.ff)
        # x2 = self.sublayers[1](x1, self.ff)
        return x2

def test_encoderlayer():
    # 假设编码器的输入
    x0 = torch.tensor([[1, 2, 3, 10],
                       [2, 5, 28, 6]])
    # 经过Embedding层
    vocab_size = 1000
    d_model = 512
    my_embed = Embeddings(vocab_size=vocab_size, d_model=d_model)
    embed_x = my_embed(x0)
    print(f'embed之后的结果是：{embed_x.shape}')
    # 经过位置编码器层（在位置编码器内部，我们其实已经融合来embed_x）
    dropout_p = 0.1
    my_pe = PositionEncoding(d_model=d_model, dropout_p=dropout_p)
    # embed_x送入位置编码器
    position_x = my_pe(embed_x)
    print(f'position_x之后的结果--》{position_x.shape}')
    # 因为是自注意力机制，简化：query=key=value=position_x
    # 实例化多头注意力机制的对象
    mha = MutiHeadAttention(embed_dim=512, head=8, dropout_p=0.1)
    # 假如加入掩码
    mask = torch.zeros(8, 4, 4)
    # 实例化前馈全连接层对象
    ff = FeedForward(d_model=512, d_ff=1024)
    # 实例化编码器层对象
    encoder_layer = EncoderLayer(size=512, self_atten=mha, ff=ff, dropout_p=dropout_p)
    output = encoder_layer(position_x, mask)
    print(f'一个编码器层得到的结果--》{output}')
    print(f'一个编码器层得到的结果--》{output.shape}')

# todo:8 定义编码器(由6个编码器层堆叠而成)
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        # layer:代表：编码器层的对象
        # N：代表：几个编码器层
        self.layers = clones(layer, N)
        # 实例化对象LayerNorm
        self.layerNorm = LayerNorm(features=layer.size)

    def forward(self, x, mask):
        # x--》来自于输入部分：wordEmbedding+positionEncoding:[2, 4, 512]
        # mask--》是否需要进行掩码，多头注意力形状--》[head, seq_len,seq_len]-->[8, 4, 4,]
        for layer in self.layers:
            x = layer(x, mask)
        return self.layerNorm(x)

def test_encoder():
    # 假设编码器的输入
    x0 = torch.tensor([[1, 2, 3, 10],
                       [2, 5, 28, 6]])
    # 经过Embedding层
    vocab_size = 1000
    d_model = 512
    my_embed = Embeddings(vocab_size=vocab_size, d_model=d_model)
    embed_x = my_embed(x0)
    # print(f'编码器embed之后的结果是：{embed_x.shape}')
    # 经过位置编码器层（在位置编码器内部，我们其实已经融合来embed_x）
    dropout_p = 0.1
    my_pe = PositionEncoding(d_model=d_model, dropout_p=dropout_p)
    # embed_x送入位置编码器
    position_x = my_pe(embed_x)
    # print(f'编码器position_x之后的结果--》{position_x.shape}')
    # 因为是自注意力机制，简化：query=key=value=position_x
    # 实例化多头注意力机制的对象
    mha = MutiHeadAttention(embed_dim=512, head=8, dropout_p=0.1)
    # 假如加入掩码
    mask = torch.zeros(8, 4, 4)
    # 实例化前馈全连接层对象
    ff = FeedForward(d_model=512, d_ff=1024)
    # 实例化编码器层对象
    encoder_layer = EncoderLayer(size=512, self_atten=mha, ff=ff, dropout_p=dropout_p)
    #  实例化编码器对象
    encoder = Encoder(layer=encoder_layer, N=6)
    output = encoder(position_x, mask)
    # print(f'编码器得到的结果--》{output}')
    # print(f'编码器得到的结果--》{output.shape}')
    return output
if __name__ == '__main__':
    # test_sublayer()
    # test_encoderlayer()
    test_encoder()