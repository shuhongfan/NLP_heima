# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import math
# todo:1.定义词嵌入层
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # vocab_size：代表词表的大小
        self.vocab_size = vocab_size
        # d_model：词嵌入维度的大小
        self.d_model = d_model
        # 定义Embedding层
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x--->[batch_size, seq_len]-->[2, 4]
        # embed_x-->[bacth_size, seq_len, d_model]-->[2, 4, 512]
        embed_x = self.embed(x)
        # 将embed_x乘以根号下d_model。原因：1.符合标准正态分布，2.增强embedding的影响
        return embed_x * math.sqrt(self.d_model)

# todo:2.定义位置编码器层（位置编码器的结果需要和embedding的结果进行相加）
class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout_p, max_len=60):
        super().__init__()
        # d_model:代表词嵌入的维度
        # dropout_p：随机失活的系数
        # max_len：样本的句子最大长度
        # 1.初始化位置编码矩阵pe-->[max_len, d_model]-->[60, 512]
        pe = torch.zeros(max_len, d_model)
        # 定义随机失活层
        self.dropout = nn.Dropout(p=dropout_p)
        # 2. 定义位置矩阵--》[max_len, 1]-->[60, 1]
        tem_vec = torch.arange(0, max_len).unsqueeze(1)
        # 3.根据公式定义转换矩阵（256个值）-->[256]
        div_vec = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0)/d_model)
        # 4.将每个位置先赋值256个向量值position:[60, 256]
        position = tem_vec * div_vec
        # 5.将pe值进行赋值，奇数位用sin，偶数位用cos
        pe[:, 0::2] = torch.sin(position)
        pe[:, 1::2] = torch.cos(position)
        # 6.需要升维度-->[1, 60, 512]
        pe = pe.unsqueeze(0)
        # 7.把pe位置编码矩阵 注册成模型的持久缓冲区buffer; 模型保存再加载时，可以根模型参数一样，一同被加载
        # 什么是buffer: 对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不参与模型训练
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x--->代表：embedding之后的结果：[batch_size, seq_len, embed_dim]-->[2, 4, 512]
        # 加上位置信息，因为之前的pe是最大句子长度的结果；注意取出真实的位置编码结果
        # self.pe[:, :x.shape[1]]-->[1, 4, 512]
        # print(f'{self.pe.requires_grad}')
        position_x = x + self.pe[:, :x.shape[1]]
        return self.dropout(position_x)

if __name__ == '__main__':
    my_embed = Embeddings(vocab_size=1000, d_model=512)
    x = torch.tensor([[1, 4, 7, 10],[2, 6, 40, 5]])
    embed_result = my_embed(x)

    position_encode = PositionEncoding(d_model=512, dropout_p=0.1)
    result = position_encode(embed_result)
    print('位置编码之后的结果', result.shape)
