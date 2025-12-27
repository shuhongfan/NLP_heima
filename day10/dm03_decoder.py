# -*-encoding:utf-8-*-
import copy
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dm01_input import *
from dm02_encoder import *

# todo: 1.定义解码器层（由三个子层连接结构构成）
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout_p):
        super().__init__()
        # size:词嵌入维度
        self.size = size
        # self_attn:代表自注意力机制：Q=K=V
        self.self_attn = self_attn
        # src_attn：代表一般注意力机制：Q!=K=V
        self.src_attn = src_attn
        # feed_forward:代表前馈全连接层
        self.feed_forward = feed_forward
        # 克隆三个子层连接结构的对象
        self.sub_layers = clones(SublayerConnection(size, dropout_p), 3)

    def forward(self, y, encoder_output, source_mask, target_mask):
        # y-->来自于解码器端的输入：【2，4， 512】
        # encoder_output--》来自于编码器的输出结果：【2，4，512】
        # source_mask：作用到第二个子层连接结构的多头注意力机制对象上，进行padding mask
        # target_mask：作用到第一个子层连接结构的多头自注意力机制对象上，进行sentence mask(casual mask)
        # 1.将y送入第一个子层连接结构得到多头自注意力机制+add+norm之后的结果
        y1 = self.sub_layers[0].forward(y, lambda y: self.self_attn(y, y, y, target_mask))
        # y1 = self.sub_layers[0](y, lambda y: self.self_attn(y, y, y, target_mask))
        # 2.将y1送入第二个子层连接结构得到多头注意力机制+add+norm之后的结果
        y2 = self.sub_layers[1].forward(y1, lambda y1: self.src_attn(y1, encoder_output, encoder_output, source_mask))
        # y2 = self.sub_layers[1](y1, lambda y1: self.src_attn(y1, encoder_output, encoder_output, source_mask))
        # 3. 将y2送入第三个子层连接结构得到前馈全连接层+add+norm之后的结果
        y3 = self.sub_layers[2](y2, self.feed_forward)

        return y3

def test_decoder_layer():
    # 定义假如解码器端的输入也是二行6列
    y = torch.tensor([[3, 4, 7, 10, 5, 70],
                      [2, 5, 8, 19, 20, 34]])
    # 1.经过embedding层得到词嵌入的结果
    my_embed = Embeddings(vocab_size=2000, d_model=512)
    embed_y = my_embed(y)
    print(f'解码器embed_y之后的结果--》{embed_y.shape}')

    # 2.经过position Encoding层得到位置编码的信息
    my_pe = PositionEncoding(d_model=512, dropout_p=0.1,)
    position_y = my_pe(embed_y)
    print(f'解码器position_y之后的结果--》{position_y.shape}')

    # 3. 实例化多头注意力机制的对象
    atten = MutiHeadAttention(embed_dim=512, head=8, dropout_p=0.1)
    self_atten = copy.deepcopy(atten)
    src_atten = copy.deepcopy(atten)

    # 4. 实例化前馈全连接层的对象
    feed_forward = FeedForward(d_model=512, d_ff=1024)

    # 5.实例化解码器层对象
    decoder_layer = DecoderLayer(size=512, self_attn=self_atten, src_attn=src_atten, feed_forward=feed_forward, dropout_p=0.1)

    # 6.整理解码器层的输入：y, encoder_output, source_mask, target_mask
    encoder_output = test_encoder()
    print(f'编码器输出encoder_output--{encoder_output.shape}')
    # target_mask--》解码器的query和key--》[2, 6, 512]-->query和key的转置相乘之后--》[2,6,6]
    target_mask = torch.zeros(8, 6, 6)
    # source_mask--》解码器的query--》[2, 6, 512]-->和key(编码器)-->[2, 4, 512]，和key的转置相乘之后--》[2,6,4]
    source_mask = torch.zeros(8, 6, 4)
    output = decoder_layer(position_y, encoder_output, source_mask, target_mask)
    print(f'解码器层最终的输出结果--》{output.shape}')
    print(f'解码器层最终的输出结果--》{output}')

# todo: 2.定义解码器（由6个解码器层堆叠构成）
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        # 克隆6个解码器层
        self.layers = clones(layer, N)
        # 规范化层
        self.norm = LayerNorm(features=layer.size)

    def forward(self, y, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            y = layer(y, encoder_output, source_mask, target_mask)
        return self.norm(y)

def test_decoder():
    # 定义假如解码器端的输入也是二行6列
    y = torch.tensor([[3, 4, 7, 10, 5, 70],
                      [2, 5, 8, 19, 20, 34]])
    # 1.经过embedding层得到词嵌入的结果
    my_embed = Embeddings(vocab_size=2000, d_model=512)
    embed_y = my_embed(y)
    print(f'解码器embed_y之后的结果--》{embed_y.shape}')

    # 2.经过position Encoding层得到位置编码的信息
    my_pe = PositionEncoding(d_model=512, dropout_p=0.1,)
    position_y = my_pe(embed_y)
    print(f'解码器position_y之后的结果--》{position_y.shape}')

    # 3. 实例化多头注意力机制的对象
    atten = MutiHeadAttention(embed_dim=512, head=8, dropout_p=0.1)
    self_atten = copy.deepcopy(atten)
    src_atten = copy.deepcopy(atten)

    # 4. 实例化前馈全连接层的对象
    feed_forward = FeedForward(d_model=512, d_ff=1024)

    # 5.实例化解码器层对象
    decoder_layer = DecoderLayer(size=512, self_attn=self_atten, src_attn=src_atten, feed_forward=feed_forward, dropout_p=0.1)

    # 6.整理解码器层的输入：y, encoder_output, source_mask, target_mask
    encoder_output = test_encoder()
    print(f'编码器输出encoder_output--{encoder_output.shape}')
    # target_mask--》解码器的query和key--》[2, 6, 512]-->query和key的转置相乘之后--》[2,6,6]
    target_mask = torch.zeros(8, 6, 6)
    # source_mask--》解码器的query--》[2, 6, 512]-->和key(编码器)-->[2, 4, 512]，和key的转置相乘之后--》[2,6,4]
    source_mask = torch.zeros(8, 6, 4)
    # 实例化解码器对象
    decoder = Decoder(layer=decoder_layer, N=6)
    output = decoder(position_y, encoder_output, source_mask, target_mask)
    print(f'解码器最终的输出结果--》{output.shape}')
    print(f'解码器最终的输出结果--》{output}')
    return output
if __name__ == '__main__':
    # test_decoder_layer()
    test_decoder()
