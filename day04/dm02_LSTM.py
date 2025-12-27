# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

def dm1_lstm_base():
    # 1.实例化模型
    # LSTM的参数说明：
    # 第一个参数input_size：输入的词嵌入维度
    # 第二个参数hidden_size：RNN单元输出的隐藏层张量的维度
    # 第三个参数num_layers：有几层RNN单元（有几个隐藏层）
    input_size = 5
    hidden_size = 6
    num_layers = 2
    model = nn.LSTM(input_size, hidden_size, num_layers)

    # 2. 获取x0输入
    # x0的参数说明
    # 第一个参数sequence_len：每个样本的长度（单词的个数）(因为RNN模型batch_first=False, seq_len放在第一位置)
    # 第二个参数batch_size：一个批次送入几个样本
    # 第三个参数input_size：输入的词嵌入维度
    sequence_len = 4
    batch_size = 3
    x0 = torch.randn(sequence_len, batch_size, input_size)

    # 3.获取h0\c0输入
    # h0、c0的参数说明
    # 第一个参数num_layers：有几层RNN单元（有几个隐藏层）
    # 第二个参数batch_size：一个批次送入几个样本
    # 第三个参数hidden_size：RNN单元输出的隐藏层张量的维度

    h0 = torch.randn(num_layers, batch_size, hidden_size)
    c0 = torch.randn(num_layers, batch_size, hidden_size)

    # 4.将输入送给RNN模型得到下一时间步的输出结果
    output, (hn, cn) = model(x0, (h0, c0))

    print(f'output--》{output}')
    print('*'*80)
    print(f'hn--》{hn}')
    print(f'cn--》{cn}')

if __name__ == '__main__':
    dm1_lstm_base()


