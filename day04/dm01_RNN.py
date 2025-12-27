# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

def dm1_rnn_base():
    # 1.实例化模型
    # RNN的参数说明：
    # 第一个参数input_size：输入的词嵌入维度
    # 第二个参数hidden_size：RNN单元输出的隐藏层张量的维度
    # 第三个参数num_layers：有几层RNN单元（有几个隐藏层）
    input_size = 5
    hidden_size = 6
    num_layers = 1
    model = nn.RNN(input_size, hidden_size, num_layers)

    # 2. 获取x0输入
    # x0的参数说明
    # 第一个参数sequence_len：每个样本的长度（单词的个数）(因为RNN模型batch_first=False, seq_len放在第一位置)
    # 第二个参数batch_size：一个批次送入几个样本
    # 第三个参数input_size：输入的词嵌入维度
    sequence_len = 1
    batch_size = 3
    x0 = torch.randn(sequence_len, batch_size, input_size)

    # 3.获取h0输入
    # h0的参数说明
    # 第一个参数num_layers：有几层RNN单元（有几个隐藏层）
    # 第二个参数batch_size：一个批次送入几个样本
    # 第三个参数hidden_size：RNN单元输出的隐藏层张量的维度

    h0 = torch.randn(num_layers, batch_size, hidden_size)

    # 4.将输入送给RNN模型得到下一时间步的输出结果
    output, hn = model(x0, h0)

    print(f'output--》{output}')
    print('*'*80)
    print(f'hn--》{hn}')


def dm2_rnn_len():
    # 修改样本的长度
    # 1.实例化模型
    # RNN的参数说明：
    # 第一个参数input_size：输入的词嵌入维度
    # 第二个参数hidden_size：RNN单元输出的隐藏层张量的维度
    # 第三个参数num_layers：有几层RNN单元（有几个隐藏层）
    input_size = 5
    hidden_size = 6
    num_layers = 1
    model = nn.RNN(input_size,  hidden_size, num_layers)

    # 2. 获取x0输入
    # x0的参数说明
    # 第一个参数sequence_len：每个样本的长度（单词的个数）(因为RNN模型batch_first=False, seq_len放在第一位置)
    # 第二个参数batch_size：一个批次送入几个样本
    # 第三个参数input_size：输入的词嵌入维度
    sequence_len = 4
    batch_size = 3
    x0 = torch.randn(sequence_len, batch_size, input_size)
    print(f'x0 -->{x0.shape}')

    # 3.获取h0输入
    # h0的参数说明
    # 第一个参数num_layers：有几层RNN单元（有几个隐藏层）
    # 第二个参数batch_size：一个批次送入几个样本
    # 第三个参数hidden_size：RNN单元输出的隐藏层张量的维度

    h0 = torch.randn(num_layers, batch_size, hidden_size)
    print(f'h0--》{h0.shape}')

    # 4.将输入送给RNN模型得到下一时间步的输出结果(一次性送入模型)
    output, hn = model(x0, h0)
    print("一次性送入模型")
    print(f'output--》{output}')
    print('*'*80)
    print(f'hn--》{hn}')
    print('&'*80)
    # 5. 将一个token一个token往RNN模型里面去送
    # x0.size(0) = 4代表sequence_len
    # x0-->[4, 1, 5]
    # print(f'x0-->{x0}')
    # print(f'一个token一个token往RNN模型里面去送')
    # for idx in range(x0.size(0)):
    #     # print(x0[idx])
    #     temp = x0[idx].unsqueeze(dim=0)
    #     # print(f'temp--》{temp}')
    #     # break
    #     output, h0 = model(temp, h0)
    #     print(f'output--》{output}')
    #     print(f'h0--》{h0}')
    #     print('*'*80)

def dm3_rnn_batch():
    # batch的位置放到第一位
    # 1.实例化模型
    # RNN的参数说明：
    # 第一个参数input_size：输入的词嵌入维度
    # 第二个参数hidden_size：RNN单元输出的隐藏层张量的维度
    # 第三个参数num_layers：有几层RNN单元（有几个隐藏层）
    # batch_first默认为False，但是如果设置为True，那么input第一个参数是batch_size
    input_size = 5
    hidden_size = 6
    num_layers = 1
    model = nn.RNN(input_size,  hidden_size, num_layers, batch_first=True)

    # 2. 获取x0输入
    # x0的参数说明
    # 第一个参数sequence_len：每个样本的长度（单词的个数）(因为RNN模型batch_first=False, seq_len放在第一位置)
    # 第二个参数batch_size：一个批次送入几个样本
    # 第三个参数input_size：输入的词嵌入维度
    sequence_len = 4
    batch_size = 3
    x0 = torch.randn(batch_size, sequence_len, input_size)
    print(f'x0 -->{x0.shape}')

    # 3.获取h0输入
    # h0的参数说明
    # 第一个参数num_layers：有几层RNN单元（有几个隐藏层）
    # 第二个参数batch_size：一个批次送入几个样本
    # 第三个参数hidden_size：RNN单元输出的隐藏层张量的维度

    h0 = torch.randn(num_layers, batch_size, hidden_size)
    print(f'h0--》{h0.shape}')

    # 4.将输入送给RNN模型得到下一时间步的输出结果(一次性送入模型)
    output, hn = model(x0, h0)
    print("一次性送入模型")
    print(f'output--》{output}')
    print('*'*80)
    print(f'hn--》{hn}')
    print('&'*80)

def dm4_rnn_numlayers():
    # 多个RNN单元
    # 1.实例化模型
    # RNN的参数说明：
    # 第一个参数input_size：输入的词嵌入维度
    # 第二个参数hidden_size：RNN单元输出的隐藏层张量的维度
    # 第三个参数num_layers：有几层RNN单元（有几个隐藏层）
    # batch_first默认为False，但是如果设置为True，那么input第一个参数是batch_size
    input_size = 5
    hidden_size = 6
    num_layers = 2
    model = nn.RNN(input_size,  hidden_size, num_layers=2)

    # 2. 获取x0输入
    # x0的参数说明
    # 第一个参数sequence_len：每个样本的长度（单词的个数）(因为RNN模型batch_first=False, seq_len放在第一位置)
    # 第二个参数batch_size：一个批次送入几个样本
    # 第三个参数input_size：输入的词嵌入维度
    sequence_len = 4
    batch_size = 3
    x0 = torch.randn(sequence_len, batch_size, input_size)
    print(f'x0 -->{x0.shape}')

    # 3.获取h0输入
    # h0的参数说明
    # 第一个参数num_layers：有几层RNN单元（有几个隐藏层）
    # 第二个参数batch_size：一个批次送入几个样本
    # 第三个参数hidden_size：RNN单元输出的隐藏层张量的维度

    h0 = torch.randn(num_layers, batch_size, hidden_size)
    print(f'h0--》{h0.shape}')

    # 4.将输入送给RNN模型得到下一时间步的输出结果(一次性送入模型)
    output, hn = model(x0, h0)
    print("一次性送入模型")
    print(f'output--》{output}')
    print('*'*80)
    print(f'hn--》{hn}')
    print('&'*80)

def dm5_rnn_1():
    rnn = nn.RNN(5, 6)
    x = torch.randn(3, 4, 5)
    output, hn = rnn(x)
    print(f'output--》{output}')
    print(f'hn--》{hn}')
if __name__ == '__main__':
    # dm1_rnn_base()
    # dm2_rnn_len()
    # dm3_rnn_batch()
    # dm4_rnn_numlayers()
    dm5_rnn_1()