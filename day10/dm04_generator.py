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
from dm03_decoder import *

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # d_model:词嵌入维度
        # vocab_size：解码器端词表的大小
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        output = F.log_softmax(self.linear(x), dim=-1)
        return output

def test_generator():
    decoder_output = test_decoder()
    # 实例化输出层对象
    my_generator = Generator(d_model=512, vocab_size=2000)
    result = my_generator(decoder_output)
    print(f'最终transformer模型的输出结果：{result}')
    print(f'最终transformer模型的输出结果：{result.shape}')

if __name__ == '__main__':
    test_generator()