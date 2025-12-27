import torch
import torch.nn as nn
import torch.nn.functional as F
# embedding = nn.Embedding(100, 3, padding_idx=3)
# input = torch.tensor([[1, 4, 5, 9,3],
#                       [2, 3, 4, 99, 3]])
# print(embedding(input))
#
# a = {"我":0, "爱": 1, "pad": 3}
# b = torch.arange(0, 60, step=2)
# print(b)
# import math
# div_term = torch.exp(torch.arange(0, 20, 2) * -(math.log(10000.0) / 20))
# print(f'div_term--》{div_term}')
#
# a = torch.tensor([[0], [1], [2]])
# b = a*div_term
# print(b)
# a = torch.randn(60, 512)
# print(a[:, 0::2].shape)
# print(a[:, 1::2].shape)
#
# import numpy as np
# def dm_test_nptriu():
#     # 测试产生上三角矩阵
#     print(np.triu([[1, 1, 1, 1, 1],
#                    [2, 2, 2, 2, 2],
#                    [3, 3, 3, 3, 3],
#                    [4, 4, 4, 4, 4],
#                    [5, 5, 5, 5, 5]], k=1))
#     print(np.triu([[1, 1, 1, 1, 1],
#                    [2, 2, 2, 2, 2],
#                    [3, 3, 3, 3, 3],
#                    [4, 4, 4, 4, 4],
#                    [5, 5, 5, 5, 5]], k=0))
#     print(np.triu([[1, 1, 1, 1, 1],
#                    [2, 2, 2, 2, 2],
#                    [3, 3, 3, 3, 3],
#                    [4, 4, 4, 4, 4],
#                    [5, 5, 5, 5, 5]], k=-1))

# dm_test_nptriu()
# print(np.ones((1, 5, 5)))
# print()
# print(torch.from_numpy(1 - np.triu(m=np.ones((1, 5, 5)), k=1).astype('uint8')))
#
# a = torch.tensor([[1.5, 2.0, 3.6],
#                   [3.5, 4.6, 9.8],
#                   [4.6, 7.8, 9.3]])
# mask = torch.tensor([[1, 0, 0],
#                      [1, 1, 0],
#                      [1, 1, 1]])
# b = a.masked_fill(mask == 0, float('-inf'))
# print(b)
# print(F.softmax(b, dim=-1))
# a = torch.tensor([[[1, 2, 3, 0, 0],
#                   [2, 4, 3, 0, 0],
#                   [10, 2, 30, 0, 0],
#                   [0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0],
#                   [1, 2, 3, 4, 5]]])

# b = torch.ones(20, 3)
# print(b)
# # d = torch.zeros(5)
# print(d)
# print(b * a + d )
# print(a != 0)
# a[a!=0] = 1
# print(a)
# print(torch.transpose(a, -2, -1))
# print(torch.permute(a, (-1, -2)))
# print(a.size()[-1])
# c = torch.ones([1])
# if c:
#     print('你好')
#
# def fun(a):
#     assert a==2
#     print('你好')
#
# fun(a=3)

# x = torch.tensor([[[2.0, 4.0, 2.0],
#                   [2.0, 0.0, 4.0]],
#                   [[2.0, 4.0, 2.0],
#                    [2.0, 0.0, 4.0]]
#                   ])
#
# # ean = x.mean(-1,keepdims=True)
# # print(ean)
# a = torch.mean(x, dim=-1, keepdim=True)
# b = torch.std(x, dim=-1, keepdim=True)
# print(b)
# print((x-a)/b)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear1 = nn.Linear(4, 5)
        # self.linear2 = nn.Linear(5, 10)
        self.seq = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 10))

    def forward(self, x):
        # a = self.linear1(x)
        # b = self.linear2(a)

        b = self.seq(x)
        return b
if __name__ == '__main__':
    my_model = MyModel()
    x = torch.randn(2, 3, 4)
    print(my_model(x).shape)

