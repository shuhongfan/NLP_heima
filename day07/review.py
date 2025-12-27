import torch
# # a = torch.randn(4, 4, 5)
# # b = torch.randn(4, 5, 6)
# #
# # c = torch.bmm(a, b)
# # print(c.shape)
# # #
# # d = torch.matmul(a, b)
# # print(d.shape)
# # c = torch.randn(3, 4)
# # d = torch.randn(4, 1)
# # # e = torch.matmul(c, d)
# # # print(e.shape)
# # f = torch.mm(c, d)
# # print(f'f--》{f.shape}')
# # a = torch.tensor([[[1, 2, 3],
# #                    [2, 3, 4]]])
# # # a-->shape->[1, 2, 3]
# # b = torch.tensor([[[10]],
# #                   [[4]]])
# # # b-->shape-->[2, 1, 1]
# # '''
# # [[[11, 12, 13],[12, 13, 14]],
# #  [[5, 6, 7],[6,7,8]]]
# # '''
# # #
# # # 广播机制：从后缘维度开始算起也就是最后一个维度开始计算，如果其中一个为1或者相等，那么就可以广播
# #
# # print(a+b)
# import re
# # s = "what!time? is it."
# # # print(s.lower().strip())
# # # print(re.sub(r'b', r'b*', s))
# # print(re.sub(r'([!.?])', r' \1', s))
# s = 'what？？？？ time is it'
# print(re.sub(r'[^a-zA-Z.!?]+', r' ', s))

# y = (1, 2)
# # print(x)
# print(y)
# a = torch.randn(2, 3, 4)
# print(a)
# b = a.view(-1, 6)
# print(b)
# print(b.shape)
# a = torch.randn(2, 3, 4)
# print(a)
# print(a[:].shape) # [2, 3, 4]
# print(a[1:])
# print(a[1:].shape) # [1, 3, 4]
# print(a[1:, :,].shape) #
# print(a[1:, :2,].shape) #[1, 2, 4]+2
# print(a[1:, :2,]) #[1, 2, 4]+2
# print(a[1:, 2,]) # [1, 4]
# print(a[0, :1, 3]) # [1]
# print(a[0, :, :300]) # [3, 4]+2;
# print(a[:, 2, :2])# [2,2,]+5
# print(a[:1, 2])#[1, 4]
# print(a[0, 2:]) # [1, 4]+3;
# print(a[0:, 1:])# [2, 2, 1]+3
# import random
#
# print(random.random())

a = torch.zeros(2, 3) # [2, 3]
print(a)
b = torch.tensor([[[1.2, 3.5, 3.6]]]) # [1, 1, 3]
print(b)
a[0] = b[0, 0]
print(a)