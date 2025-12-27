# import string
# all_letters = string.ascii_letters + " ,;.'"
# print(all_letters)
# print(all_letters.find('A'))
#
# list1 = [1, 2, "张三"]
# print(list1.index("张三"))
import torch
import torch.nn as nn

# a = torch.tensor([[1.0, 1.0, 1.0],
#                   [2.0, 2.0, 2.0]])
# # a(2,3)
# c = nn.Softmax(dim=-1)
# print(c(a))
#
# print('%.3f你好' % (1))

import json

# dict1 = {"loss": [1.0, 2.0, 3.0],
#          "acc": [3.6, 4.8, 6.7]}
# print(type(dict1))
# dict1_str = json.dumps(dict1)
# print(f'dict1_str--》{dict1_str}')
# print(f'dict1_str--》{type(dict1_str)}')
# with open('a.json', 'a') as fr:
#     fr.write(dict1_str)

# with open('a.json', 'r') as fr:
#     b = fr.readlines()
#     print(b[0])
#     print(type(b[0]))
#     c = json.loads(b[0])
#     print(c["loss"])
#     print(c)
#     print(type(c))
# a = torch.tensor([[8.2, 3.5, 4.6],
#                   [2.5, 4.3, 9.5]])
# # a.shape(2, 3)
# topv, topi = torch.topk(a, k=1, dim=0)
# print(f'topv--》{topv}')
# print(f'topi--》{topi}')

# a = 'abcde ef gf ji'
# print(a.split(' '))



