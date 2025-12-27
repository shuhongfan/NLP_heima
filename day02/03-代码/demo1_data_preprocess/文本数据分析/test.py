import jieba
# def fun(x):
#     return x + 2
#
# a = map(fun, [1, 2, 3])
# # print(*a)
#
# b = map(lambda x: x+2, [1, 2, 3])
# print(list(b))
from itertools import chain
# list1 = [1, 2, 3]
# list2 = [2, 3, 4]
# a = chain(list1, list2)
# print(set(a))
# # map(lambda x: jieba.lcut(x), train_data["sentence"])

list1 = ["我爱黑马", "我爱你, 我爱黑马"]
a = map(lambda x: jieba.lcut(x), list1)
b = chain(*a)
print(set(b))