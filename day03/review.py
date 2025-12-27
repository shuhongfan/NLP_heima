# map: 第一个参数：func--》代表函数对象，第二个参数：iterable代表可迭代对象
# map的作用：将可迭代对象中的每个元素都经过func的处理，得到新的map对象（迭代器--》生成器）
# 可迭代对象：凡是能用for循环呢取出数据的对象，就是可迭代对象

# def fun(a):
#     return a+1


# list1 = [1, 2, 3]
#
# b = map(lambda a: a+1, list1)
# print(list(b))
# from itertools import chain
#
# list1 = [1, 2, 3]
# list2 = [2, 3]
# a = chain(list1, list2)
# print(set(a))
# a = list2 + list1
# print(a)
#
# list1.extend(list2)
# print(list1)
# abb = [[1, 2, 3], [2, 3]]
# d = chain(*abb)
# print(list(d))

# b = map(lambda x: [len(x)], ["nihao", 'ha', 'tes'])
#
# c = chain(*b)
# print(list(c))


# def fun(a):
#     return a+6
#
#
# # def fun1(c):
# #     return fun(c) + 2
# #
# #
# # print(fun1(4))
#
# a = lambda c: fun(c) + 2
#
# b = map(a, [1, 2, 3])
# print(list(b)) #[9, 10, 11]

# list1 = [1, 2, 3, 5]
# list2 = [2, 3, 4]
# list3 = [1, 2, 3, 4, 5, 6]
# a = zip(list1, list2, list3)
# print(list(a))
# b ,c, d = zip(*zip(list1, list2, list3))
# print(b)
# print(c)
# print(d)
# alist = [[1, 2, 3, 5],
#          [2, 3, 4],
#          [1, 2, 3, 4, 5, 6]]
# b = zip(*alist)
# print(list(b))
# input_list = [1, 3, 2, 1, 5, 3]
# # print()
# c = zip(*[input_list[i:] for i in range(2)])
# print(list(c))
# from torch.nn.utils.rnn import pad_sequence
