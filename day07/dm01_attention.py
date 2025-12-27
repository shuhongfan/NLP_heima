# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# todo：进行按照第一种规则实现注意力的计算（按照讲义降维升维）

class MyAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super().__init__()
        # query_size:Q张量的最后一个维度
        self.query_size = query_size
        # key_size:K张量的最后一个维度
        self.key_size = key_size
        # value_size1:V张量的中间的维度
        self.value_size1 = value_size1
        # value_size2:V张量的最后一个维度
        self.value_size2 = value_size2
        # output_size:注意力指定最后输出维度
        self. output_size= output_size

        # 定义第一个全连接层：计算注意力的权重值
        # 输入特征，注意：Q和k要进行拼接，然后在输入：Q-->[1, 1, 32],K-->[1, 1, 32]-->[1,1,64]
        # 输出特征，注意：因为Linear之后的结果要和V--》[1, 32, 64],所以输出的维度一定是32
        self.att_weight = nn.Linear(query_size+key_size, value_size1)

        # 定义第二个全连接层：计算最终的注意力结果
        # 输入特征，注意：Q和第一步计算的结果要进行拼接，然后在输入：Q-->[1, 1, 32],步骤1-->[1, 1, 64]-->[1,1,96]
        # 输出特征，指定的输出output_size
        self.out = nn.Linear(query_size+value_size2, output_size)

    def forward(self, Q, K, V):
        # Q--》shape-->[1, 1, 32], K--》shape-->[1, 1, 32], V--》shape-->[1, 32, 64]
        # 按照注意力计算的步骤，三步走开始实现注意力的运算
        # 第一步：按照第一种注意力计算规则来实现Q\K｜V的运算
        # 1.1 将Q和K进行拼接;经过Linear得到权重
        # Q[0]-->[1, 32];K[0]-->[1, 32]-->concat-->[1, 64]
        # concat-->[1, 64]经过linear得到结果--》[1, 32]
        atten_weight = F.softmax(self.att_weight(torch.cat((Q[0], K[0]), dim=-1)), dim=-1)
        # print(f'atten_weight--》{atten_weight.shape}')
        # 1.2 将得到权重和V进行矩阵相乘
        # atten_weight-->[1, 32]-->升维-->[1, 1, 32], V--》[1, 32, 64]
        # temp_result -->[1, 1, 64]
        temp_result = torch.bmm(atten_weight.unsqueeze(dim=0), V)

        # 第二步：将Q和第一步计算的结果temp_result，进行拼接
        # Q[0]-->[1, 32];temp_result[0]-->[1, 64]-->concat-->[1, 96]
        cat_tensor = torch.cat((Q[0], temp_result[0]), dim=-1)
        # print(f'cat_tensor-->{cat_tensor.shape}')
        # 第三步：将第二步拼接之后的结果按照指定尺寸输出
        output = self.out(cat_tensor).unsqueeze(dim=0)
        # print(output.shape)
        return output, atten_weight



# todo：进行按照第一种规则实现注意力的计算（不进行降维升维）

class OriginalAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super().__init__()
        # query_size:Q张量的最后一个维度
        self.query_size = query_size
        # key_size:K张量的最后一个维度
        self.key_size = key_size
        # value_size1:V张量的中间的维度
        self.value_size1 = value_size1
        # value_size2:V张量的最后一个维度
        self.value_size2 = value_size2
        # output_size:注意力指定最后输出维度
        self. output_size= output_size

        # 定义第一个全连接层：计算注意力的权重值
        # 输入特征，注意：Q和k要进行拼接，然后在输入：Q-->[1, 1, 32],K-->[1, 1, 32]-->[1,1,64]
        # 输出特征，注意：因为Linear之后的结果要和V--》[1, 32, 64],所以输出的维度一定是32
        self.att_weight = nn.Linear(query_size+key_size, value_size1)

        # 定义第二个全连接层：计算最终的注意力结果
        # 输入特征，注意：Q和第一步计算的结果要进行拼接，然后在输入：Q-->[1, 1, 32],步骤1-->[1, 1, 64]-->[1,1,96]
        # 输出特征，指定的输出output_size
        self.out = nn.Linear(query_size+value_size2, output_size)

    def forward(self, Q, K, V):
        # Q--》shape-->[1, 1, 32], K--》shape-->[1, 1, 32], V--》shape-->[1, 32, 64]
        # 按照注意力计算的步骤，三步走开始实现注意力的运算
        # 第一步：按照第一种注意力计算规则来实现Q\K｜V的运算
        # 1.1 将Q和K进行拼接;经过Linear得到权重
        # Q-->[1, 1, 32];K[0]-->[1, 1, 32]-->concat-->[1,1, 64]
        # concat-->[1, 1, 64]经过linear得到结果--》[1, 1, 32]
        atten_weight = F.softmax(self.att_weight(torch.cat((Q, K), dim=-1)), dim=-1)
        # print(f'atten_weight--》{atten_weight.shape}')
        # 1.2 将得到权重和V进行矩阵相乘
        # atten_weight-->[1, 1, 32], V--》[1, 32, 64]
        # temp_result -->[1, 1, 64]
        temp_result = torch.bmm(atten_weight, V)

        # 第二步：将Q和第一步计算的结果temp_result，进行拼接
        # Q->[1, 1, 32];temp_result--->[1,1, 64]-->concat-->[1, 1, 96]
        cat_tensor = torch.cat((Q, temp_result), dim=-1)
        # print(f'cat_tensor-->{cat_tensor.shape}')
        # 第三步：将第二步拼接之后的结果按照指定尺寸输出
        output = self.out(cat_tensor)
        # print(output.shape)
        return output, atten_weight

if __name__ == '__main__':
    query_size = 32
    key_size = 32
    value_size1 = 32
    value_size2 = 64
    output_size = 32
    my_atten1 = MyAttention(query_size, key_size, value_size1, value_size2, output_size)
    print(my_atten1)
    Q = torch.randn((1, 1, 32))
    K = torch.randn((1, 1, 32))
    V = torch.randn((1, 32, 64))
    output1, atten_weight1 = my_atten1(Q, K, V)
    print(f'output1--》{output1.shape}')
    print(f'atten_weight1--》{atten_weight1.shape}')
    my_atten2 = OriginalAttention(query_size, key_size, value_size1, value_size2, output_size)
    output, atten_weight = my_atten2(Q, K, V)
    print(f'output--》{output.shape}')
    print(f'atten_weight--》{atten_weight.shape}')
