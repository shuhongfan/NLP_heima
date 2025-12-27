# -*-coding:utf-8-*-
import fasttext
# todo:1.数据未处理之前
# 1.初始训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking.train')
# 2.保存模型
# model.save_model('./fasttext_data/ai23_old.bin')
# 3.加载模型
# model = fasttext.load_model('./fasttext_data/ai23_old.bin')
# 4.实现预测
# result1 = model.predict("Which baking dish is best to bake a banana bread ?")
# result1 = model.predict("Why not put knives in the dishwasher?")
# print(f'初始模型预测的结果--》{result1}')
# 5.查看模型在验证集上的表现
# result2 = model.test('./fasttext_data/cooking.valid')
# print(f'数据未经过处理的验证集结果--》{result2}')

# # todo:2.数据清洗之后
# # 1.初始训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train')
# # 2.保存模型
# model.save_model('./fasttext_data/ai23_new.bin')
# # 3.加载模型
# model = fasttext.load_model('./fasttext_data/ai23_new.bin')
# # 4.实现预测
# # result1 = model.predict("Which baking dish is best to bake a banana bread ?")
# result1 = model.predict("Why not put knives in the dishwasher?")
# print(f'初始模型预测的结果--》{result1}')
# # 5.查看模型在验证集上的表现
# result2 = model.test('./fasttext_data/cooking.valid')
# print(f'数据经过预处理的验证集结果--》{result2}')
#todo:2.增加训练轮次
# 训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train', epoch=25)
# #2.实现预测
# # result1 = model.predict("Which baking dish is best to bake a banana bread ?")
# # result1 = model.predict("Why not put knives in the dishwasher?")
# # print(f'初始模型预测的结果--》{result1}')
# # 3.查看模型在验证集上的表现
# result2 = model.test('./fasttext_data/cooking.valid')
# print(f'增加训练轮次后的验证集结果--》{result2}')

#todo:3.修改学习率
# # 1. 训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train', epoch=25, lr=1.0)
# #2.实现预测
# # result1 = model.predict("Which baking dish is best to bake a banana bread ?")
# # result1 = model.predict("Why not put knives in the dishwasher?")
# # print(f'初始模型预测的结果--》{result1}')
# # 3.查看模型在验证集上的表现
# result2 = model.test('./fasttext_data/cooking.valid')
# print(f'修改完学习率的验证集结果--》{result2}')

# #todo:3. 增加n-gram特征
# # 1. 训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train', epoch=25, lr=1.0,wordNgrams=2)
# #2.实现预测
# # result1 = model.predict("Which baking dish is best to bake a banana bread ?")
# # result1 = model.predict("Why not put knives in the dishwasher?")
# # print(f'初始模型预测的结果--》{result1}')
# # 3.查看模型在验证集上的表现
# result2 = model.test('./fasttext_data/cooking.valid')
# print(f'增加n-gram特征后的验证集结果--》{result2}')


# #todo:4. 修改损失方式loss='hs'加速训练
# # 1. 训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train', epoch=25, lr=1.0,wordNgrams=2, loss='hs')
# #2.实现预测
# # result1 = model.predict("Which baking dish is best to bake a banana bread ?")
# # result1 = model.predict("Why not put knives in the dishwasher?")
# # print(f'初始模型预测的结果--》{result1}')
# # 3.查看模型在验证集上的表现
# result2 = model.test('./fasttext_data/cooking.valid')
# print(f'修改损失方式loss的验证集结果--》{result2}')

# #todo:5. 自动超参数调优
# # 1. 训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train', autotuneValidationFile='./fasttext_data/cooking_pre.valid', autotuneDuration=60)
# #2.实现预测
# # result1 = model.predict("Which baking dish is best to bake a banana bread ?")
# # result1 = model.predict("Why not put knives in the dishwasher?")
# # print(f'初始模型预测的结果--》{result1}')
# # 3.查看模型在验证集上的表现
# result2 = model.test('./fasttext_data/cooking.valid')
# print(f'自动超参数调优验证集结果--》{result2}')


#todo:6.修改损失方式"ova"
# 1. 训练模型
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train', epoch=25, lr=0.5, wordNgrams=2, loss='ova')
# 2.保存模型
# model.save_model('./fasttext_data/ai23_new1.bin')
# 3.加载模型
model = fasttext.load_model('./fasttext_data/ai23_new1.bin')
#2.实现预测
result1 = model.predict("Which baking dish is best to bake a banana bread ?", k=-1, threshold=0.9)
# result1 = model.predict("Why not put knives in the dishwasher?", k=-1, threshold=0.2)
print(f'初始模型预测的结果--》{result1}')
# 3.查看模型在验证集上的表现
# result2 = model.test('./fasttext_data/cooking.valid')
# print(f'修改模型loss为ova后的验证集结果--》{result2}')