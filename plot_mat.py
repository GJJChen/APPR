import matplotlib.pyplot as plt
import numpy as np

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

tag_1 = ('SGD', 'adaDdelta', 'adaGrad', 'DRLP')
ratio_1 = [0.29471, 0.30394, 0.35456, 0.29675]
ratio_2 = [0.44627, 0.55417, 0.46719, 0.44853]
half_ratio = [0.37049, 0.42907, 0.41086, 0.37264]
ratio_1 = np.array(ratio_1)
ratio_2 = np.array(ratio_2)
half_ratio = np.array(half_ratio)
# half_ratio = (ratio_1 + ratio_2)/2

# plt.bar(tag_1, ratio_1)
# plt.title('平均恢复误差')

# plt.bar(tag_1, ratio_2)
# plt.title('平均预测误差')
#
plt.bar(tag_1, half_ratio)
plt.title('平均合计误差')

plt.show()

# def readData():
#
#     X = []
#     f = open('./Record/1/Record.txt')  # 打开数据文件文件
#     for line in f :  # 把lines中的数据逐行读取出来
#         list = line.strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
#         X.append(list)  # 把处理后的数据放到方阵A中。
#     return X
#
# data = np.array(readData())
# x_data = np.array(200)
# y_data = data[:, 1]
# # print(y_data)
# y_data2 = data[:, 1]
# # print(y_data2)
# # y_data3 = (y_data2 + y_data)/2
#
# ln1 = plt.plot(x_data, y_data, color='red', linewidth=2.0, linestyle='--')
# ln2 = plt.plot(x_data, y_data2, color='blue', linewidth=3.0, linestyle='-.')
# ln3 = plt.plot(x_data, y_data3, color='yellow', linewidth=2.5)
#
# plt.title("电子产品销售量")  # 设置标题及字体
#
# plt.legend(handles=[ln1, ln2, ln3], labels=['平均恢复误差', '平均预测误差','平均合计误差'])
#
# ax = plt.gca()
# ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
# ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
#
# plt.show()