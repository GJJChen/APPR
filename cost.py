import numpy as np
import matplotlib.pyplot as plt
#
#
# def readSample():
#
#     #读取数据
#     X = np.zeros((20, 48), dtype=float)
#     f = open('actionSample.txt')  # 打开数据文件文件
#     lines = f.readlines()  # 把全部数据文件读到一个列表lines中
#     X_row = 0  # 表示矩阵的行，从0行开始
#     for i in range(20):  # 把lines中的数据逐行读取出来
#         list = lines[i].strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
#         X[X_row:] = list[0:48]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
#         X_row += 1  # 然后方阵A的下一行接着读
#     return X
#
# def readW():
#     X = np.zeros((20, 48), dtype=float)
#     f = open('W.txt')  # 打开数据文件文件
#     lines = f.readlines()  # 把全部数据文件读到一个列表lines中
#     X_row = 0  # 表示矩阵的行，从0行开始
#     for i in range(20):  # 把lines中的数据逐行读取出来
#         list = lines[i].strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
#         X[X_row:] = list[0:48]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
#         X_row += 1  # 然后方阵A的下一行接着读
#     return X
#
#
# X = readSample()
# W = readW()
# sum_cost = 0
# other_cost = 0
# t = 10

# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         if X[i][j] != 0:
#             sum_cost += W[i][j]
#
# print(sum_cost)
#
# for i in range(X.shape[1]-t):
#     j = 0
#     sample = np.zeros(20)
#     while j < 10:
#         b = np.random.randint(20, size=1)
#         if sample[b] == 0:
#             sample[b] = 1
#             j += 1
#     for k in range(len(sample)):
#         if sample[k] == 1:
#             other_cost += W[k][i+t]
#
# for i in range(X.shape[0]):
#     for j in range(t):
#         other_cost += W[i][j]
#
# print(other_cost)

# plt.figure(1)
#
# name_list = ['1000', '500', '250', '50', '20']
# num_list1 = [17, 36, 54, 200, 200]
# num_list2 = [21, 47, 65, 141, 200]
# num_list3 = [10, 16, 31, 177, 200]
# num_list = [10, 16, 29, 102, 183]
# x = list(range(len(num_list)))
# index = np.arange(len(name_list))
# total_width, n = 0.8, 4
# width = total_width / n
#
# plt.bar(index - 1.5 * width, num_list1, width=width, label='SGD', fc='r')
# plt.bar(index - 0.5 * width, num_list2, width=width, label='AdaDelta', fc='b')
# plt.bar(index + 0.5 * width, num_list3, width=width, label='Momentum', fc='g')
# plt.bar(index + 1.5 * width, num_list, width=width, label='DRRP', fc='purple')
#
# for a,b in zip(index - 1.5 * width,num_list1):   #柱子上的数字显示
#     plt.text(a, b, '%d' % b, ha='center', va='bottom',fontsize=7)
# for a,b in zip(index - 0.5 * width,num_list2):
#     plt.text(a, b, '%d' % b, ha='center', va='bottom',fontsize=7)
# for a, b in zip(index + 0.5 * width, num_list3):  # 柱子上的数字显示
#     plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=7)
# for a, b in zip(index + 1.5 * width, num_list):
#     plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=7)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.ylim(0, 241)
# plt.yticks(np.arange(0, 241, 40),np.arange(0, 241, 40))
# plt.xticks(np.arange(5), name_list)
# plt.grid(axis='y',linestyle=':')
# plt.xlabel("误差阈值（α=0.0015,λ=0.005）")
# plt.ylabel("分解开销")
# plt.legend()
# fig = plt.gcf()
# plt.show()
# fig.savefig("第1组分解开销.pdf")

#
# plt.figure(2)
#
# name_list = ['1000', '500', '250', '20', '10']
# num_list1 = [9, 18, 27, 200, 200]
# num_list2 = [11, 23, 32, 138, 200]
# num_list3 = [5, 8, 16, 200, 200]
# num_list = [5, 8, 16, 83, 138]
# x = list(range(len(num_list)))
# index = np.arange(len(name_list))
# total_width, n = 0.8, 4
# width = total_width / n
#
# plt.bar(index - 1.5 * width, num_list1, width=width, label='SGD', fc='r')
# plt.bar(index - 0.5 * width, num_list2, width=width, label='AdaDelta', fc='b')
# plt.bar(index + 0.5 * width, num_list3, width=width, label='Momentum', fc='g')
# plt.bar(index + 1.5 * width, num_list, width=width, label='DRRP', fc='purple')
#
# for a,b in zip(index - 1.5 * width,num_list1):   #柱子上的数字显示
#     plt.text(a, b, '%d' % b, ha='center', va='bottom',fontsize=7)
# for a,b in zip(index - 0.5 * width,num_list2):
#     plt.text(a, b, '%d' % b, ha='center', va='bottom',fontsize=7)
# for a, b in zip(index + 0.5 * width, num_list3):  # 柱子上的数字显示
#     plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=7)
# for a, b in zip(index + 1.5 * width, num_list):
#     plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=7)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.ylim(0, 241)
# plt.yticks(np.arange(0, 241, 40),np.arange(0, 241, 40))
# plt.xticks(np.arange(5), name_list)
# plt.grid(axis='y',linestyle=':')
# plt.xlabel("误差阈值（α=0.003,λ=0.005）")
# plt.ylabel("分解开销")
# plt.legend()
# fig = plt.gcf()
# plt.show()
# fig.savefig("第2组分解开销.pdf")
#
# plt.figure(3)
#
# name_list = ['1000', '500', '250', '50', '20']
# num_list = [10, 16, 30, 105, 166]
# num_list1 = [18, 36, 53, 200, 200]
# num_list2 = [21, 47, 65, 141, 200]
# num_list3 = [10, 15, 30, 175, 200]
# x = list(range(len(num_list)))
# index = np.arange(len(name_list))
# total_width, n = 0.8, 4
# width = total_width / n
#
# plt.bar(index - 1.5 * width, num_list1, width=width, label='SGD', fc='r')
# plt.bar(index - 0.5 * width, num_list2, width=width, label='AdaDelta', fc='b')
# plt.bar(index + 0.5 * width, num_list3, width=width, label='Momentum', fc='g')
# plt.bar(index + 1.5 * width, num_list, width=width, label='DRRP', fc='purple')
#
# for a,b in zip(index - 1.5 * width,num_list1):   #柱子上的数字显示
#     plt.text(a, b, '%d' % b, ha='center', va='bottom',fontsize=7)
# for a,b in zip(index - 0.5 * width,num_list2):
#     plt.text(a, b, '%d' % b, ha='center', va='bottom',fontsize=7)
# for a, b in zip(index + 0.5 * width, num_list3):  # 柱子上的数字显示
#     plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=7)
# for a, b in zip(index + 1.5 * width, num_list):
#     plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=7)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.ylim(0, 241)
# plt.yticks(np.arange(0, 241, 40),np.arange(0, 241, 40))
# plt.xticks(np.arange(5), name_list)
# plt.grid(axis='y',linestyle=':')
# plt.xlabel("误差阈值（α=0.0015,λ=0.0025）")
# plt.ylabel("分解开销")
# plt.legend()
# fig = plt.gcf()
# plt.show()
# fig.savefig("第3组分解开销.pdf")
#
# plt.figure(4)
#
# name_list = ['1000', '500', '250', '20', '10']
# num_list = [5, 8, 16, 87, 160]
# num_list1 = [9, 18, 26, 200, 200]
# num_list2 = [11, 23, 32, 136, 200]
# num_list3 = [5, 8, 16, 200, 200]
# x = list(range(len(num_list)))
# index = np.arange(len(name_list))
# total_width, n = 0.8, 4
# width = total_width / n
#
# plt.bar(index - 1.5 * width, num_list1, width=width, label='SGD', fc='r')
# plt.bar(index - 0.5 * width, num_list2, width=width, label='AdaDelta', fc='b')
# plt.bar(index + 0.5 * width, num_list3, width=width, label='Momentum', fc='g')
# plt.bar(index + 1.5 * width, num_list, width=width, label='DRRP', fc='purple')
#
# for a,b in zip(index - 1.5 * width,num_list1):   #柱子上的数字显示
#     plt.text(a, b, '%d' % b, ha='center', va='bottom',fontsize=7)
# for a,b in zip(index - 0.5 * width,num_list2):
#     plt.text(a, b, '%d' % b, ha='center', va='bottom',fontsize=7)
# for a, b in zip(index + 0.5 * width, num_list3):  # 柱子上的数字显示
#     plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=7)
# for a, b in zip(index + 1.5 * width, num_list):
#     plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=7)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.ylim(0, 241)
# plt.yticks(np.arange(0, 241, 40),np.arange(0, 241, 40))
# plt.xticks(np.arange(5), name_list)
# plt.grid(axis='y',linestyle=':')
# plt.xlabel("误差阈值（α=0.003,λ=0.0025）")
# plt.ylabel("分解开销")
# plt.legend()
# fig = plt.gcf()
# plt.show()
# fig.savefig("第4组分解开销.pdf")
#
# plt.figure(5)
#
# name_list = ['二部图采样', '随机采样']
# num_list = [66637.0, 123658.4]
#
# index = np.arange(len(name_list))
# width = 0.2
# plt.bar(index, num_list,width=width, fc='r')
#
# for a,b in zip(index,num_list):   #柱子上的数字显示
#     plt.text(a, b, '%d' % b, ha='center', va='bottom',fontsize=7)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.ylim(0, 150000)
# plt.yticks(np.arange(0, 150001, 30000),np.arange(0, 150001, 30000))
# plt.xlim(-1,2)
# plt.xticks(np.arange(2), name_list)
# plt.grid(axis='y',linestyle=':')
# plt.xlabel("采样规则")
# plt.ylabel("采样开销")
# # plt.legend()
# fig = plt.gcf()
# plt.show()
# fig.savefig("采样矩阵建立开销.pdf")


'''new plot 2021/11/10'''
# plt.figure(1)
#
# name_list = ['1', '2', '3', '4', '5']
# num_list1 = [0.06798827, 0.06625897, 0.05439265, 0.07704558, 0.05054526]# DLRP
# num_list2 = [0.05395416, 0.07432658, 0.05438211, 0.03759727, 0.07789838]# DLRP+
# num_list3 = [0.07142306, 0.07271974, 0.07211996, 0.07258188, 0.07142306]# SGD
# num_list4 = [0.06293935, 0.06957014, 0.06982583, 0.06016032, 0.10612155]# Momentum
# num_list5 = [0.05437866, 0.11949009, 0.07155935, 0.09815233, 0.06530363]# AdaDelta
# x = list(range(len(num_list1)))
# index = np.arange(len(name_list))
# total_width, n = 0.8, 5
# width = total_width / n
#
# plt.bar(index - 2 * width, num_list1, width=width, label='DLRP', fc='r')
# plt.bar(index - 1 * width, num_list2, width=width, label='DLRP+', fc='b')
# plt.bar(index + 0 * width, num_list3, width=width, label='SGD', fc='g')
# plt.bar(index + 1 * width, num_list4, width=width, label='Momentum', fc='purple')
# plt.bar(index + 2 * width, num_list5, width=width, label='AdaDelta', fc='orange')
#
# for a,b in zip(index - 2 * width, num_list1):   #柱子上的数字显示
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a,b in zip(index - 1 * width,num_list2):
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 0 * width, num_list3):  # 柱子上的数字显示
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 1 * width, num_list4):
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 2 * width, num_list5):
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.ylim(0.03, 0.13)
# plt.yticks(np.arange(0.03, 0.14, 0.01), )
# plt.xticks(np.arange(5), name_list)
# plt.grid(axis='y', linestyle=':')
# plt.xlabel("Experimental group number")
# plt.ylabel("Mean Square Error of Predicted Values")
# plt.legend(loc=2)
# fig = plt.gcf()
# plt.show()
# fig.savefig("TestPredictionMSE.pdf")
#
#
# plt.figure(2)
#
# name_list = ['1', '2', '3', '4', '5']
# num_list1 = [0.08505835, 0.08126006, 0.07732865, 0.08518213, 0.06873782]# DLRP
# num_list2 = [0.07267894, 0.08860825, 0.07491135, 0.06088892, 0.0899171]# DLRP+
# num_list3 = [0.08785919, 0.0884113, 0.0882322, 0.08830982, 0.08785919]# SGD
# num_list4 = [0.07982597, 0.08898777, 0.08931422, 0.08155011, 0.10646406]# Momentum
# num_list5 = [0.07945854, 0.1173716, 0.08789151, 0.10131661, 0.07894715]# AdaDelta
# x = list(range(len(num_list1)))
# index = np.arange(len(name_list))
# total_width, n = 0.8, 5
# width = total_width / n
#
# plt.bar(index - 2 * width, num_list1, width=width, label='DLRP', fc='r')
# plt.bar(index - 1 * width, num_list2, width=width, label='DLRP+', fc='b')
# plt.bar(index + 0 * width, num_list3, width=width, label='SGD', fc='g')
# plt.bar(index + 1 * width, num_list4, width=width, label='Momentum', fc='purple')
# plt.bar(index + 2 * width, num_list5, width=width, label='AdaDelta', fc='orange')
#
# for a,b in zip(index - 2 * width, num_list1):   #柱子上的数字显示
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a,b in zip(index - 1 * width,num_list2):
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 0 * width, num_list3):  # 柱子上的数字显示
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 1 * width, num_list4):
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 2 * width, num_list5):
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.ylim(0.03, 0.13)
# plt.yticks(np.arange(0.03, 0.14, 0.01))
# plt.xticks(np.arange(5), name_list)
# plt.grid(axis='y', linestyle=':')
# plt.xlabel("Experimental group number")
# plt.ylabel("Error Ratio of Predicted Values")
# plt.legend()
# fig = plt.gcf()
# plt.show()
# fig.savefig("TestPredictionMSEPercent.pdf")
#
# plt.figure(3)
#
# name_list = ['1', '2', '3', '4', '5']
# num_list1 = [0.0552, 0.0525, 0.0556, 0.0514, 0.0531]# DLRP
# num_list2 = [0.0539, 0.0522, 0.0597, 0.0567, 0.0554]# DLRP+
# num_list3 = [0.0614, 0.0611, 0.061, 0.061, 0.0614]# SGD
# num_list4 = [0.0819, 0.0641, 0.0678, 0.0688, 0.0607]# Momentum
# num_list5 = [0.0618, 0.0674, 0.0649, 0.0635, 0.063]# AdaDelta
# num_list6 = [1.68666767, 1.83143214, 2.2713839, 1.87112622, 1.90361939]# Linear
# x = list(range(len(num_list1)))
# index = np.arange(len(name_list))
# total_width, n = 0.8, 5
# width = total_width / n
#
# plt.bar(index - 2 * width, num_list1, width=width, label='DLRP', fc='r')
# plt.bar(index - 1 * width, num_list2, width=width, label='DLRP+', fc='b')
# plt.bar(index + 0 * width, num_list3, width=width, label='SGD', fc='g')
# plt.bar(index + 1 * width, num_list4, width=width, label='Momentum', fc='purple')
# plt.bar(index + 2 * width, num_list5, width=width, label='AdaDelta', fc='orange')
#
# for a,b in zip(index - 2 * width, num_list1):   #柱子上的数字显示
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a,b in zip(index - 1 * width,num_list2):
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 0 * width, num_list3):  # 柱子上的数字显示
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 1 * width, num_list4):
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 2 * width, num_list5):
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.ylim(0.03, 0.08)
# plt.yticks(np.arange(0.03, 0.09, 0.005))
# plt.xticks(np.arange(5), name_list)
# plt.grid(axis='y', linestyle=':')
# plt.xlabel("Experimental group number")
# plt.ylabel("Mean Square Error")
# plt.legend()
# fig = plt.gcf()
# plt.show()
# fig.savefig("TestRecoveryPercent.pdf")
#
# plt.figure(4)
#
# name_list = ['1', '2', '3', '4', '5']
# num_list1 = [0.0836, 0.0804, 0.0827, 0.0795, 0.0806]# DLRP
# num_list2 = [0.0813, 0.081, 0.0851, 0.0821, 0.0847]# DLRP+
# num_list3 = [0.0875, 0.0873, 0.0874, 0.0874, 0.0875]# SGD
# num_list4 = [0.1011, 0.0894, 0.0921, 0.0925, 0.0881]# Momentum
# num_list5 = [0.0888, 0.0941, 0.0919, 0.0905, 0.0889]# AdaDelta
# num_list6 = [0.47455997, 0.49150163, 0.56713293, 0.50018886, 0.49501413]# Linear
# x = list(range(len(num_list1)))
# index = np.arange(len(name_list))
# total_width, n = 0.8, 5
# width = total_width / n
#
# plt.bar(index - 2 * width, num_list1, width=width, label='DLRP', fc='r')
# plt.bar(index - 1 * width, num_list2, width=width, label='DLRP+', fc='b')
# plt.bar(index + 0 * width, num_list3, width=width, label='SGD', fc='g')
# plt.bar(index + 1 * width, num_list4, width=width, label='Momentum', fc='purple')
# plt.bar(index + 2 * width, num_list5, width=width, label='AdaDelta', fc='orange')
#
# for a,b in zip(index - 2 * width, num_list1):   #柱子上的数字显示
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a,b in zip(index - 1 * width,num_list2):
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 0 * width, num_list3):  # 柱子上的数字显示
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 1 * width, num_list4):
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 2 * width, num_list5):
#     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.ylim(0.06, 0.11)
# plt.yticks(np.arange(0.06, 0.12, 0.005))
# plt.xticks(np.arange(5), name_list)
# plt.grid(axis='y', linestyle=':')
# plt.xlabel("Experimental group number")
# plt.ylabel("Mean Square Error")
# plt.legend()
# fig = plt.gcf()
# plt.show()
# fig.savefig("TestRecoveryPercent.pdf")

plt.figure(5)

#上下绘制两张图 ax2是下面的 ax1是上面的
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=100, figsize=(6, 4.5))
fig.text(0.5, 0.03, 'Measurements', ha='center', va='center')
fig.text(0.02, 0.5, 'Compaction', ha='center', va='center', rotation='vertical')
ax2.set_ylim(0.04, 0.10)  # 子图1设置y轴范围，只显示部分图
ax2.set_yticks(np.arange(0.04, 0.11, 0.01))
ax1.set_ylim(0.2, 2.0)  # 子图2设置y轴范围，只显示部分图
ax1.set_yticks(np.arange(0.2, 2.1, 0.3))
ax2.xaxis.set_major_locator(plt.NullLocator())
ax2.xaxis.set_major_formatter(plt.NullFormatter())
ax1.spines['bottom'].set_visible(False)  # 关闭子图1中底部0
ax2.spines['top'].set_visible(False)  ##关闭子图2中顶部脊
ax1.grid(axis='y', which='major', linestyle=':')
ax2.grid(axis='y', which='major', linestyle=':')
ax2.set_xticks(np.arange(4))

name_list = ['Pred Error', 'Pred Error Ratio', 'RE Error', 'RE Error Ratio']
num_list1 = [0.063246146, 0.079513402, 0.05356, 0.08136]# DLRP
num_list2 = [0.0596317, 0.077400912, 0.05558, 0.08284]# DLRP+
num_list3 = [0.07205354, 0.08813434, 0.06118, 0.08742]# SGD
num_list4 = [0.073723438, 0.089228426, 0.06866, 0.09264]# Momentum
num_list5 = [0.081776812, 0.092997082, 0.06412, 0.09084]# AdaDelta
num_list6 = [1.36790268, 0.343154644, 1.912845864, 0.505679504]# Linear 6的下面都不要
num_list7 = [0.705498538, 0.282020154, 0.0, 0.0]# Lstm 7的上面两个不要

x = list(range(len(num_list1)))
index = np.arange(len(name_list))
total_width, n = 0.8, 7
width = total_width / n
ax1.labels = ()

ax1.bar(index - 3 * width, num_list1, width=width, label='DLRP', fc='r')
ax1.bar(index - 2 * width, num_list2, width=width, label='DLRP+', fc='b')
ax1.bar(index - 1 * width, num_list3, width=width, label='SGD', fc='g')
ax1.bar(index + 0 * width, num_list4, width=width, label='Momentum', fc='purple')
ax1.bar(index + 1 * width, num_list5, width=width, label='AdaDelta', fc='orange')
ax1.bar(index + 2 * width, num_list6, width=width, label='Linear', fc='midnightblue')
ax1.bar(index + 3 * width, num_list7, width=width, label='LSTM', fc='limegreen')

ax2.bar(index - 3 * width, num_list1, width=width, fc='r')
ax2.bar(index - 2 * width, num_list2, width=width, fc='b')
ax2.bar(index - 1 * width, num_list3, width=width, fc='g')
ax2.bar(index + 0 * width, num_list4, width=width, fc='purple')
ax2.bar(index + 1 * width, num_list5, width=width, fc='orange')
ax2.bar(index + 2 * width, num_list6, width=width, fc='midnightblue')
ax2.bar(index + 3 * width, num_list7, width=width, fc='limegreen')

'''上面图的值'''
# for a, b in zip(index - 3 * width, num_list1):   #柱子上的数字显示
#     ax1.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index - 2 * width, num_list2):   #柱子上的数字显示
#     ax1.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index - 1 * width,num_list3):
#     ax1.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 0 * width, num_list4):  # 柱子上的数字显示
#     ax1.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 1 * width, num_list5):
#     ax1.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
for a, b in zip(index + 2 * width, num_list6):
    ax1.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
for a, b in zip(index + 3 * width, num_list7):
    if a == 3+3 * width:
        continue
    if a == 2+3 * width:
        continue
    ax1.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)

''' 下面的图的 '''
for a, b in zip(index - 3 * width, num_list1):   #柱子上的数字显示
    ax2.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
for a, b in zip(index - 2 * width, num_list2):   #柱子上的数字显示
    ax2.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
for a, b in zip(index - 1 * width,num_list3):
    ax2.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
for a, b in zip(index + 0 * width, num_list4):  # 柱子上的数字显示
    ax2.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
for a, b in zip(index + 1 * width, num_list5):
    ax2.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 2 * width, num_list6):
#     if a == 3 + 2 * width:
#         continue
#     ax2.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)
# for a, b in zip(index + 3 * width, num_list7):
#     if a == 3+3 * width:
#         continue
#     if a == 2+3 * width:
#         continue
#     ax2.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


d = .85  # 设置倾斜度
# 绘制断裂处的标记
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=5,
              linestyle='none', color='r', mec='r', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
ax1.legend(loc="upper left", prop={'size': 7})
plt.xticks(np.arange(4), name_list)
plt.show()
fig.savefig("average.pdf", dpi=600, bbox_inches="tight")

# ax1.legend(loc="upper left")
# plt.savefig("5b.png",dpi=600,bbox_inches="tight")
# plt.show()
# plt.ylim(0.0, 1.5)
# plt.yticks(np.arange(0.0, 1.5, 0.3))

# plt.grid(axis='y', linestyle=':')
# plt.xlabel("Experimental group number")
# plt.ylabel("Compaction")
# plt.legend()
# fig = plt.gcf()
# plt.show()
# fig.savefig("average.pdf")
