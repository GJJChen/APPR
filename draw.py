import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


# # 这两行代码解决 plt 中文显示的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# tag = ('SGD', 'adaDdelta', 'adaGrad', 'DRLP')
# ratio =[0.766666667, 0.333333333, 0.1, 0.666666667 ]
# # ratio = [43.56521739, 22.6, 195.3333333, 51.625]
#
# plt.bar(tag, ratio)
# plt.title('约定迭代值收敛率')
# # plt.title('收敛平均步长')
#
# plt.show()

def readData(file):
    X = []
    f = open(file)  # 打开数据文件文件
    for line in f:  # 把lines中的数据逐行读取出来
        list = line.strip('\n').strip(':').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        X.append(list)  # 把处理后的数据放到方阵A中。
    return X


if __name__ == '__main__':

    filename1 = './actions/SGD/Record_SGD_0.0025_0.0010_1.txt'
    filename2 = './actions/SGD/Record_SGD_0.0025_0.0050_1.txt'
    filename3 = './actions/SGD/Record_SGD_0.0025_0.0050_1.txt'
    # filename4 = './Record/4/Record.txt'

    data = readData(filename1)
    data2 = readData(filename2)
    data3 = readData(filename3)

    x_data = []

    y_data = []
    z_data = []
    k_data = []
    w_data = []

    y_data2 = []
    z_data2 = []
    k_data2 = []
    w_data2 = []

    y_data3 = []
    z_data3 = []
    k_data3 = []
    w_data3 = []

    y_data5 = []

    err = np.zeros(199)
    err_2 = np.zeros(199)
    err_3 = np.zeros(199)
    # err_4 = np.zeros(199)

    for i in range(200):
        x_data.append(str(i))

    lasty_1 = 0
    lasty_2 = 0
    lasty_3 = 0
    lastz_1 = 0
    lastz_2 = 0
    lastz_3 = 0
    lastk_1 = 0
    lastk_2 = 0
    lastk_3 = 0
    lastw_1 = 0
    lastw_2 = 0
    lastw_3 = 0
    for i in range(200):
        if i < len(data):
            y_data.append(float(data[i][2]))
            z_data.append(float(data[i][4]))
            k_data.append(float(data[i][6]))
            w_data.append(float(data[i][8]))
            lasty_1 = y_data[i]
            lastz_1 = z_data[i]
            lastk_1 = w_data[i]
            lastw_1 = z_data[i]
        else:
            y_data.append(lasty_1)
            z_data.append(lastz_1)
            k_data.append(lastk_1)
            w_data.append(lastw_1)


        if i < len(data2):
            y_data2.append(float(data2[i][2]))
            z_data2.append(float(data2[i][4]))
            k_data2.append(float(data2[i][6]))
            w_data2.append(float(data2[i][8]))
            lasty_2 = y_data2[i]
            lastz_2 = z_data2[i]
            lastk_2 = w_data2[i]
            lastw_2 = z_data2[i]
        else:
            y_data2.append(lasty_2)
            z_data2.append(lastz_2)
            k_data2.append(lastk_2)
            w_data2.append(lastw_2)

        if i < len(data3):
            y_data3.append(float(data3[i][2]))
            z_data3.append(float(data3[i][4]))
            k_data3.append(float(data3[i][6]))
            w_data3.append(float(data3[i][8]))
            lasty_3 = y_data3[i]
            lastz_3 = z_data3[i]
            lastk_3 = w_data3[i]
            lastw_3= z_data3[i]
        else:
            y_data3.append(lasty_3)
            z_data3.append(lastz_3)
            k_data3.append(lastk_3)
            w_data3.append(lastw_3)


    # for i in range(36):
    #     y_data5.append(float(data5[i][1]))

    # all_data = np.array(y_data) + np.array(z_data)
    # all_data2 = np.array(y_data2) + np.array(z_data2)
    # all_data3 = np.array(y_data3) + np.array(z_data3)
    # all_data4 = np.array(y_data4) + np.array(z_data4)

    x_data_2 = np.arange(199)

    for i in range(199):
        err[i] = np.abs(y_data[i + 1] - y_data[i])
        err_2[i] = np.abs(y_data2[i + 1] - y_data2[i])
        err_3[i] = np.abs(y_data3[i + 1] - y_data3[i])
        # err_4[i] = np.abs(y_data4[i + 1] - y_data4[i])

    # for i in range(200):
    #     min_1 = np.min(all_data)
    #     min_2 = np

    # ln1, = plt.plot(x_data, y_data, color='red', linewidth=2.0, linestyle='--')
    # ln2, = plt.plot(x_data, y_data2, color='blue', linewidth=2.0, linestyle='-.')
    # ln3, = plt.plot(x_data, y_data3, color='green', linewidth=2.0, linestyle='--')
    # ln4, = plt.plot(x_data, y_data4, color='purple', linewidth=2.0, linestyle='--')

    # ln1, = plt.plot(x_data, z_data, color='red', linewidth=2.0, linestyle='--')
    # ln2, = plt.plot(x_data, z_data2, color='blue', linewidth=2.0, linestyle='-.')
    # ln3, = plt.plot(x_data, z_data3, color='green', linewidth=2.0, linestyle='--')
    # ln4, = plt.plot(x_data, z_data4, color='purple', linewidth=2.0, linestyle='--')
    #
    plt.figure(1)

    # plt.xlim(-2, 220)
    # ax1 = plt.subplot(1, 2, 1)
    # ax2 = plt.subplot(1, 2, 2)

    # plt.sca(ax1)

    ln1, = plt.plot(x_data, y_data, color='red', linewidth=1.0, linestyle='--', marker='o', markevery=0.1)
    ln2, = plt.plot(x_data, y_data2, color='blue', linewidth=1.0, linestyle='-.', marker='v', markevery=0.1)
    ln3, = plt.plot(x_data, y_data3, color='green', linewidth=1.0, linestyle='--', marker='^', markevery=0.1)
    # ln4, = plt.plot(x_data, y_data4, color='purple', linewidth=1.0, linestyle=':', marker='s', markevery=10)

    plt.xlim(0, 40)
    plt.xticks(np.arange(0, 41, 2), np.arange(0, 41, 2))
    plt.legend(handles=[ln1, ln2, ln3], labels=['beta = 0.0010', 'beta = 0.0025', 'beta = 0.005'])
    # plt.xlim(-5, 300)
    plt.ylim(0, 50)
    plt.yticks(np.arange(0, 50, 5),np.arange(0, 50, 5))
    plt.grid(linestyle='-.')

    # ax1.set_title("合计平均误差")  # 设置标题及字体
    plt.xlabel("迭代次数（α=0.003,λ=0.0025）")
    plt.ylabel("合计误差")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # plt.legend(handles=[ln1, ln2, ln3, ln4], labels=['SGD', 'AadDelta', 'Monmentum', 'DLRP'])
    # plt.gcf().set_facecolor(np.ones(3) * 248 / 255)
    # plt.grid(linestyle='-.')
    fig = plt.gcf()
    plt.show()
    fig.savefig("第4组平均合计误差.pdf")

    # ln1, = plt.plot(all_data, x_data, color='red', linewidth=2.0, linestyle='--')
    # ln2, = plt.plot(all_data2, x_data, color='blue', linewidth=2.0, linestyle='-.')
    # ln3, = plt.plot(all_data3, x_data, color='green', linewidth=2.0, linestyle='--')
    # ln4, = plt.plot(all_data4, x_data, color='purple', linewidth=2.0, linestyle='--')

    # plt.figure(2)
    #
    # ln5, = plt.plot(x_data_2, err, color='red', linewidth=1.0, linestyle='--', marker='o', markevery=10)
    # ln6, = plt.plot(x_data_2, err_2, color='blue', linewidth=1.0, linestyle='-.', marker='v', markevery=10)
    # ln7, = plt.plot(x_data_2, err_3, color='green', linewidth=1.0, linestyle='--', marker='^', markevery=10)
    # # ln8, = plt.plot(x_data_2, err_4, color='purple', linewidth=1.0, linestyle=':', marker='s', markevery=10)
    #
    # # plt.title("收敛性")  # 设置标题及字体
    # plt.xlabel("迭代次数（α=0.003,λ=0.0025）")
    # plt.ylabel("误差差分")
    #
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # # my_font = fm.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
    # # plt.title("合计平均误差")  # 设置标题及字体
    # # plt.title("计算开销")
    # # plt.title("收敛性")  # 设置标题及字体
    # # plt.title("平均预测误差")  # 设置标题及字体
    # # plt.title("平均恢复误差")  # 设置标题及字体
    #
    # # plt.legend(handles=[ln1, ln2, ln3], labels=['SGD', 'AadDelta', 'AdaGrad'])
    # plt.legend(handles=[ln1, ln2, ln3], labels=['beta = 0.0010', 'beta = 0.0025', 'beta = 0.005'])
    #
    # # ax = plt.gca()
    # # ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    # # ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    # plt.ylim(0, 1)
    # plt.yticks(np.arange(0, 12, 0.5), np.arange(0, 12, 0.5))
    # plt.xlim(-3, 200)
    # plt.xticks(np.arange(0, 201, 20),np.arange(0, 201, 20))
    # # plt.yticks(np.arange(-0.02, 0.18, 0.02))
    # # plt.xlim(0.5, 3)
    # # plt.yticks(range(0, 200, 20))
    # # plt.gcf().set_facecolor(np.ones(3) * 248 / 255)
    # plt.grid(linestyle='-.')
    # fig = plt.gcf()
    # plt.show()
    # fig.savefig("第4组收敛性.pdf")

    # plt.figure(3)
    # plt.xlabel("训练次数（α=0.003,λ=0.0025）")
    # plt.ylabel("回报")
    # a = np.arange(0, 36)
    # # ln9, = plt.plot(a, y_data5, color='red', linewidth=1.0, linestyle='--', marker='o', markevery=3)
    # plt.ylim(15000, 35000)
    # plt.yticks(np.arange(0, 30, 1), np.arange(0, 30, 1))
    # plt.xticks(np.arange(0, 30, 1), np.arange(0, 30, 1))
    # # plt.gcf().set_facecolor(np.ones(3) * 248 / 255)
    # plt.grid(linestyle='-.')
    # fig = plt.gcf()
    # plt.show()
    # fig.savefig("第4组回报.pdf")
