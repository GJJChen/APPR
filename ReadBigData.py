import numpy as np
import random
import copy

def readSample():
    X = np.zeros((400, 480), dtype=float)
    f = open('data/bigSample2X.txt')  # 打开数据文件文件
    X_row = 0  # 表示矩阵的行，从0行开始
    for i in range(400):  # 把lines中的数据逐行读取出来
        line = f.readline()  # 把全部数据文件读到一个列表lines中
        lis = line.split()  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        float_num = list(map(float,lis))
        X[X_row:] = float_num[0:480]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
        X_row += 1  # 然后方阵A的下一行接着读
    X = np.array(X)
    return X

def readValid():
    X = np.zeros((400, 480), dtype=float)
    f = open('data/bigX.txt')  # 打开数据文件文件
    X_row = 0  # 表示矩阵的行，从0行开始
    for i in range(400):  # 把lines中的数据逐行读取出来
        line = f.readline()  # 把全部数据文件读到一个列表lines中
        lis = line.split()  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        float_num = list(map(float, lis))
        X[X_row:] = float_num[0:480]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
        X_row += 1  # 然后方阵A的下一行接着读
    X = np.array(X)
    return X

def readTest():
    X = np.zeros((400, 480), dtype=float)
    f = open('data/bigSampleY.txt')  # 打开数据文件文件
    X_row = 0  # 表示矩阵的行，从0行开始
    for i in range(400):  # 把lines中的数据逐行读取出来
        line = f.readline()  # 把全部数据文件读到一个列表lines中
        lis = line.split()  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        float_num = list(map(float, lis))
        X[X_row:] = float_num[0:480]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
        X_row += 1  # 然后方阵A的下一行接着读
    X = np.array(X)

    X2 = np.zeros((400, 480), dtype=float)
    f = open('data/bigY.txt')  # 打开数据文件文件
    X_row = 0  # 表示矩阵的行，从0行开始
    for i in range(400):  # 把lines中的数据逐行读取出来
        line = f.readline()  # 把全部数据文件读到一个列表lines中
        lis = line.split()  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        float_num = list(map(float, lis))
        X2[X_row:] = float_num[0:480]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
        X_row += 1  # 然后方阵A的下一行接着读
    X2 = np.array(X2)

    return X, X2[:, 479]