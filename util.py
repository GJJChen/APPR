import numpy as np
import copy
#from sklearn import metrics
import numpy as np
#from sklearn.preprocessing import normalize
import copy
from scipy import stats, special

def readSample():
    #读取数据
    X = np.zeros((20, 48), dtype=float)
    f = open('actionSample.txt')  # 打开数据文件文件
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    X_row = 0  # 表示矩阵的行，从0行开始
    for i in range(20):  # 把lines中的数据逐行读取出来
        list = lines[i].strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        X[X_row:] = list[0:48]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
        X_row += 1  # 然后方阵A的下一行接着读

    # #归一化
    # temp = copy.deepcopy(X).T
    # temp = normalize(temp, axis=0, norm='max')
    # X = copy.deepcopy(temp).T
    return X

def readX():
    X = np.zeros((20, 48), dtype=float)
    f = open('Y5.txt')  # 打开数据文件文件 y4 train y5 test
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    X_row = 0  # 表示矩阵的行，从0行开始
    for i in range(20):  # 把lines中的数据逐行读取出来
        list = lines[i].strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        X[X_row:] = list[0:48]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
        X_row += 1  # 然后方阵A的下一行接着读
    return X

def readY():
    #读取数据
    X = np.zeros((20, 48), dtype=float)
    f = open('Y5.txt')  # 打开数据文件文件
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    X_row = 0  # 表示矩阵的行，从0行开始
    for i in range(20):  # 把lines中的数据逐行读取出来
        list = lines[i].strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        X[X_row:] = list[0:48]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
        X_row += 1  # 然后方阵A的下一行接着读
    Y = X[:, 47:]
    # #归一化
    # temp = copy.deepcopy(X).T
    # temp = normalize(temp, axis=0, norm='max')
    # X = copy.deepcopy(temp).T
    return Y

def get_data(row_dim, column_dim):
    n = np.random.rand(row_dim, column_dim)
    # for i in range(column_dim * row_dim):
    #     column = np.random.randint(0, column_dim - 1)
    #     row = np.random.randint(0, row_dim - 1)
    #     n[row, column] = 0

    data = copy.deepcopy(n)
    frist_state = copy.deepcopy(n)
    frist_state[:, -1] = 0
    return data, frist_state

#def cal_mse(x, y):
#    return metrics.mean_squared_error(x, y)

def mynormalize(a):
    tmp = copy.deepcopy(a)
    for i in range(np.shape(a)[0]):
        max_num = max(a[i,:])
        min_num = min(a[i,:])
        tmp[i, :] = (a[i, :] - min_num)/(max_num - min_num)
    return tmp

def normalize_array(a):
    max_num = max(a)
    min_num = min(a)
    tmp = (a - min_num)/(max_num - min_num)
    return tmp

def boxcox_normal(a):
    matrix = a[:, :]
    dataVector = matrix[:]
    (transfVector, alpha) = stats.boxcox(dataVector[dataVector > 0])
    maxV = np.max(transfVector)
    minV = np.min(transfVector)
    transfMatrix = matrix.copy()
    # boxcox
    transfMatrix[transfMatrix != 0] = stats.boxcox(transfMatrix[transfMatrix != 0], alpha)
    # normalization
    transfMatrix[transfMatrix != 0] = (transfMatrix[transfMatrix != 0] - minV) / (maxV - minV)
    return transfMatrix, maxV, minV, alpha


def deboxcox_normal(predVec,maxV,minV,alpha):
    # denormalization
    predVec = (maxV - minV) * predVec + minV
    # deboxcox
    predVec = argBoxcox(predVec, alpha)
    return predVec

def matrix_debox(predMat,maxV,minV,alpha):
    # denormalization
    predMat = (maxV - minV) * predMat + minV
    predMat[predMat < 0] *= -1
    # deboxcox4
    predMat = argBoxcox(predMat, alpha)
    return predMat

def argBoxcox(y, alpha):
    if alpha != 0:
        x = np.power((alpha * y + 1), (1 / alpha))
    else:
        x = np.exp(y)
    return x