import numpy as np


class Init():
    def __init__(self, k=20, samples=400, features=480):
        self.k = k
        self.samples = samples
        self.features = features
        self.U = np.random.uniform(0.1, 1, size=(self.samples, self.k))
        self.V = np.random.uniform(0.1, 1, size=(self.features, self.k))

        f = open('U.txt', 'w')  # 打开数据文件
        # 写入UV
        for i in range(self.samples):
            for j in range(self.k):
                f.write(str(self.U[i, j]))
                f.write('\t')
            f.write('\n')
        f.close()

        f = open('V.txt', 'w')  # 打开数据文件文件
        for i in range(self.features):
            for j in range(k):
                f.write(str(self.V[i, j]))
                f.write('\t')
            f.write('\n')
        f.close()

    def init_u(self):

        f_U = open('V.txt', 'r')
        U = np.zeros((self.samples,self.k), dtype=float)

        lines = f_U.readlines()  # 把全部数据文件读到一个列表lines中
        U_row = 0  # 表示矩阵的行，从0行开始
        for line in lines:  # 把lines中的数据逐行读取出来
            list = line.strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
            U[U_row:] = list[0:self.k]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
            U_row += 1  # 然后方阵A的下一行接着读
            # print(line)
        # print(U)
        f_U.close()
        return U

    def init_v(self):

        f_V = open('V.txt', 'r')
        V = np.zeros((self.features,self.k), dtype=float)

        lines = f_V.readlines()  # 把全部数据文件读到一个列表lines中
        V_row = 0  # 表示矩阵的行，从0行开始
        for line in lines:  # 把lines中的数据逐行读取出来
            list = line.strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
            V[V_row:] = list[0:self.k]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
            V_row += 1  # 然后方阵A的下一行接着读
            # print(line)
        # print(V)
        f_V.close()
        return V

if __name__ == '__main__':

    a = Init()
    a.init_u()
    a.init_v()
