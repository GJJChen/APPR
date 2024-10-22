import numpy as np
import util
import ReadBigData
import sys
import os
from scipy import stats
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

curPath = os.path.abspath(os.path.dirname(__file__)) #获取当前绝对路径
rootPath = os.path.split(curPath)[0] #获取当前目录的上一级目录路径
sys.path.append(rootPath)


NOW = 0
# from variables import sgdVariables
# from variables import adaDeltaVariable
# from variables import adaGradVariable

class txt2UV():
    def __init__(self, k=10, samples=400, features=480):
        self.k = k
        self.samples = samples
        self.features = features

    def init_u(self):
        U = np.random.uniform(0.1, 1, (self.samples, self.k))
        return U

    def init_v(self):
        V = np.random.uniform(0.1, 1, (self.features, self.k))
        return V

class temp():
    def __init__(self, samples, features):
        self.b_v_u = np.zeros(samples, dtype=float)
        self.b_v_v = np.zeros(features, dtype=float)
        self.v_u_1 = 0
        self.v_v_1 = 0
        self.v_u_2 = 0
        self.v_v_2 = 0

class Momentum():
    def __init__(self):
        self.g1 = 0
        self.g2 = 0
        self.bg1 = 0
        self.bg2 = 0

class MF():

    def __init__(self, X, Y, action_dim):
        """
        参数
        - X (ndarray)   : 采样矩阵
        - k (int)       : 维度设置（秩）
        - alpha (float) : 学习步长
        - beta (float)  : 正则项参数
        """
        self.alpha = 0.0025
        self.beta = 0.0015
        self.theta = 0.0006
        self.gamma = 0.3
        self.delta = 0.7
        self.action_dim = action_dim
        self.k = 26

        # origin backpack
        self.Xor = X
        self.Xor[self.Xor == 0] = np.nan
        self.Yor = Y
        self.Yor[self.Yor == 0] = np.nan

        self.X, self.maxV, self.minV, self.boxAlpha = util.boxcox_normal(X)
        self.X[self.X == 0] = np.nan
        self.not_nan_index_X = (np.isnan(self.X) == False)

        # boxcox & normalization
        self.Y = stats.boxcox(Y[Y != -1], self.boxAlpha)
        self.Y = (self.Y[self.Y != -1] - self.minV) / (self.maxV - self.minV)
        self.Y[self.Y == 0] = np.nan
        self.not_nan_index_Y = (np.isnan(self.Y) == False)



        self.num_samples, self.num_features = X.shape
        a = txt2UV(self.k, self.num_samples, self.num_features)
        self.U = a.init_u()
        self.V = a.init_v()

        self.temp_adadelta = temp(self.num_samples , self.num_features)
        self.temp_adagrad = temp(self.num_samples , self.num_features)
        self.b_u = np.zeros(self.num_samples)
        self.b_v = np.zeros(self.num_features)

        self.b = np.mean(self.X[np.where(self.not_nan_index_X)])
        # Create a list of training samples
        #！！！救命！我才看懂这是个三元组
        self.samples = [
            (i, j, self.X[i, j])
            for i in range(self.num_samples)
            for j in range(self.num_features)
            if not np.isnan(self.X[i, j])
        ]
        self.m = Momentum()

        # True if not nan

    def reset(self):
        # return self.U.dot(self.V.T), self.U, self.V, self.b_u, self.b_v, self.temp_adadelta.v_u_1, self.temp_adadelta.v_v_1, self.m.g1, self.m.g2
        return self.U.dot(self.V.T), self.U, self.V

    def train(self, action_num, iterations):
        # Initialize factorization matrix U and V

        # gamma = 0.9
        # Initialize the biases
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        np.random.shuffle(self.samples)

        self.bias_iteration()

        if action_num == 0:
            self.sgd()
        elif action_num == 1:
            self.adadelta()
        elif action_num == 2:
            self.monentum()
        # total square error
        # se 是reward 合并se就行 整两个参数
        se, se_y, per_mean, per_mean_y, num_1 = self.square_error()
        # pdc, per_pdc, num_2 = self.perdiction_error()
        training_process.append((iterations+1, se, per_mean, se_y, per_mean_y))
        if se < self.theta:
            done = True
        else:
            done = False

        print("Action: %d ; Iteration: %d ; error = %.4f ; error_per = %.4f "
              % (action_num, iterations+1, se, per_mean))
        x_next = self.full_matrix()
        u_next = self.U
        v_next = self.V
        reward = self.gamma * se + self.delta * se_y
        return x_next, u_next, v_next, (20 - reward), done, training_process

    def square_error(self):
        """
        A function to compute the total square error
        """
        predicted = self.full_matrix()
        error = 0
        percent = 0
        num = 0

        ''' X Part '''
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if self.not_nan_index_X[i, j]:
                    if predicted[i,j] < 0:
                        val = predicted[i,j]*(-1)
                        # val = min(val,0.999)
                        val = util.deboxcox_normal(val, self.maxV, self.minV, self.boxAlpha)
                        valr = util.deboxcox_normal(self.X[i,j], self.maxV, self.minV, self.boxAlpha)
                        error += pow(abs(val+valr), 2)
                    else:
                        deboxValP = util.deboxcox_normal(predicted[i,j],self.maxV,self.minV,self.boxAlpha)
                        error += pow(abs(self.Xor[i, j] - deboxValP), 2)
                    percent += pow(self.Xor[i, j], 2)
                    #percent += pow(pow(self.X[i, j] - predicted[i, j], 2), 0.5) / self.X[i, j]
                    num += 1

        all_percent = pow(error/percent, 0.5)

        error_y = 0
        percent_y = 0
        num_y = 0

        ''' Y Part '''
        for j in range(self.num_samples):
            if self.not_nan_index_Y[j]:
                if predicted[j, self.num_features-1] < 0:
                    val = predicted[j, self.num_features-1]*(-1)
                    # val = min(val, 0.999)
                    val = util.deboxcox_normal(val, self.maxV, self.minV, self.boxAlpha)
                    valr = util.deboxcox_normal(self.Y[j], self.maxV, self.minV, self.boxAlpha)
                    error_y += pow(abs(val + valr), 2)
                else:
                    # val = min(predicted[j, self.num_features-1], 0.999)
                    deboxValP = util.deboxcox_normal(predicted[j, self.num_features-1], self.maxV, self.minV, self.boxAlpha)
                    error_y += pow(abs(self.Yor[j] - deboxValP), 2)
                percent_y += pow(self.Yor[j], 2)
                # percent_y += pow(pow(self.Y[j] - predicted[j, self.num_features-1], 2), 0.5)/self.Y[j]
                num_y += 1

        all_percent_y = pow(error_y/percent_y, 0.5)
        # if (NOW + 1) % 200 == 0:
        # print("percent is",percent)
        # print("Average percent is",(percent/num))
        print(f'Y_loss mean loss is :{error_y/num_y:10.8f}'+f'Y_loss percent is:{all_percent_y:10.8f}')
#        f_1.write('Y_loss mean loss is' + str("%.4f" % (error_y/num_y)) + ' ' + str("%.4f" % all_percent_y) + '\n')
        return error/num, error_y/num_y, all_percent, all_percent_y,num

    def bias_iteration(self):

        for i, j, x in self.samples:
            # 计算机预测与误差
            prediction = self.get_x(i, j)
            e = (x - prediction)#预测的误差

            #SGD
            # if action_num == 0:
            self.b_u[i] += self.alpha * (2 * e - self.beta * self.b_u[i])
            self.b_v[j] += self.alpha * (2 * e - self.beta * self.b_v[j])

    def sgd(self):
        """
        随机梯度下降法的过程
        """
        beta = 0.9

        for i, j, x in self.samples:
            # 计算机预测与误差
            prediction = self.get_x(i, j)
            e = (x - prediction)#预测的误差

            self.temp_adadelta.v_u_1 = \
                beta * self.temp_adadelta.v_u_1 + (1 - beta) * (2 * e * self.V[j, :] - self.beta * self.U[i, :]) ** 2
            self.temp_adadelta.v_v_1 = \
                beta * self.temp_adadelta.v_v_1 + (1 - beta) * (2 * e * self.U[i, :] - self.beta * self.V[j, :]) ** 2
            self.temp_adagrad.v_u_1 = \
                self.temp_adagrad.v_u_1 + (2 * e * self.V[j, :] - self.beta * self.U[i, :]) ** 2
            self.temp_adagrad.v_v_1 = \
                self.temp_adagrad.v_v_1 + (2 * e * self.U[i, :] - self.beta * self.V[j, :]) ** 2
            # 更新 biases
            # self.b_u[i] += self.alpha * (2 * e - self.beta * self.b_u[i])
            # self.b_v[j] += self.alpha * (2 * e - self.beta * self.b_v[j])
            self.b_u[i] += self.alpha * (2 * e - self.beta * self.b_u[i])
            self.b_v[j] += self.alpha * (2 * e - self.beta * self.b_v[j])
            # 更新因子矩阵 U 和 V
            self.U[i, :] += self.alpha * (2 * e * self.V[j, :] - self.beta * self.U[i,:])
            self.V[j, :] += self.alpha * (2 * e * self.U[i, :] - self.beta * self.V[j,:])

        indexU = np.where(self.U < 0)
        self.U[indexU] = 0

        indexV = np.where(self.V < 0)
        self.V[indexV] = 0

    def adadelta(self):

        beta = 0.9
        eps = float("1e-8")
        # v_u = self.temp_adadelta.v_u
        # v_v = self.temp_adadelta.v_v
        # v_b_u = self.temp_adadelta.v_b_u
        # v_b_v = self.temp_adadelta.v_b_v

        for i, j, x in self.samples:
            prediction = self.get_x(i, j)  # 这里的x为啥会生效？
            e = (x - prediction)

            self.temp_adadelta.v_u_1 = \
                beta * self.temp_adadelta.v_u_1 + (1 - beta) * (2 * e * self.V[j, :] - self.beta * self.U[i, :]) ** 2
            self.temp_adadelta.v_v_1 = \
                beta * self.temp_adadelta.v_v_1 + (1 - beta) * (2 * e * self.U[i, :] - self.beta * self.V[j, :]) ** 2
            self.temp_adagrad.v_u_1 = \
                self.temp_adagrad.v_u_1 + (2 * e * self.V[j, :] - self.beta * self.U[i, :]) ** 2
            self.temp_adagrad.v_v_1 = \
                self.temp_adagrad.v_v_1 + (2 * e * self.U[i, :] - self.beta * self.V[j, :]) ** 2
            # self.temp_adagrad.v_u_1 = \
            #     self.temp_adagrad.v_u_1 + (2 * e * self.V[j, :] - self.beta * self.U[i, :]) ** 2
            # self.temp_adagrad.v_v_1 = \
            #     self.temp_adagrad.v_v_1 + (2 * e * self.U[i, :] - self.beta * self.V[j, :]) ** 2
            # 改U
            self.U[i, :] += self.alpha * \
                            ((2 * e * self.V[j, :] - self.beta * self.U[i, :]) / (np.sqrt(self.temp_adadelta.v_u_1) + eps))
            # 改V
            self.V[j, :] += self.alpha * \
                            ((2 * e * self.U[i, :] - self.beta * self.V[j, :]) / (np.sqrt(self.temp_adadelta.v_v_1) + eps))

        indexU = np.where(self.U < 0)
        self.U[indexU] = 0

        indexV = np.where(self.V < 0)
        self.V[indexV] = 0

    def monentum(self):

        gamma = 0.9

        for i, j, x in self.samples:
            prediction = self.get_x(i, j)  # 这里的x为啥会生效？
            e = (x - prediction)

            self.m.g1 = gamma * self.m.g1 + self.alpha * (2 * e * self.V[j, :] - self.beta * self.U[i,:])
            self.m.g2 = gamma * self.m.g2 + self.alpha * (2 * e * self.U[i, :] - self.beta * self.V[j,:])

            self.U[i, :] += self.m.g1
            self.V[j, :] += self.m.g2

        indexU = np.where(self.U < 0)
        self.U[indexU] = 0

        indexV = np.where(self.V < 0)
        self.V[indexV] = 0

    def get_x(self, i, j):
        """
        Get the predicted x of sample i and feature j
        """
        prediction = self.b + self.b_u[i] + self.b_v[j] + self.U[i, :].dot(self.V[j, :].T)
        return prediction

    def full_matrix(self):
        """
               通过因子矩阵和偏置解释
               """
        return self.b + self.b_u[:, np.newaxis] + self.b_v[np.newaxis, :] + self.U.dot(self.V.T)

    def replace_nan(self, X_hat):
        """
               用预测值替换 nan 值
               """
        X = np.copy(self.X)
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if np.isnan(X[i, j]):
                    X[i, j] = X_hat[i, j]
        return X

    def get_U(self):
        return self.U

    def get_V(self):
        return self.V


if __name__ == '__main__':
    gamma = 0.5
    # Y = ReadBigData.readValid()[:, 479]# 这里为撒是479来着。。。

    # action_num = 0 #0-2 0:sgd 1:AdaDelta 2:Momentum
    # action_times = 1 #each for five
    action_dim = 3
    iterations = 120

    # Test
    Sample, Y = ReadBigData.readTest()

    # Train 收敛性
    # Sample = ReadBigData.readSample()
    # Y = ReadBigData.readValid()
    # Y = Y[:, 479]

    for action_num in range(3):
        for action_times in range(5):

            mf = MF(Sample, Y, action_dim)

            if action_num == 0:
                f = open("./actions/SGD/test/Record_SGD_" + str("%.4f" % mf.alpha) + str("_%.4f_" % mf.beta) + str(action_times) + ".txt", "w")
                f_result = "./actions/SGD/test/Result_SGD_" + str("%.4f" % mf.alpha) + str("_%.4f_" % mf.beta) + str(action_times) + ".txt"
            elif action_num == 1:
                f = open("./actions/AdaDelta/test/Record_AdaDelta_" + str("%.4f" % mf.alpha) + str("_%.4f_" % mf.beta) + str(action_times) + ".txt", "w")
                f_result = "./actions/AdaDelta/test/Result_AdaDelta_" + str("%.4f" % mf.alpha) + str("_%.4f_" % mf.beta) + str(action_times) + ".txt"
            else:
                f = open("./actions/Momentum/test/Record_Momentum_" + str("%.4f" % mf.alpha)+str("_%.4f_" % mf.beta)+str(action_times)+".txt", "w")
                f_result = "./actions/Momentum/test/Result_Momentum_" + str("%.4f" % mf.alpha) + str("_%.4f_" % mf.beta) + str(action_times)+ ".txt"


            sum_err_1 = 0
            sum_err_2 = 0
            err_list = []

            for i in range(iterations):
                a, b, c, d, e, g = mf.train(action_num, i)
                err_list.append((g[0][1], g[0][2], g[0][3], g[0][4]))
                # err_2_list_1.append((g[0][1] + g[0][3]) / 2)
                f.write(str(i) + ':' +
                          ' ' + str("%.6f" % (err_list[i][0])) +
                          ' ' + str("%.6f" % (err_list[i][1])) +
                          ' ' + str("%.6f" % (err_list[i][2])) +
                          ' ' + str("%.6f" % (err_list[i][3])) + '\n')

                f.flush()
                if e == True:
                    break

            predicted = mf.full_matrix()
            predicted = util.matrix_debox(predicted, mf.maxV, mf.minV, mf.boxAlpha)
            np.savetxt(f_result, np.c_[predicted], fmt='%f', delimiter='\t')

    # for i in range(iterations):
    #     a, b, c, d, e, g = mf_2.train(action_num_2, i)
    #     err_list_2.append((g[0][1], g[0][2], g[0][3], g[0][4]))
    #     # err_2_list_2.append((g[0][1] + g[0][3]) / 2)
    #     f_2.write(str(i) + ':' +
    #                 ' ' + str("%.6f" % (err_list_2[i][0])) +
    #                 ' ' + str("%.6f" % (err_list_2[i][1])) +
    #                 ' ' + str("%.6f" % (err_list_2[i][2])) +
    #                 ' ' + str("%.6f" % (err_list_2[i][3])) + '\n')
    #     f_2.flush()

    # for i in range(iterations):
    #     a, b, c, d, e, g = mf_3.train(action_num_3, i)
    #     err_list_3.append((g[0][1], g[0][2], g[0][3], g[0][4]))
    #     # err_2_list_3.append((g[0][1] + g[0][3]) / 2)
    #     f_3.write(str(i) + ':' +
    #                 ' ' + str("%.6f" % (err_list_3[i][0])) +
    #                 ' ' + str("%.6f" % (err_list_3[i][1])) +
    #                 ' ' + str("%.6f" % (err_list_3[i][2])) +
    #                 ' ' + str("%.6f" % (err_list_3[i][3])) + '\n')
    #     f_3.flush()

    # for i in range(iterations):
    #     a, b, c, d, e, g = mf_3.train(action_num_3, i)
    #     err_list_3.append((g[0][1], g[0][2], g[0][3], g[0][4]))
    #     # err_2_list_3.append((g[0][1] + g[0][3]) / 2)
    #     f_3.write(str(i + 1) + ' ' + str("%.4f" % err_list_3[i][0])
    #             + ' ' + str("%.4f" % err_list_3[i][1]) + '\n')
    #     f_3.flush()

    # average_1 = sum_err_1 / 10
    # average_2 = sum_err_2 / 10
    # f.write("Average MSE: " + str(average_1) + " Average predict_MSE: " + str(average_2) + '\n')
    # f.flush()

    # X_hat = mf_1.full_matrix()
    # X_comp = mf_1.replace_nan(X_hat)

    # X_hat = mf_2.full_matrix()
    # X_comp = mf_2.replace_nan(X_hat)

    # X_hat = mf_3.full_matrix()
    # X_comp = mf_3.replace_nan(X_hat)


    # print(X_hat)
    # print(X_comp)
#
#     mf = MF(X, k=2, alpha=0.005, beta=0.005, iterations=200, action_num=2)
#     mf.train()
#     X_hat = mf.full_matrix()
#     X_comp = mf.replace_nan(X_hat)
#
#
#     print(X_hat)
#     print(X_comp)
#     print(X)
#
#     # mf = MF(X, k=2, alpha=0.005, beta=0.005, iterations=200, action_num=4)
#     # mf.train()
#     # X_hat = mf.full_matrix()
#     # X_comp = mf.replace_nan(X_hat)
#     #
#     # print(X_hat)
#     # print(X_comp)
#     # print(X)
