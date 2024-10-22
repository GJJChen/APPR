import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.autograd import Variable
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__)) #获取当前绝对路径
rootPath = os.path.split(curPath)[0] #获取当前目录的上一级目录路径
sys.path.append(rootPath)

import util
import actions
import ReadBigData




'''  arguments   '''
column_dim = 480 #48
row_dim = 400 #20
k = 10 #10
time2update = 4
MAX_EPISODES = 1 #36 #40
MAX_EP_STEPS = 1 #120
LR = 0.005   # learning rate for Q Net
GAMMA = 0.9     # reward discount
MEMORY_CAPACITY = 1600 #600 #800
BATCH_SIZE = 400 #20
EPSILON = 0.3
fc1_hidden_dim = 4800 #480 4800
fc2_hidden_dim = 480 #240 480
fc3_hidden_dim = 48 #120 48

state_dim_x = row_dim * column_dim
state_dim_u = row_dim * k
state_dim_v = k * column_dim
state_dim = state_dim_x + state_dim_u + state_dim_v
action_dim = 3

# model_name = '2111_101801_20'
model_name = '2409_252358_1'
model_path = str(os.getcwd())+'/model/'+ model_name
time_now = time.strftime('%y%m_%d%H%M')
action_times = 1
fError = open("./actions/DRLP/Error_Record_" + str(action_times) + ".txt", 'w')
# fReward = open("./actions/DRLP/Reward_Record_" + str(action_times) + ".txt", 'w')
# fRecord = open("./actions/DRLP/Record_" + str(action_times) + ".txt", 'w')
# fLastStep = open("./actions/DRLP/LastStep_" + str(action_times) + ".txt", 'w')

X, Y = ReadBigData.readTest()




class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet,self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1) # initialization
        self.fc2 = nn.Linear(fc1_hidden_dim, fc2_hidden_dim)
        self.fc2.weight.data.normal_(0, 0.1) # initialization
        self.fc3 = nn.Linear(fc2_hidden_dim, fc3_hidden_dim)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        # self.gru = nn.GRU(fc3_hidden_dim, action_dim, batch_first=True)
        self.out = nn.Linear(fc3_hidden_dim, action_dim)
        self.out.weight.data.normal_(0, 0.1) # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        # hidden = self.init_hidden(batch_size) if hidden is None else hidden
        # # before go to RNN, reshape the input to (barch, seq, feature)
        # x = x.reshape(batch_size, max_seq, 512)
        # return self.gru(x)

        x = self.out(x)
        print(x.shape)
        return x

class DQN(object):
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + 1 + 1 + 1), dtype=np.float32)
        self.pointer = 0
        self.learn_step_counter = 0
        self.target_Net = QNet(state_dim, action_dim)
        self.eval_Net = QNet(state_dim, action_dim)
        # 可能这里也要改吧 这里的train 要改成啥呢？
        self.train = torch.optim.Adam(self.eval_Net.parameters(), lr=LR)
        self.loss_td = nn.MSELoss()

    # 根据状态得到Q值并选择动作
    def choose_action(self, s):

        s = torch.unsqueeze(torch.FloatTensor(s), 0)

        # 获得所有动作的Q值 使用贪婪法选择动作
        q = self.eval_Net(s).detach().numpy()
        max_q_action_index = np.argmax(q)
        # min_q_action_index = np.argmin(q)
        # a = np.zeros(3)

        # 选择Q值最大的动作
        if np.random.uniform() < (1 - EPSILON):
            a = max_q_action_index
        # 随机选择其余动作
        # elif np.random.uniform() < 0.5:
        else:
            a = np.random.randint(3, size=1)
            a = a[0]
        # else:
        #     a = np.ones(3)
        #     a[max_q_action_index] = 0
        #     a[min_q_action_index] = 0
        # print(a)
        return a

# def load_checkpoint(dqn):
#     # target = torch.load(model_dir_target)
#     dqn.target_Net.load_state_dict(model_dir_target)
#     dqn.eval_Net.load_state_dict(model_dir_eval)
#     memory = torch.load(model_dir_memory)
#     dqn.memory = memory['memory']
#     print('loading checkpoint!')
#     return dqn

def loadMemo(path,dqn):
    with open(path,'r') as f:
        X_row = 0  # 表示矩阵的行，从0行开始
        for i in range(MEMORY_CAPACITY):  # 把lines中的数据逐行读取出来
            line = f.readline()
            if i%100 == 0:
                print('load'+str(i))
            lis = line.split()  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
            numbers_float = list(map(float, lis))
            dqn.memory[X_row,:] = numbers_float[0:state_dim * 2 + 1 + 1 + 1]
            X_row += 1  # 然后方阵A的下一行接着读
    return dqn

dqn = DQN(state_dim)
dqn.eval_Net.load_state_dict(torch.load(model_path+'/eval.pth'))
dqn.target_Net.load_state_dict(torch.load(model_path+'/target.zip'))

dqn = loadMemo(model_path+'/memo.txt', dqn)
# dqn = DQN()
# dqn.load_state_dict(torch.load(model_dir), map_location="cpu")
# dqn = torch.load(model_dir, map_location="cpu")
correct = 0
total = 0

mf_test = actions.MF(X=X, Y=Y, action_dim=3)
error = 0
with torch.no_grad():
    for j in range(MAX_EP_STEPS):
        current_state_x, current_state_u, current_state_v = mf_test.reset()
        current_state_x = util.mynormalize(current_state_x)
        current_state_u = util.mynormalize(current_state_u)
        current_state_v = util.mynormalize(current_state_v)
        # # 擅自修改处
        # current_state_x_n = current_state_x.reshape(state_dim_x)
        # current_state_u_n = current_state_u.reshape(state_dim_u)
        # current_state_v_n = current_state_v.reshape(state_dim_v)

        current_state = np.append(current_state_x, np.append(current_state_u, current_state_v))
        current_state = current_state.reshape(state_dim)
        current_action = dqn.choose_action(current_state)


        next_state_x, next_state_u, next_state_v, reward, end_tag, error = mf_test.train(current_action, iterations=j)

        next_state_x_n = util.mynormalize(next_state_x)
        next_state_u_n = util.mynormalize(next_state_u)
        next_state_v_n = util.mynormalize(next_state_v)

        current_state_x = next_state_x
        current_state_u = next_state_u
        current_state_v = next_state_v

        fError.write(str(j) + ' recover_MSE: ' + str("%.4f" % (error[0][1])) + ' recover_MRE: ' + str(
            "%.4f" % (error[0][2])) + ' recover_MSE_y: ' + str("%.4f" % (error[0][3])) + ' recover_MRE_y: ' + str(
            "%.4f" % (error[0][4])) + '\n')

        if end_tag:
            done = 1
        else:
            done = 0

        # 擅自修改处
        print(str(j) + ' ' + str("%.4f" % (error[0][1])) + ' ' + str("%.4f" % (error[0][2])))
        if done:
            break
