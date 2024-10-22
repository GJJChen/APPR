# DQN Implement #
# Use Linear in Q Net #
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.autograd import Variable
import util
import actions
import ReadBigData
import joblib
import sys
curPath = os.path.abspath(os.path.dirname(__file__)) #获取当前绝对路径
rootPath = os.path.split(curPath)[0] #获取当前目录的上一级目录路径
sys.path.append(rootPath)

#####################  hyper parameters  ####################

column_dim = 480 #48
row_dim = 400 #20
k = 26 #10
time2update = 4
MAX_EPISODES = 15 #36 #40
MAX_EP_STEPS = 200 #120
LR = 0.005   # learning rate for Q Net
GAMMA = 0.9     # reward discount
MEMORY_CAPACITY = 1600 #600 #800
BATCH_SIZE = 400 #20
EPSILON = 0.3
fc1_hidden_dim = 4800 #480 4800
fc2_hidden_dim = 480 #240 480
fc3_hidden_dim = 48 #120 48

action_times = 1
fError = open("./actions/DRLP/Error_Record_"+str(action_times)+".txt", 'w')
fReward = open("./actions/DRLP/Reward_Record_"+str(action_times)+".txt", 'w')
fRecord = open("./actions/DRLP/Record_"+str(action_times)+".txt", 'w')
fLastStep = open("./actions/DRLP/LastStep_"+str(action_times)+".txt", 'w')


###############################  DQN  ####################################
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1) # initialization
        self.fc2 = nn.Linear(fc1_hidden_dim, fc2_hidden_dim)
        self.fc2.weight.data.normal_(0, 0.1) # initialization
        self.fc3 = nn.Linear(fc2_hidden_dim, fc3_hidden_dim)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(fc3_hidden_dim, action_dim)
        self.out.weight.data.normal_(0, 0.1) # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
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
        #可能这里也要改吧 这里的train 要改成啥呢？
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
        else:
            a = np.random.randint(3,size=1)
            a = a[0]
        return a

    # 训练Q网络
    def learn(self):

        if self.learn_step_counter % time2update == 0:
            self.target_Net.load_state_dict(self.eval_Net.state_dict())
        self.learn_step_counter += 1

        # 从经验回放池中采样BATCH_SIZE个样本
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        sample = self.memory[indices, :]
        state = torch.FloatTensor(sample[:, :self.state_dim])
        action = torch.LongTensor(sample[:, self.state_dim: self.state_dim + 1].astype(int))
        reward = torch.FloatTensor(sample[:, -self.state_dim - 1: -self.state_dim])
        next_state = torch.FloatTensor(sample[:, -self.state_dim - 1: -1])
        is_end = sample[:, -1: ]

        # 使用Q网络计算当前目标Q值 Q(next_state, next_action, w)

        q_eval = self.eval_Net(state).gather(1, action)
        q_next = self.target_Net(next_state).detach()
        # print(q_next)
        #这一步是在干什么
        q_target = torch.FloatTensor(np.zeros((BATCH_SIZE, 1)))

        #这块也不是很懂 好像是照着公式更新Q_target网络
        for i in range(BATCH_SIZE):
            if is_end[i,:] == 1:
                q_target[i, :] = reward[i, :]
            else:
                max_index = np.argmax(q_next[i, :])
                q_target[i, :] = reward[i, :] + GAMMA * q_next[i, max_index]

        # # 使用Q网络计算 Q(state, action, w)
        # q_temp = self.eval_Net(state).detach()
        # q_v = torch.FloatTensor(np.zeros((BATCH_SIZE, 1)))
        # for i in range(BATCH_SIZE):
        #     action_index = np.argmax(action.detach().numpy())#转为ndarray
        #     q_v[i, :] = q_tmp[i, action_index]

        td_error = self.loss_td(q_eval, q_target)
        # td_error = Variable(td_error, requires_grad=True)
        self.train.zero_grad()
        td_error.backward()
        self.train.step()

    def store_transition(self, state, action, reward, next_state, done):
        next_state = next_state.reshape(state_dim)
        transition = np.hstack((state, [action], [reward], next_state, [done]))
        #更新经验回放池
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

def saveMemo(path, memo):
    np.savetxt(path, np.c_[memo], fmt='%f', delimiter=' ')
    # with open(path, 'w') as f:
    #     for i in range(memo.shape[0]):
    #         if i%5 ==0:
    #             print('writing :'+str(i))
    #         for j in range(memo.shape[1]):
    #             f.write(str(memo[i][j])+' ')
    #         f.write('\n')
    #         # np.zeros((MEMORY_CAPACITY, state_dim * 2 + 1 + 1 + 1), dtype=np.float32)
    #         f.flush()
    # f.close()

###############################  training  ####################################

# state_dim = row_dim * column_dim
#擅自修改处
state_dim_x = row_dim * column_dim
state_dim_u = row_dim * k
state_dim_v = k * column_dim
state_dim = state_dim_x + state_dim_u + state_dim_v
action_dim = 3
# action_bound = 2**action_dim


model_name = '2409_240410_1'
model_path = str(os.getcwd())+'/model/'+ model_name
dqn = DQN(state_dim)

'''加载之前训练好的模型 接着训练'''
# dqn.target_Net.load_state_dict(torch.load(model_path+'/target.pt'))
# dqn.eval_Net.load_state_dict(torch.load(model_path+'/target.pt'))
# dqn = loadMemo(model_path+'/memo.txt', dqn)



# original_data = util.readX()# 这句没有被用到。。。
# original_y = util.readY()
# sample_data = util.readSample()# 用的是这句
original_data = ReadBigData.readValid()# 这句没有被用到。。。
original_y = original_data[:, 479]#这个需要检查一下之前的readY 是只提供最后一列的数据
np.reshape(original_y, (400, 1))
sample_data = ReadBigData.readSample()
# min_recover_MSE = 0.35
# min_predict_MSE = 0.47
is_first = 0
# record = np.ones(300)
satisfy_tag = 0
t1 = time.time()
step_size = 10

#开始训练
for i in range(MAX_EPISODES):

    mf = actions.MF(sample_data, original_y, 3)
    #擅自修改处

    current_state_x, current_state_u, current_state_v = mf.reset()

    all_reward = 0
    for j in range(MAX_EP_STEPS):

        # current_state = util.mynormalize(current_state)
        #擅自修改处

        # 之前已经normalize过了这里应该不再需要normalize
        # current_state_x = util.mynormalize(current_state_x)
        # current_state_u = util.mynormalize(current_state_u)
        # current_state_v = util.mynormalize(current_state_v)

        # 擅自修改处
        current_state_x_n = current_state_x.reshape(state_dim_x)
        current_state_u_n = current_state_u.reshape(state_dim_u)
        current_state_v_n = current_state_v.reshape(state_dim_v)

        current_state = np.append(current_state_x, np.append(current_state_u, current_state_v))


        current_action = dqn.choose_action(current_state)

        next_state_x, next_state_u, next_state_v, reward, end_tag, error = mf.train(current_action, j)

        if end_tag:
            done = 1
        else:
            done = 0
        fError.write(str(i) + ':' +str(j)+ str("%.4f" % (error[0][1])) + str("%.4f" % (error[0][2])) +' '+str("%.4f"%(error[0][3]))+' '+str("%.4f"% (error[0][4]))+'\n')
        fError.flush()
        #擅自修改处
        # next_state_x_n = util.mynormalize(next_state_x)
        # next_state_u_n = util.mynormalize(next_state_u)
        # next_state_v_n = util.mynormalize(next_state_v)
        next_state_x_n = next_state_x
        next_state_u_n = next_state_u
        next_state_v_n = next_state_v

        next_state_n = np.append(next_state_x_n, np.append(next_state_u_n, next_state_v_n))
        # 将结果放入经验回放池
        dqn.store_transition(current_state, current_action, reward, next_state_n, done)
        if dqn.pointer > MEMORY_CAPACITY:
            # control exploration
            EPSILON *= .999    # decay the action randomness
            dqn.learn()

        #擅自修改处
        all_reward += reward
        current_state_x = next_state_x
        current_state_u = next_state_u
        current_state_v = next_state_v


        # 每一轮迭代输出学习结果
        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Reward: %i' % int(all_reward), 'Explore: %.2f' % EPSILON)
            fReward.write(str(i) + ' ' + str(all_reward) + ' ' + str(EPSILON)+'\n')
            fLastStep.write(str(i) + ':' + str(j) + ' ' + str("%.4f" % (error[0][1])) + ' ' + str("%.4f" % (error[0][2])) +' '+str("%.4f"%(error[0][3]))+' '+str("%.4f"% (error[0][4]))+ '\n')
            fReward.flush()
            fLastStep.flush()

        if i == MAX_EPISODES - 1:
            fRecord.write(str(i) + ':' + str(j) + ' ' + str("%.4f" % (error[0][1])) + ' ' + str("%.4f" % (error[0][2])) +' '+str("%.4f"%(error[0][3]))+' '+str("%.4f"% (error[0][4]))+'\n')
            fRecord.flush()

        if done:
            break


    if ((i+1) % 10 == 0):
        time_now = time.strftime('%y%m_%d%H%M')
        model_file_dir = str(str(os.getcwd()) + '/model' + '/{}_{}'.format(time_now, 1))
        os.mkdir(model_file_dir)
        torch.save(dqn.target_Net.state_dict(), model_file_dir + '/target.pt')
        torch.save(dqn.eval_Net.state_dict(), model_file_dir + '/eval.zip')
        saveMemo(model_file_dir + '/memo.txt', dqn.memory)
        predicted = mf.full_matrix()
        predicted = util.matrix_debox(predicted, mf.maxV, mf.minV, mf.boxAlpha)
        np.savetxt(model_file_dir + '/predicted.txt', np.c_[predicted], fmt='%f', delimiter='\t')


print('Running time: ', time.time() - t1)
