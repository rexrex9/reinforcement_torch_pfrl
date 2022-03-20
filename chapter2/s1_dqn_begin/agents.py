import numpy as np
import torch

class DQNAgent(object):

    def __init__(self, q_func, optimizer, n_act, gamma=0.9, e_greed=0.1):
        self.q_func = q_func #Q函数

        self.optimizer = optimizer #优化器
        self.criterion = torch.nn.MSELoss() #损失函数

        self.n_act = n_act  # 动作数量
        self.gamma = gamma  # 收益衰减率
        self.epsilon = e_greed  # 探索与利用中的探索概率

    # 根据经验得到action
    def predict(self, obs):
        Q_list = self.q_func(obs)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    # 根据探索与利用得到action
    def act(self, obs):
        if np.random.uniform(0, 1) < self.epsilon:  #探索
            action = np.random.choice(self.n_act)
        else: # 利用
            action = self.predict(obs)
        return action

    # 更新Q表格
    def learn(self, obs, action, reward, next_obs, done):
        # predict_Q
        pred_Vs = self.q_func(obs)
        predict_Q = pred_Vs[action]

        #target_Q
        next_pred_Vs = self.q_func(next_obs)
        best_V = next_pred_Vs.max()
        target_Q = reward + (1 - float(done))*self.gamma * best_V

        # 更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(predict_Q,target_Q)
        loss.backward()
        self.optimizer.step()
