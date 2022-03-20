import torch
from utils import torchUtils
import copy

class DQNAgent(object):

    def __init__(self,q_func, optimizer, explorer,replay_buffer, batch_size, replay_start_size,update_target_steps, n_act, gamma=0.9):
        '''
        :param q_func:  Q函数
        :param optimizer: 优化器
        :param explorer: 探索器
        :param replay_buffer: 经验回放器
        :param batch_size: 批次数量
        :param replay_start_size: 开始回放的次数
        :param update_target_steps: 同步参数的次数
        :param n_act: 动作数量
        :param gamma: 收益衰减率
        '''
        self.pred_func = q_func
        self.target_func = copy.deepcopy(q_func)
        self.update_target_steps = update_target_steps

        self.explorer = explorer

        self.rb = replay_buffer
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size

        self.optimizer = optimizer
        self.criterion = torch.nn.MSELoss()

        self.global_step = 0
        self.gamma = gamma  # 收益衰减率
        self.n_act = n_act # 动作数量

    # 根据经验得到action
    def predict(self, obs):
        obs = torch.FloatTensor(obs)
        Q_list = self.pred_func(obs)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    # 根据探索与利用得到action
    def act(self, obs):
        return self.explorer.act(self.predict,obs)

    def learn_batch(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done):
        # predict_Q
        pred_Vs = self.pred_func(batch_obs)
        action_onehot = torchUtils.one_hot(batch_action, self.n_act)
        predict_Q = (pred_Vs * action_onehot).sum(1)
        # target_Q
        next_pred_Vs = self.target_func(batch_next_obs)
        best_V = next_pred_Vs.max(1)[0]
        target_Q = batch_reward + (1 - batch_done) * self.gamma * best_V

        self.optimizer.zero_grad()  # 梯度归0
        loss = self.criterion(predict_Q, target_Q)
        loss.backward()
        self.optimizer.step()

    def learn(self, obs, action, reward, next_obs, done):
        self.global_step += 1
        self.rb.append((obs, action, reward, next_obs, done))
        if len(self.rb) > self.replay_start_size and self.global_step % self.rb.num_steps == 0:
            self.learn_batch(*self.rb.sample(self.batch_size))
        if self.global_step % self.update_target_steps==0:
            self.sync_target()

    def sync_target(self):
        for target_param, param in zip(self.target_func.parameters(), self.pred_func.parameters()):
            target_param.data.copy_(param.data)
