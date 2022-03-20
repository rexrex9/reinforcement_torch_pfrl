
from chapter2.s2_dqn_reply_buffer import agents,modules,replay_buffers
import gym
import torch

class TrainManager():

    def __init__(self,
                 env,  #环境
                 episodes=1000,  #轮次数量
                 batch_size=32,  #每一批次的数量
                 num_steps=4,  #进行学习的频次
                 memory_size = 2000,  #经验回放池的容量
                 replay_start_size = 200,  #开始回放的次数
                 lr=0.001,  #学习率
                 gamma=0.9,  #收益衰减率
                 e_greed=0.1  #探索与利用中的探索概率
                 ):
        self.env = env
        self.episodes = episodes

        n_act = env.action_space.n
        n_obs = env.observation_space.shape[0]
        q_func = modules.MLP(n_obs, n_act)
        optimizer = torch.optim.AdamW(q_func.parameters(), lr=lr)
        rb = replay_buffers.ReplayBuffer(memory_size,num_steps)

        self.agent = agents.DQNAgent(
            q_func=q_func,
            optimizer=optimizer,
            replay_buffer = rb,
            batch_size=batch_size,
            replay_start_size = replay_start_size,
            n_act=n_act,
            gamma=gamma,
            e_greed=e_greed)

    # 训练一轮游戏
    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()
        while True:
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            if done: break
        return total_reward

    # 测试一轮游戏
    def test_episode(self):
        total_reward = 0
        obs = self.env.reset()
        while True:
            action = self.agent.predict(obs)
            next_obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            obs = next_obs
            self.env.render()
            if done: break
        return total_reward

    def train(self):
        for e in range(self.episodes):
            ep_reward = self.train_episode()
            print('Episode %s: reward = %.1f' % (e, ep_reward))

            if e % 100 == 0:
                test_reward = self.test_episode()
                print('test reward = %.1f' % (test_reward))


if __name__ == '__main__':
    env1 = gym.make("CartPole-v0")
    tm = TrainManager(env1)
    tm.train()

