from chapter2.s1_dqn_begin import agents,modules
import gym
import torch

class TrainManager():

    def __init__(self,env,episodes=1000,lr=0.001,gamma=0.9,e_greed=0.1):
        self.env = env
        n_act = env.action_space.n
        n_obs = env.observation_space.shape[0]
        q_func = modules.MLP(n_obs, n_act)
        optimizer = torch.optim.AdamW(q_func.parameters(), lr=lr)
        self.agent = agents.DQNAgent(
            q_func=q_func,
            optimizer=optimizer,
            n_act=n_act,
            gamma=gamma,
            e_greed=e_greed)
        self.episodes = episodes

    # 训练一轮游戏
    def train_episode(self):
        total_reward = 0
        obs = self.env.reset() # 重置环境
        obs = torch.FloatTensor(obs)
        while True:
            action = self.agent.act(obs)  # 根据算法选择一个动作
            next_obs, reward, done, info = self.env.step(action) # 与环境进行一个交互
            next_obs = torch.FloatTensor(next_obs)
            self.agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            if done: break
        return total_reward

    # 测试一轮游戏
    def test_episode(self):
        total_reward = 0
        obs = self.env.reset()
        obs = torch.FloatTensor(obs)
        while True:
            action = self.agent.predict(obs)
            next_obs, reward, done, _ = self.env.step(action)
            next_obs = torch.FloatTensor(next_obs)
            obs = next_obs
            total_reward += reward
            self.env.render()
            if done:break
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