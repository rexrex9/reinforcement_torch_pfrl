import gym
import torch
import pfrl
import numpy
from chapter2.s5_dqn_pfrl import modules

class TrainManager():

    def __init__(self,
                env,#环境
                episodes=1000,#轮次数量
                batch_size = 32,#每一批次的数量
                num_steps=4,#进行学习的频次
                memory_size = 2000,#经验回放池的容量
                replay_start_size = 200,#开始回放的次数
                update_target_steps = 200,#同步参数的次数
                lr=0.001,#学习率
                gamma=0.9, #收益衰减率
                e_greed=0.1, #探索与利用中的探索概率
                #e_gredd_decay = 1e-6 #探索与利用中探索概率的衰减步长
                ):

        n_act = env.action_space.n
        n_obs = env.observation_space.shape[0]

        self.env = env
        self.episodes = episodes

        explorer = pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=e_greed, random_action_func=env.action_space.sample)
        q_func = modules.MLP(n_obs, n_act)
        optimizer = torch.optim.AdamW(q_func.parameters(), lr=lr)
        rb = pfrl.replay_buffers.ReplayBuffer(capacity=memory_size,num_steps=num_steps)

        self.agent = pfrl.agents.DQN(
            q_function=q_func,
            optimizer=optimizer,
            explorer=explorer,
            replay_buffer=rb,
            minibatch_size=batch_size,
            replay_start_size=replay_start_size,
            target_update_interval=update_target_steps,
            gamma=gamma,
            phi=lambda x: x.astype(numpy.float32, copy=False)
        )

    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()
        while True:
            action = self.agent.act(obs)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.agent.observe(obs, reward, done, done)
            if done: break
        return total_reward


    def test_episode(self,is_render=False):
        with self.agent.eval_mode():
            total_reward = 0
            obs = self.env.reset()
            while True:
                action = self.agent.act(obs)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                if is_render:self.env.render()
                if done: break
            return total_reward

    def train(self):
        for e in range(self.episodes):
            ep_reward = self.train_episode()
            print('Episode %s: reward = %.1f' % (e, ep_reward))
            if e % 100 == 0:
                test_reward = self.test_episode(True)
                print('test reward = %.1f' % (test_reward))

        # 进行最后的测试
        total_test_reward = 0
        for i in range(5):
            total_test_reward += self.test_episode(False)
        print('final test reward = %.1f' % (total_test_reward/5))

if __name__ == '__main__':
    env1 = gym.make("CartPole-v0")
    tm = TrainManager(env1)
    tm.train()

