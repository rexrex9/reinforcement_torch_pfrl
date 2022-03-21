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
                ):

        n_act = env.action_space.n
        n_obs = env.observation_space.shape[0]

        self.env = env
        self.episodes = episodes

        explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=e_greed, random_action_func=env.action_space.sample)
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

    def train(self):
        pfrl.experiments.train_agent_with_evaluation(
            self.agent,
            self.env,
            steps=20000,  # Train the agent for 2000 steps
            eval_n_steps=None,  # We evaluate for episodes, not time
            eval_n_episodes=10,  # 10 episodes are sampled for each evaluation
            train_max_episode_len=200,  # Maximum length of each episode
            eval_interval=1000,  # Evaluate the agent after every 1000 steps
            outdir='result',  # Save everything to 'result' directory
        )


if __name__ == '__main__':
    env1 = gym.make("CartPole-v0")
    tm = TrainManager(env1)
    tm.train()

