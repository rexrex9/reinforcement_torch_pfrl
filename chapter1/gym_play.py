import gym
import random
import gridworld

env = gym.make("CartPole-v1")
#env = gridworld.CliffWalkingWapper(env)
#CartPole-v1
state = env.reset()
while True:
    #action = random.randint(0,3)
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    env.render() #渲染一帧动画
    if done:
        break