import gym
from RL_brain import PolicyGradient
import numpy as np
import time



def preprocess(observation):
    # observation=observation[0]*0.299+observation[1]*0.587+observation[2]*0.114
    observation=np.resize(observation,[84,84,3])
    return observation

def run():
    RENDER = True  # 在屏幕上显示模拟窗口会拖慢运行速度, 我们等计算机学得差不多了再显示模拟
    DISPLAY_REWARD_THRESHOLD = 10  # 当 回合总 reward 大于 400 时显示模拟窗口
    step=0
    for i_episode in range(20000):
        observation = env.reset()
        #observation=preprocess(observation)
        while True:

            if RENDER:env.render()
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            RL.store_transition(observation, action, reward)
            #print("run")

            if done:
                ep_rs_sum = sum(RL.ep_rs)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True  # 判断是否显示模拟
                print("episode:", i_episode, "  reward:", int(running_reward))

                vt = RL.learn()  # 学习, 输出 vt, 我们下节课讲这个 vt 的作用

                # if i_episode == 0:
                #     plt.plot(vt)  # plot 这个回合的 vt
                #     plt.xlabel('episode steps')
                #     plt.ylabel('normalized state-action value')
                #     plt.show()
                break
            observation = observation_
            #step += 1



if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env = env.unwrapped
    env.seed(1)
    print(env.action_space)  # 查看这个环境中可用的 action 有多少个
    print(env.observation_space)  # 查看这个环境中可用的 state 的 observation 有多少个
    print(env.observation_space.shape[0])


    RL = PolicyGradient(n_actions=env.action_space.n,
                      n_features=[210,160,3],
                      learning_rate=0.001,reward_decay=0.99)

    run()
