import warnings
import numpy
import numpy as np
import gym
import time
from numba import autojit

ENV_NAME = 'CartPole-v1'
warnings.filterwarnings('ignore')


class AGENT:
    def __init__(self):
        np.random.seed(1)
        self.env = gym.make(ENV_NAME)
        self.best_reward = 0.0
        self.factors = np.random.rand(5)
        print("初始值", self.factors)

    def train(self):            #爬山算法
        for _ in range(10000):
            self.__hill_climbing()
            if self.best_reward >= 499:
                break

        print(self.best_reward, self.factors)
        return self.best_reward, self.factors

    def get_action(self, observation1) -> int:          #在训练后调用此函数 得到决策
        return self.__get_action(observation1, factors=self.factors)

    @autojit
    def __get_action(self, observation, factors: numpy.ndarray) -> int:         #用线性方程 y = ax + by + cz + dw + e 计算结果根据 y 的符号做出决策
        y = factors[0] * observation[0] + factors[1] * observation[1] + factors[2] * observation[2] \
            + factors[3] * observation[3] + factors[4]
        if y >= 0.0:
            return 1
        else:
            return 0
    def __get_avg_reward_by_factors(self, factors: numpy.ndarray, show=False) -> float:         #训练过程中尝试新的参数时调用本方法以得到 10 次 测试得分的均值
        sum_reward = 0.0
        for _ in range(10):
            _observation1 = self.env.reset()
            for t in range(1000):
                if show:
                    time.sleep(0.01)
                    self.env.render()
                _action = self.__get_action(_observation1, factors=factors)
                _observation1, _reward, _done, _info = self.env.step(_action)
                sum_reward += _reward
                if _done:
                    break
        return sum_reward / 10.0

    def __random_walk(self):            #给现有的参数加上一个随机偏移 返回随机游走后的参数
        return self.factors + np.random.normal(0, 0.2, 5)

    def __hill_climbing(self):
        cur_factors = self.__random_walk()
        cur_sum_reward = self.__get_avg_reward_by_factors(cur_factors)

        if cur_sum_reward > self.best_reward:
            self.best_reward = cur_sum_reward
            self.factors = cur_factors
            print(self.best_reward)


if __name__ == '__main__':
    agent = AGENT()
    agent.train()
    env = gym.make(ENV_NAME)
    for i in range(3):
        obs = env.reset()
        sum_reward = 0.0
        while True:
            time.sleep(0.01)
            env.render()
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            sum_reward += reward
            if done:
                print(sum_reward)
                break
    env.close()
    print("end")
