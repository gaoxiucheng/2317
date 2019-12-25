import time

import gym
from numpy.ma import arcsin

Cart_Position_MIN = -2.4
Cart_Position_MAX = 2.4


def get_choice(observation) -> int:
    car_x = observation[0]
    pole_alpha_sin = observation[2]
    pole_top_v = observation[3]
    arcsin_alpha = arcsin(pole_alpha_sin) * 180 / 3.14
    if car_x - Cart_Position_MIN < 0.1:
        return 1
    elif Cart_Position_MAX - car_x < 0.1:
        return 0
    if arcsin_alpha < - 2:
        return 0
    elif arcsin_alpha > 2:
        return 1
    if pole_top_v < 0.0:
        return 0
    else:
        return 1


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation1 = env.reset()
        for step in range(1000):
            env.render()
            action = get_choice(observation1)
            observation1, reward, done, info = env.step(action)
            if done:
                get_choice(observation1)
                print("Episode finished after {} timesteps".format(step + 1), " ", i_episode, "reward: ", reward)
                time.sleep(1)
                break
    env.close()
    print("end")
