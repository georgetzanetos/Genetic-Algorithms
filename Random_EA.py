import gym
import random

env = gym.make('CartPole-v0')

n_gen = 10

for gen_cnt in range(n_gen):
    step_cnt = 0
    fit = 0
    done = False
    state = env.reset()

    while not done:
        next_state, reward, done, info = env.step(env.action_space.sample())
        env.render()
        step_cnt += 1
        fit += reward
        state = next_state

    print('Generation: {}, Step count: {}, Generation reward: {}'.format(gen_cnt,step_cnt,fit))
env.close()    
