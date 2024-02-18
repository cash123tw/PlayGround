import os.path

import gym
import numpy as np
from dueling_dqn import *


def train(env:gym.Env,epochs, target_learn_rate=100, max_iteration=500,model_save_rate = 50,model_save_path=None,save_name=''):
    action_size = env.action_space.n
    state_shape = env.observation_space.shape

    agent = Agent([128,128], 0.9, 0.001, action_size, state_shape, epsilon_decay=1e-5,target_learn_rate=target_learn_rate,on_policy=False)

    for epoch in range(epochs):

        iter_count = 0
        total_reward = 0
        state = env.reset()[0]
        action = agent.choose_action(state)

        while max_iteration <= 0 or iter_count < max_iteration:
            iter_count += 1

            next_state,reward,done,_,info = env.step(action)
            next_action = agent.choose_action(next_state)
            total_reward += reward

            agent.insert_memory(state,next_state,action,next_action,reward,int(done))
            agent.learn()

            state = next_state
            action = next_action

            if done or _:
                break

        print(f'epoch {epoch:>4d} play round {iter_count:>4d} rewards {int(total_reward):>4d} epsilon {agent.epsilon:>4.2f}')

        if model_save_path is not None and epoch != 0 and epoch % model_save_rate == 0:
            agent.save_model(model_save_path,name=save_name)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2',render_mode='human')

    train(env,200,model_save_path=os.path.join('../../lib/dueling_dqn'),max_iteration=0)
