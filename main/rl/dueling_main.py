import os.path

import gym
import numpy as np
from dueling_dqn import *
from main.utils.plot import make_img


def train(env:gym.Env,epochs, target_learn_rate=100, max_iteration=500,model_save_rate = 50,model_save_path=None,save_name=''):
    action_size = env.action_space.n
    state_shape = env.observation_space.shape

    agent = Agent([128,128], 0.9, 0.001, action_size, state_shape,memory_size=100000,
                  batch_size=64, epsilon_decay=1e-3,target_learn_rate=target_learn_rate,on_policy=False)

    reward_history = []
    avg_reward_history = []
    epsilon_history = []

    for epoch in range(epochs):

        iter_count = 0
        total_reward = 0
        state = env.reset()[0]
        action = agent.choose_action(state)
        prevent_sand_point = False

        while max_iteration <= 0 or iter_count < max_iteration:
            iter_count += 1

            next_state,reward,done,_,info = env.step(action)
            next_action = agent.choose_action(next_state)
            total_reward += reward

            agent.insert_memory(state,next_state,action,next_action,reward,int(done))
            agent.learn()

            state = next_state
            action = next_action

            if iter_count % 1000 == 0 and agent.epsilon == agent.min_epsilon:
                agent.epsilon = 0.6
                prevent_sand_point = True

            if iter_count % 100 == 0:
                print(f'\rround {iter_count} epsilon {agent.epsilon}',end='')

            if done:
                print(f'\r',end='')
                if prevent_sand_point:
                    agent.epsilon = agent.min_epsilon
                break


        reward_history.append(total_reward)
        avg_reward_history.append(np.mean(reward_history[-100:]))
        epsilon_history.append(agent.epsilon)

        print(f'epoch {epoch:>4d} play round {iter_count:>4d} rewards {int(total_reward):>4d} avg rewards {int(avg_reward_history[-1]):>4d} epsilon {agent.epsilon:>4.2f}')

        if model_save_path is not None and epoch != 0 and epoch % model_save_rate == 0:
            agent.save_model(model_save_path,name=save_name)

        #save track img
        if epoch > 0 and epochs//5 and epoch % (epochs//5) == 0:
            make_img('Track', 'avg_reward', 'epsilon', avg_reward_history, epsilon_history,
                     show=False, save_path=save_path,save_name=f'{env.unwrapped.spec.id}_{epoch}.jpg')

    return reward_history,avg_reward_history,epsilon_history

if __name__ == '__main__':
    game_name = 'LunarLander-v2'
    save_path = '../../lib/dueling_dqn'

    env = gym.make(game_name, render_mode='human')

    history = train(env, 500, model_save_path=os.path.join(save_path), max_iteration=0)

    make_img('Track','avg_reward','epsilon',history[1],history[2],show = True,save_path=save_path,save_name=f'{game_name}.jpg')
