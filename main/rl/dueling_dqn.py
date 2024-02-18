import os

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import numpy as np

from main.utils.replay_buffer import ReplayBuffer

class DuelDQN(keras.Model):
    def __init__(self, layer_units:list, action_size):
        super(DuelDQN,self).__init__()

        layer_list = []

        for i,unit in enumerate(layer_units):
            layer_list.append(layers.Dense(
                units=unit,activation='relu',name=f'layer{i}'
            ))

        self.layer_list = layer_list
        #value layer
        self.V = layers.Dense(1,activation=None)
        #action layer
        self.A = layers.Dense(units=action_size, activation=None)

    def call(self,state):
        x = state

        for layer in self.layer_list:
            x = layer(x)

        A = self.A(x)
        V = self.V(x)

        Q = (V + (A - tf.reduce_mean(A,axis=1,keepdims=True)))

        return Q

    def advantage(self,state):
        x = state

        for layer in self.layer_list:
            x = layer(x)

        return self.A(x)

class Agent():
    def __init__(self, layer_list:list, gamma:float, lr:float,
                 action_size:int, state_size:tuple,
                 memory_size=3000, on_policy = False,
                 batch_size=32, target_learn_rate=100,
                 epsilon=1, epsilon_decay=0.001, min_epsilon=0.01):

        self.gamma = gamma
        self.lr = lr
        self.action_size = action_size
        self.state_size =state_size
        self.batch_size = batch_size
        self.target_learn_round = target_learn_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.on_policy = on_policy

        self.lear_counter = 0
        self.memory_buffer = ReplayBuffer(memory_size,state_size,(action_size,))

        self.eval_net = DuelDQN(layer_list,action_size)
        self.target_net = DuelDQN(layer_list,action_size)

        self.eval_net.compile(loss='mse',optimizer=optimizers.Adam(learning_rate=lr))
        self.target_net.compile(loss='mse',optimizer=optimizers.Adam(learning_rate=lr))

    def choose_action(self,state):
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = np.array([state])
            actions = self.eval_net.advantage(state)
            return np.argmax(actions[0])

    def save_model(self,dir,name):
        self.eval_net.save_weights(os.path.join(dir,f'{name} eval_net.keras'))
        self.target_net.save_weights(os.path.join(dir,f'{name} target_net.keras'))

    def load_model(self,dir,name):
        eval_path = os.path.join(dir, f'{name} eval_net.keras')
        target_path = os.path.join(dir, f'{name} target_net.keras')

        if os.path.exists(eval_path) and os.path.exists(target_path):
            self.eval_net.load_weights(eval_path)
            self.target_net.load_weights(target_path)
            print('load model ...')

    def insert_memory(self,state, next_state, action, next_action, reward, done):
        self.memory_buffer.add_memory(state, next_state, action, next_action, reward, done)

    def learn(self):
        if not self.memory_buffer.ready(self.batch_size):
            return

        if self.lear_counter % self.target_learn_round == 0:
            self.target_net.set_weights(self.eval_net.get_weights())

        self.lear_counter += 1

        state, next_state, action, next_action, reward, done = self.memory_buffer.sample(self.batch_size)

        done = np.where(done==1,0,1)

        index = np.arange(state.shape[0]).reshape((-1, 1))
        next_actions = self.target_net(next_state).numpy()
        #Q learn formula : Q(S,A) + lr * [R + gamma * Q max (S',A') - Q(S,A)]
        if self.on_policy:
            target = reward + self.gamma * next_actions[index,next_action.reshape(-1,1)] * done
        else:
            target = reward + self.gamma * np.max(next_actions,axis=1,keepdims=True) * done

        y = self.eval_net(state).numpy()
        y[index,action.reshape((-1, 1))] = target

        self.eval_net.train_on_batch(state,y)

        self.epsilon = self.epsilon - self.epsilon_decay \
            if self.epsilon > self.min_epsilon else self.min_epsilon