import numpy as np


class ReplayBuffer():

    def __init__(self, memory_size: int, state_size: tuple, action_size: tuple, action_dtype=int):
        self.memory_size = memory_size
        self.counter = 0
        self.state = np.zeros((memory_size,) + state_size, dtype=np.float32)
        self.next_state = np.zeros((memory_size,) + state_size, dtype=np.float32)
        self.action = np.zeros((memory_size,1), dtype=action_dtype)
        self.reward = np.zeros((memory_size, 1), dtype=np.float32)
        self.done = np.zeros((memory_size, 1), dtype=int)

    def add_memory(self, state, next_state, action, reward, done):
        i = self.counter % self.memory_size

        self.state[i] = np.array(state) \
            if type(state) is list else state
        self.next_state[i] = np.array(next_state) \
            if type(next_state) is list else next_state
        self.action[i] = np.array(action) \
            if type(action) is list else action
        self.reward[i] = np.array(reward) \
            if type(reward) is list else reward
        self.done[i] = np.array(done) \
            if type(done) is list else done

        self.counter += 1

    # sample data
    def sample(self, size=32):
        # specify *replace to False ,prevent duplicate index
        random_index = np.random.choice(min(self.counter, self.memory_size), size, replace=False)

        state = self.state[random_index]
        next_state = self.next_state[random_index]
        action = self.action[random_index]
        reward = self.reward[random_index]
        done = self.done[random_index]

        return state, next_state, action, reward, done

    # check memory content len is greater than size len
    def ready(self, size=32):
        return self.counter > size

    def log_type(self):
        members = ['state', 'next_state', 'action', 'reward', 'done']
        fields = [(key, getattr(self, key)) for key in members if not callable(getattr(self, key))]
        print('ReplayBuffer : ')
        print(''.join(
            [f'\t[{key:>10s}] shape {val.shape.__str__():17s} type {val.dtype.__str__():>10s}\n' for key, val in
             fields]))
