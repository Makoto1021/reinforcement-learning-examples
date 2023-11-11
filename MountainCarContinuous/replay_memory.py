from collections import namedtuple, deque
import random



class ReplayMemory(object):
    """
    Replay Memory class that stores the transitions that the agent observes
    """

    def __init__(self, capacity, transition):
        self.memory = deque([], maxlen=capacity)
        self.transition = transition

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)