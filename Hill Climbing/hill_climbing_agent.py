import numpy as np

from policy import POLICY


class HC_AGENT:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        np.random.seed(seed)

        self.policy = POLICY(self.state_size, self.action_size, seed)


    
