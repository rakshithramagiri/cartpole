import torch
import torch.nn as nn
import numpy as np

from policy import POLICY



DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class REINFORCE_AGENT:
    def __init__(self, state_space, action_space, seed=42):
        self.seed = torch.manual_seed(seed)
        self.state_space = state_space

        self.policy = POLICY(state_space, action_space, seed)


    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        log_probs = self.policy(state)
        action_distribution = torch.distributions.Categorical(log_probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)
