import torch
import numpy as np
import torch.nn as nn


class POLICY(nn.Module):
    def __init__(self, state_space, action_space, hidden=16):
        super(Policy, self).__init__()

        self.state_space = state_space
        self.hidden = hidden

        self.fc = nn.Sequential(
            nn.Linear(self.state_space, self.hidden),
            nn.ReLU(),

            nn.Linear(self.hidden, action_space)
        )

        self.softmax = nn.Softmax(dim=1)


    def forward(self, state):
        output = self.fc(state)
        return self.softmax(output)


    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        log_probs = self.forward(state)
        action_distribution = torch.distributions.Categorical(log_probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)
