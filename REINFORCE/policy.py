import torch
import numpy as np
import torch.nn as nn


class POLICY(nn.Module):
    def __init__(self, state_space, action_space, seed, hidden=16):
        super(Policy, self).__init__()

        self.state_space = state_space
        self.hidden = hidden
        torch.manual_seed(seed)

        self.fc = nn.Sequential(
            nn.Linear(self.state_space, self.hidden),
            nn.ReLU(),

            nn.Linear(self.hidden, action_space)
        )

        self.softmax = nn.Softmax(dim=1)


    def forward(self, state):
        output = self.fc(state)
        return self.softmax(output)
