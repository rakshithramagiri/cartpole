import torch
import torch.nn as nn


class DQ_NETWORK(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden=512):
        super(DQ_NETWORK, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden = hidden
        torch.manual_seed(seed)

        self.fc = nn.Sequential(
            nn.Linear(self.state_size, self.hidden),
            nn.ReLU(),

            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),

            nn.Linear(self.hidden, self.action_size)
        )

    def forward(self, state):
        output = self.fc(state)
        return output
