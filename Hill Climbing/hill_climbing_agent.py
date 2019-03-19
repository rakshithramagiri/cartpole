import numpy as np

from policy import POLICY


GAMMA = 0.99


class HC_AGENT:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        np.random.seed(seed)

        self.policy = POLICY(self.state_size, self.action_size, seed)

        self.rewards = list()
        self.discounts = list()
        self.rewards_count = 0
        self.reward = -np.Inf
        self.best_reward = -np.Inf
        self.best_weight = self.policy.weights
        self.noise_scale = 1e-2


    def act(self, state, deterministic=True):
        action_values = self.policy.forward(state)

        if deterministic:
            action = np.argmax(action_values)
            return action
        return np.random.choice(np.arange(self.action_size), p=action_values)


    def step(self, reward):
        self.rewards.append(reward)
        self.rewards_count += 1


    def learn(self):
        self.discounts = [GAMMA**i for i in range(self.rewards_count+1)]
        self.reward = np.sum([gamma*reward for gamma, reward in zip(self.discounts, self.rewards)])

        if self.reward >= self.best_reward:
            self.best_reward = self.reward
            self.best_weight = self.policy.weights
            self.noise_scale = max(1e-3, self.noise_scale/2)
            self.update_policy_weights()
        else:
            self.noise_scale = min(2, self.noise_scale*2)
            self.update_policy_weights(regress=True)


    def update_policy_weights(self, regress=False):
        if regress:
            self.policy.weights = self.best_weight + self.noise_scale * np.random.randn(*self.policy.weights.shape)
        else:
            self.policy.weights = self.policy.weights + self.noise_scale * np.random.randn(*self.policy.weights.shape)
