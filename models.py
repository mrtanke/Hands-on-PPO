import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical

class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs):
        x = self.shared(obs)
        logits = self.policy_head(x) # unnormalized log probabilities
        value = self.value_head(x).squeeze(-1)
        return logits, value
    
    def get_action_and_value(self, obs):
        logits, value = self.forward(obs) # logits: [batch_size, action_dim], value: [batch_size]
        dist = Categorical(logits=logits) # = softmax 
        action = dist.sample() # index of sampled action [batch_size]
        log_prob = dist.log_prob(action) # essential for computing the loss later
        return action, log_prob, value, dist
        
        # dimentions of each return parameter:
        # action: [batch_size]
        # log_prob: [batch_size]
        # value: [batch_size]
        # dist: Categorical distribution object
