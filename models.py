import torch.nn as nn
import torch
from torch.distributions import Categorical, Normal

# In discrete: act_dim = number of choices/catogories
# In continuous: act_dim = number of continuous action dimensions

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

class ContinuousPolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        # log_std as a parameter, one per action dimension
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs):
        x = self.shared(obs)
        mean = self.mean_head(x) # [batch_size, act_dim]
        value = self.value_head(x).squeeze(-1) # [batch_size]
        return mean, self.log_std, value
    
    def get_action_and_value(self, obs):
        mean, log_std, value = self.forward(obs) # mean: [batch_size, act_dim], log_std: [act_dim], value: [batch_size]
        std = torch.exp(log_std)
        dist = Normal(mean, std) # dist: [batch_size, act_dim]
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value, dist
    
# if __name__ == "__main__":
#     obs_dim = 3
#     act_dim = 2
#     net = ContinuousPolicyValueNet(obs_dim, act_dim)

#     obs = torch.randn(4, obs_dim)
#     action, log_prob, value, dist = net.get_action_and_value(obs)

#     print("obs:", obs.shape)
#     print("action:", action.shape)
#     print("log_prob:", log_prob.shape)
#     print("value:", value.shape)