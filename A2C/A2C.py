import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch

class A2C(nn.Module):
    """
    Network to predict both actor policy and critic values
    """
    def __init__(self, state_size, n_actions):
        super(A2C, self).__init__()
        # Actor Layers
        self.act1 = nn.Linear(state_size, 32)
        self.act2 = nn.Linear(32, 64)
        self.act3 = nn.Linear(64, n_actions)
        # Critic Layers
        self.crit1 = nn.Linear(state_size, 32)
        self.crit2 = nn.Linear(32, 64)
        self.crit3 = nn.Linear(64, 1)


    def forward(self, input):
        """
        Method to pass an input through both the actor and critic networks
        ::Params::
            input (Tensor): Input tensor to the networks, representing the current state
        ::Output::
            Policy (Tensor of shape (n_actions)): Probability distribution over actions
            Value: (Tensor of shape (1)): Predicted value of state
        """
        # Actor prediction
        policy = F.relu(self.act1(input))
        policy = F.relu(self.act2(policy))
        policy = F.softmax(self.act3(policy))

        # Critic prediction
        value = F.relu(self.crit1(input))
        value = F.relu(self.crit2(value))
        value = self.crit3(value)
        return policy, value
