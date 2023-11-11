import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class ActorNetwork(nn.Module):

#     def __init__(self, n_observations):
#         super(ActorNetwork, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 16)
#         self.bn1 = nn.BatchNorm1d(16)  # BatchNorm after the first layer
#         self.layer2 = nn.Linear(16, 32)
#         self.bn2 = nn.BatchNorm1d(32)  # BatchNorm after the second layer
#         self.layer3 = nn.Linear(32, 1)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.layer2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.layer3(x)
#         x = torch.tanh(x)
#         return x
    
# class CriticNetwork(nn.Module):

#     def __init__(self, state_action_stacked_shape):
#         super(CriticNetwork, self).__init__()
#         self.layer1 = nn.Linear(state_action_stacked_shape, 16)
#         self.layer2 = nn.Linear(16, 32)
#         self.layer3 = nn.Linear(32, 1)

#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         x = self.layer3(x)
#         return x


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, action_low, action_high,
                 activation_fn, bn_momentum, relu_alpha, dropout, hidden_layer_sizes):
        super(ActorNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = torch.Tensor(np.array(action_low))
        self.action_high = torch.Tensor(np.array(action_high))
        self.action_range = self.action_high - self.action_low

        self.activation_fn = activation_fn
        self.bn_momentum = bn_momentum
        self.relu_alpha = relu_alpha
        self.dropout = dropout
        self.hidden_layer_sizes = hidden_layer_sizes

        # Input layer
        self.layers = nn.ModuleList([nn.Linear(self.state_size, self.hidden_layer_sizes[0])])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.hidden_layer_sizes[0], momentum=self.bn_momentum)])
        
        # Hidden layers
        for i in range(len(self.hidden_layer_sizes)-1):
            self.layers.append(nn.Linear(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1]))
            self.bns.append(nn.BatchNorm1d(self.hidden_layer_sizes[i+1], momentum=self.bn_momentum))
        
        # Output layer
        self.out_layer = nn.Linear(self.hidden_layer_sizes[-1], self.action_size)
        
        # Dropout
        # self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.bns[i](x)
            if self.relu_alpha > 0:
                x = F.leaky_relu(x, negative_slope=self.relu_alpha)
            else:
                x = F.relu(x)
            
        # x = self.dropout_layer(x)

        if self.activation_fn == 'tanh':
            x = self.out_layer(x)
            x = torch.tanh(x)
        elif self.activation_fn == 'sigmoid':
            x = self.out_layer(x)
            x = torch.sigmoid(x)
            x = x * self.action_range + self.action_low
        else:
            raise ValueError("Expected 'activation_fn' to be one of: 'tanh', or 'sigmoid'.")

        return x


class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, relu_alpha, dropout, hidden_layer_sizes):
        super(CriticNetwork, self).__init__()

        # Check if the provided hidden_layer_sizes is a list of two lists of equal length
        assert len(hidden_layer_sizes) == 2 and len(hidden_layer_sizes[0]) == len(hidden_layer_sizes[1]), \
            "Expected hidden_layer_sizes to be a list of two arrays of equal length."

        self.state_size = state_size
        self.action_size = action_size
        self.relu_alpha = relu_alpha
        self.dropout = dropout
        self.hidden_layer_sizes = hidden_layer_sizes

        # State pathway
        self.state_layers = nn.ModuleList()
        input_size = state_size
        for size in hidden_layer_sizes[0]:
            self.state_layers.append(nn.Linear(input_size, size))
            input_size = size

        # Action pathway
        self.action_layers = nn.ModuleList()
        input_size = action_size
        for size in hidden_layer_sizes[1]:
            self.action_layers.append(nn.Linear(input_size, size))
            input_size = size

        # Combine pathways
        self.combine = nn.Linear(input_size * 2, 1)  # Multiplying by 2 since we're concatenating state & action pathways

        # if dropout > 0:
        #     self.dropout_layer = nn.Dropout(dropout)
        # else:
        #     self.dropout_layer = None

    def forward(self, states, actions):
        net_states = states.clone()  # Create a new tensor that doesn't share memory with `states`
        for layer in self.state_layers:
            net_states = F.leaky_relu(layer(net_states), negative_slope=self.relu_alpha)

        net_actions = actions.clone()  # Create a new tensor that doesn't share memory with `actions`
        for layer in self.action_layers:
            net_actions = F.leaky_relu(layer(net_actions), negative_slope=self.relu_alpha)

        # Combine state and action pathways
        net = torch.cat((net_states, net_actions), dim=1)
        # if self.dropout_layer:
        #     net = self.dropout_layer(net)
        q_value = self.combine(net)

        return q_value






