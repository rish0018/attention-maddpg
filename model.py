import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


# ===================== ACTOR ===================== #
class ActorQNetwork(nn.Module):
    """
    Actor network: maps a single agent's state → action.
    Input:  state_size  (e.g. 33)
    Output: action_size (e.g.  4)
    """

    def __init__(self, state_size, action_size, seed, h_1_size=256, h_2_size=128):
        super(ActorQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc_1  = nn.Linear(state_size, h_1_size)
        self.fc_2  = nn.Linear(h_1_size,   h_1_size)
        self.fc_3  = nn.Linear(h_1_size,   h_2_size)
        self.output = nn.Linear(h_2_size,  action_size)
        self.reset_parameters()

    def forward(self, state):
        y = F.relu(self.fc_1(state))
        y = F.relu(self.fc_2(y))
        y = F.relu(self.fc_3(y))
        return torch.tanh(self.output(y))   # keep actions in [-1, 1]

    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_3.weight.data.uniform_(*hidden_init(self.fc_3))
        self.output.weight.data.uniform_(-3e-3, 3e-3)


# ===================== ATTENTION CRITIC ===================== #
class CriticQNetwork(nn.Module):
    """
    Attention-based Centralized Critic

    Input:
        state  = (batch, 66) → [s1, s2]
        action = (batch,  8) → [a1, a2]

    Process:
        - Split states
        - Apply attention
        - Recombine
        - Pass through MLP
    """

    def __init__(self, state_size, action_size, seed,
                 h_1_size=256, h_2_size=128, attn_dim=64):
        super(CriticQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        self.num_agents = 2
        self.single_state = state_size // 2   # 33

        # -------- ATTENTION LAYERS -------- #
        self.W_q = nn.Linear(self.single_state, attn_dim)
        self.W_k = nn.Linear(self.single_state, attn_dim)
        self.W_v = nn.Linear(self.single_state, attn_dim)

        # -------- POST-ATTENTION MLP -------- #
        self.fc_1 = nn.Linear(attn_dim * 2 + action_size, h_1_size)
        self.fc_2 = nn.Linear(h_1_size, h_2_size)
        self.fc_3 = nn.Linear(h_2_size, h_2_size)
        self.output = nn.Linear(h_2_size, 1)

        self.reset_parameters()

    def forward(self, state, action):
        # -------- SPLIT STATES -------- #
        s1 = state[:, :self.single_state]     # (batch, 33)
        s2 = state[:, self.single_state:]     # (batch, 33)

        # -------- Q, K, V -------- #
        Q1 = self.W_q(s1)
        K1 = self.W_k(s1)
        V1 = self.W_v(s1)

        Q2 = self.W_q(s2)
        K2 = self.W_k(s2)
        V2 = self.W_v(s2)

        # -------- ATTENTION SCORES -------- #
        scale = torch.sqrt(torch.tensor(Q1.size(-1), dtype=torch.float32)).to(state.device)

        score_12 = torch.sum(Q1 * K2, dim=1, keepdim=True) / scale
        score_21 = torch.sum(Q2 * K1, dim=1, keepdim=True) / scale

        # -------- SOFTMAX -------- #
        alpha_12 = torch.softmax(score_12, dim=1)
        alpha_21 = torch.softmax(score_21, dim=1)

        # -------- AGGREGATION -------- #
        h1 = V1 + alpha_12 * V2
        h2 = V2 + alpha_21 * V1

        # -------- CONCAT -------- #
        h = torch.cat((h1, h2), dim=1)        # (batch, 128)
        x = torch.cat((h, action), dim=1)     # (batch, 136)

        # -------- MLP -------- #
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))

        return self.output(x)

    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_3.weight.data.uniform_(*hidden_init(self.fc_3))
        self.output.weight.data.uniform_(-3e-3, 3e-3)