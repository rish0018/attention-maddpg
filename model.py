"""
model.py  —  Attention-MATD3 Networks
==============================================
Novelty stack:
  1. Multi-Head Attention Critic  (vs single-head in vanilla MAAC/original)
  2. Action-Conditioned K/V — actions feed directly into Key+Value projections
     so the critic reasons about "who is doing WHAT" not just "who is where"
  3. Twin Critic heads sharing one attention trunk  (MATD3 overestimation fix)
  4. BatchNorm on actor input + critic fused representation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / math.sqrt(fan_in)
    return (-lim, lim)


# ═══════════════════════════════════════════════════════════════
#  ACTOR
# ═══════════════════════════════════════════════════════════════
class ActorQNetwork(nn.Module):
    """Deterministic actor: state -> action (single agent, decentralised)."""

    def __init__(self, state_size, action_size, seed, h1=256, h2=128):
        super().__init__()
        torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h1)
        self.fc3 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, action_size)
        self._reset()

    def forward(self, state):
        if state.shape[0] > 1:
            state = self.bn0(state)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.out(x))

    def _reset(self):
        for fc in [self.fc1, self.fc2, self.fc3]:
            fc.weight.data.uniform_(*hidden_init(fc))
        self.out.weight.data.uniform_(-3e-3, 3e-3)


# ═══════════════════════════════════════════════════════════════
#  NOVEL: Action-Conditioned Multi-Head Attention Block
# ═══════════════════════════════════════════════════════════════
class ActionConditionedMHA(nn.Module):
    """
    For each of N agents: Q from state_i, K and V from (state_i || action_i).
    Runs scaled dot-product multi-head attention across all agents.

    Key novelties vs MAAC (Iqbal & Sha 2019):
      - MAAC: state-only Q/K/V; actions enter AFTER attention
      - Here: actions condition K and V so the critic learns
        "this agent's info is relevant because of WHAT they did"
      - n_heads > 1 captures multiple coordination modes in parallel

    Input:  states  (B, N, state_size)
            actions (B, N, action_size)
    Output: context (B, N, attn_dim)
            weights (B, N, N)  — averaged over heads, for interpretability
    """

    def __init__(self, state_size, action_size, attn_dim=64, n_heads=4):
        super().__init__()
        assert attn_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = attn_dim // n_heads
        self.attn_dim = attn_dim
        self.scale    = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(state_size, attn_dim)
        self.W_k = nn.Linear(state_size + action_size, attn_dim)
        self.W_v = nn.Linear(state_size + action_size, attn_dim)
        self.W_o = nn.Linear(attn_dim, attn_dim)

    def forward(self, states, actions):
        B, N, _ = states.shape
        sa = torch.cat([states, actions], dim=-1)

        Q = self.W_q(states)   # (B, N, D)
        K = self.W_k(sa)
        V = self.W_v(sa)

        def split_heads(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)                            # (B, H, N, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, N, self.attn_dim)
        context = self.W_o(context)

        return context, weights.mean(dim=1)   # (B, N, D), (B, N, N)


# ═══════════════════════════════════════════════════════════════
#  NOVEL: Twin-Head Attention Critic  (Attention-MATD3 Critic)
# ═══════════════════════════════════════════════════════════════
class TwinAttentionCritic(nn.Module):
    """
    Centralised critic with:
      - One shared action-conditioned multi-head attention trunk
      - TWO independent MLP heads (Q1, Q2) for MATD3 twin-critic trick
        TD target = min(Q1_target, Q2_target)  to suppress overestimation

    forward() returns (q1, q2).
    Q1()     returns q1 only (used for actor loss).
    """

    def __init__(self, state_size, action_size, seed,
                 h1=256, h2=128, attn_dim=64, n_heads=4, n_agents=2):
        super().__init__()
        torch.manual_seed(seed)

        self.state_size  = state_size
        self.action_size = action_size
        self.n_agents    = n_agents

        self.attention = ActionConditionedMHA(
            state_size, action_size, attn_dim=attn_dim, n_heads=n_heads
        )

        mlp_in = attn_dim * n_agents + action_size * n_agents

        self.q1_fc1 = nn.Linear(mlp_in, h1)
        self.q1_fc2 = nn.Linear(h1, h2)
        self.q1_out = nn.Linear(h2, 1)

        self.q2_fc1 = nn.Linear(mlp_in, h1)
        self.q2_fc2 = nn.Linear(h1, h2)
        self.q2_out = nn.Linear(h2, 1)

        self.bn_fused       = nn.BatchNorm1d(mlp_in)
        self.last_attn_map  = None
        self._reset()

    def _mlp_head(self, x, fc1, fc2, out):
        return out(F.relu(fc2(F.relu(fc1(x)))))

    def forward(self, states, actions):
        B = states.shape[0]
        s = states.view(B, self.n_agents, self.state_size)
        a = actions.view(B, self.n_agents, self.action_size)

        context, attn_map = self.attention(s, a)
        self.last_attn_map = attn_map.detach()

        fused = torch.cat([context.view(B, -1), a.view(B, -1)], dim=1)
        if fused.shape[0] > 1:
            fused = self.bn_fused(fused)

        q1 = self._mlp_head(fused, self.q1_fc1, self.q1_fc2, self.q1_out)
        q2 = self._mlp_head(fused, self.q2_fc1, self.q2_fc2, self.q2_out)
        return q1, q2

    def Q1(self, states, actions):
        q1, _ = self.forward(states, actions)
        return q1

    def _reset(self):
        for fc in [self.q1_fc1, self.q1_fc2, self.q2_fc1, self.q2_fc2]:
            fc.weight.data.uniform_(*hidden_init(fc))
        for out in [self.q1_out, self.q2_out]:
            out.weight.data.uniform_(-3e-3, 3e-3)


# backward-compat alias
CriticQNetwork = TwinAttentionCritic