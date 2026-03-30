# ddpg_agent.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from model import CriticQNetwork, ActorQNetwork
from utils import OUNoise


def soft_update(local_model, target_model, tau):
    """θ_target = τ*θ_local + (1-τ)*θ_target"""
    for target_param, local_param in zip(target_model.parameters(),
                                         local_model.parameters()):
        target_param.data.copy_(
            tau * local_param.data + (1.0 - tau) * target_param.data
        )


# ================================================================
#  CENTRALISED CRITIC
#  One shared critic that sees all agents' states + actions.
#
#  Replay buffer stores:
#      states      : (batch, state_size  * 2)   e.g. (128, 66)
#      actions     : (batch, action_size * 2)   e.g. (128,  8)
#      rewards     : (batch, 2)
#      next_states : (batch, state_size  * 2)
#      dones       : (batch, 2)
# ================================================================
class DdpgCritic:

    def __init__(self, state_size, action_size, seed,
                 critic_lr, weight_decay, tau, update_every, gamma, device,
                 hidden_1_size=256, hidden_2_size=128):

        self.state_size  = state_size   # single-agent state size (33)
        self.action_size = action_size  # single-agent action size (4)
        self.gamma  = gamma
        self.tau    = tau
        self.device = device

        torch.manual_seed(seed)

        # Critic receives the full multi-agent (2-agent) concat tensors.
        # We pass these "full" sizes to CriticQNetwork so fc_1 = (66+8)=74.
        critic_state_size  = state_size  * 2   # 66
        critic_action_size = action_size * 2   #  8

        self.critic_train  = CriticQNetwork(
            critic_state_size, critic_action_size,
            seed, hidden_1_size, hidden_2_size
        ).to(device)

        self.critic_target = CriticQNetwork(
            critic_state_size, critic_action_size,
            seed, hidden_1_size, hidden_2_size
        ).to(device)

        self.critic_optimizer = optim.Adam(
            self.critic_train.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay
        )

        self.actor_loss  = 0.0
        self.critic_loss = 0.0

    # ------------------------------------------------------------------
    def learn(self, experiences, actor):
        """
        experiences : (states, actions, rewards, next_states, dones)
            states      (batch, 66)  — two agents concatenated
            actions     (batch,  8)  — two agents concatenated
            rewards     (batch,  2)
            next_states (batch, 66)
            dones       (batch,  2)
        actor : the DdpgActor whose networks we update here
        """
        states, actions, rewards, next_states, dones = experiences

        # ---- slice out a single agent's state for the actor ----
        # Agent 0 occupies the first `state_size` columns.
        states_agent0      = states[:, :self.state_size]       # (batch, 33)
        next_states_agent0 = next_states[:, :self.state_size]  # (batch, 33)

        # ---- compute target Q ----
        with torch.no_grad():
            # Target actor predicts next action for agent 0
            next_action_agent0 = actor.actor_target(next_states_agent0)  # (batch, 4)

            # Build full 2-agent next-action tensor:
            # (we mirror agent-0's action for agent-1 — a simplification
            #  acceptable for the baseline; will be replaced with proper
            #  per-agent actors in the full MADDPG extension.)
            next_actions_full = torch.cat(
                [next_action_agent0, next_action_agent0], dim=1
            )  # (batch, 8)

            Q_target_next = self.critic_target(next_states, next_actions_full)  # (batch, 1)

            # Use reward of agent 0
            Q_target = (
                rewards[:, 0].unsqueeze(1)
                + self.gamma * Q_target_next * (1 - dones[:, 0].unsqueeze(1))
            )  # (batch, 1)

        # ---- current Q estimate ----
        # `actions` already contains the real actions taken by both agents
        Q_expected = self.critic_train(states, actions)  # (batch, 1)

        # ---- critic loss & update ----
        critic_loss = F.mse_loss(Q_expected, Q_target)
        self.critic_loss = critic_loss.item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_train.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---- actor loss & update ----
        actions_pred_agent0 = actor.actor_train(states_agent0)  # (batch, 4)
        actions_pred_full   = torch.cat(
            [actions_pred_agent0, actions_pred_agent0], dim=1
        )  # (batch, 8)

        actor_loss = -self.critic_train(states, actions_pred_full).mean()
        self.actor_loss = actor_loss.item()

        actor.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.actor_train.parameters(), 1.0)
        actor.actor_optimizer.step()

        # ---- soft update both target networks ----
        soft_update(self.critic_train,     self.critic_target,    self.tau)
        soft_update(actor.actor_train,     actor.actor_target,    self.tau)


# ================================================================
#  ACTOR
#  One per trained agent.  Owns its own actor networks + noise.
#  Shares the centralised DdpgCritic with other agents.
# ================================================================
class DdpgActor:

    def __init__(self, state_size, action_size, seed, batch_size,
                 actor_lr, experience_buffer, critic,
                 weight_decay, tau, update_every, gamma, device,
                 hidden_1_size=256, hidden_2_size=128):

        self.state_size  = state_size
        self.action_size = action_size
        self.device      = device

        self.actor_train  = ActorQNetwork(
            state_size, action_size, seed, hidden_1_size, hidden_2_size
        ).to(device)

        self.actor_target = ActorQNetwork(
            state_size, action_size, seed, hidden_1_size, hidden_2_size
        ).to(device)

        self.actor_optimizer = optim.Adam(
            self.actor_train.parameters(), lr=actor_lr
        )

        self.noise  = OUNoise(action_size, seed)
        self.critic = critic
        self.memory = experience_buffer

    # ------------------------------------------------------------------
    def act(self, state, add_noise=True, epsilon=1.0):
        """
        state : numpy array of shape (state_size,)
        returns: numpy array of shape (1, action_size)
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.actor_train.eval()
        with torch.no_grad():
            action = self.actor_train(state_t).cpu().numpy()
        self.actor_train.train()

        if add_noise:
            action += self.noise.sample() * epsilon

        return np.clip(action, -1, 1)  # shape: (1, action_size)