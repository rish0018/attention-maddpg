"""
ddpg_agent.py  —  Attention-MATD3 Agents
=========================================
Novelty stack (on top of fixed MADDPG):
  1. Twin-critic TD target: min(Q1_target, Q2_target) — no more overestimation
  2. Target policy smoothing: Gaussian noise on next-actions used for TD target
  3. Delayed actor updates: actor updated every POLICY_DELAY critic steps
  4. Each agent gets its OWN independent TwinAttentionCritic (true MATD3)
     rather than one shared critic (which was the previous simplification)
  5. Full checkpoint save/load for all networks + optimisers
"""

import torch
import torch.optim as optim
import numpy as np

from model import TwinAttentionCritic, ActorQNetwork
from utils import OUNoise


def soft_update(local_model, target_model, tau):
    for tp, lp in zip(target_model.parameters(), local_model.parameters()):
        tp.data.copy_(tau * lp.data + (1.0 - tau) * tp.data)


# Target-policy-smoothing constants
TARGET_NOISE_SIGMA = 0.2
TARGET_NOISE_CLIP  = 0.5
POLICY_DELAY       = 2      # update actor every N critic steps


class DdpgCritic:
    """
    Per-agent TwinAttentionCritic.

    Each agent owns one of these; they see the full joint (state, action).
    Using one critic per agent (standard MATD3) is more correct than one
    shared critic because each agent may have a different reward signal.
    """

    def __init__(self, state_size, action_size, seed,
                 critic_lr, weight_decay, tau, update_every, gamma, device,
                 hidden_1_size=256, hidden_2_size=128,
                 attn_dim=64, n_heads=4, n_agents=2):

        self.state_size  = state_size
        self.action_size = action_size
        self.gamma  = gamma
        self.tau    = tau
        self.device = device
        self.n_agents = n_agents

        self.critic_train  = TwinAttentionCritic(
            state_size, action_size, seed,
            h1=hidden_1_size, h2=hidden_2_size,
            attn_dim=attn_dim, n_heads=n_heads, n_agents=n_agents
        ).to(device)

        self.critic_target = TwinAttentionCritic(
            state_size, action_size, seed,
            h1=hidden_1_size, h2=hidden_2_size,
            attn_dim=attn_dim, n_heads=n_heads, n_agents=n_agents
        ).to(device)

        soft_update(self.critic_train, self.critic_target, tau=1.0)

        self.critic_optimizer = optim.Adam(
            self.critic_train.parameters(),
            lr=critic_lr, weight_decay=weight_decay
        )

        self.actor_loss  = 0.0
        self.critic_loss = 0.0
        self._learn_step = 0       # internal counter for delayed actor update

    # ------------------------------------------------------------------
    def learn(self, experiences, actor, all_actors, agent_idx):
        """
        experiences : (states, actions, rewards, next_states, dones)
            states/next_states : (B, state_size * n_agents)
            actions            : (B, action_size * n_agents)
            rewards            : (B, n_agents)
            dones              : (B, n_agents)

        actor      : the DdpgActor whose policy we update (this critic's agent)
        all_actors : list[DdpgActor] — used to build next_actions_full
        agent_idx  : int — which agent this critic belongs to
        """
        self._learn_step += 1
        states, actions, rewards, next_states, dones = experiences

        # ── NOVEL: Target policy smoothing ────────────────────────────
        with torch.no_grad():
            next_actions_list = []
            for i, a in enumerate(all_actors):
                s_i = next_states[:, i * self.state_size:(i + 1) * self.state_size]
                next_a = a.actor_target(s_i)

                # Add clipped Gaussian noise to smooth the target policy
                noise = torch.randn_like(next_a) * TARGET_NOISE_SIGMA
                noise = noise.clamp(-TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)
                next_a = (next_a + noise).clamp(-1.0, 1.0)
                next_actions_list.append(next_a)

            next_actions_full = torch.cat(next_actions_list, dim=1)

            # ── NOVEL: Twin-critic target — take min to kill overestimation
            Q1_next, Q2_next = self.critic_target(next_states, next_actions_full)
            Q_next = torch.min(Q1_next, Q2_next)

            # Use this agent's own reward
            r    = rewards[:, agent_idx].unsqueeze(1)
            done = dones[:, agent_idx].unsqueeze(1)
            Q_target = r + self.gamma * Q_next * (1.0 - done)

        # ── Critic loss ───────────────────────────────────────────────
        Q1_exp, Q2_exp = self.critic_train(states, actions)
        import torch.nn.functional as F
        critic_loss = F.mse_loss(Q1_exp, Q_target) + F.mse_loss(Q2_exp, Q_target)
        self.critic_loss = critic_loss.item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_train.parameters(), 1.0)
        self.critic_optimizer.step()

        # ── NOVEL: Delayed actor update ───────────────────────────────
        if self._learn_step % POLICY_DELAY == 0:
            s_i = states[:, agent_idx * self.state_size:(agent_idx + 1) * self.state_size]
            actions_pred_i = actor.actor_train(s_i)

            # Substitute predicted action for this agent only
            actions_pred_full = actions.clone()
            actions_pred_full[
                :, agent_idx * self.action_size:(agent_idx + 1) * self.action_size
            ] = actions_pred_i

            actor_loss = -self.critic_train.Q1(states, actions_pred_full).mean()
            self.actor_loss = actor_loss.item()

            actor.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.actor_train.parameters(), 1.0)
            actor.actor_optimizer.step()

            soft_update(actor.actor_train, actor.actor_target, self.tau)

        soft_update(self.critic_train, self.critic_target, self.tau)

    # ------------------------------------------------------------------
    def save(self, prefix, idx):
        torch.save(self.critic_train.state_dict(),   f"{prefix}_critic{idx}_train.pth")
        torch.save(self.critic_target.state_dict(),  f"{prefix}_critic{idx}_target.pth")
        torch.save(self.critic_optimizer.state_dict(), f"{prefix}_critic{idx}_opt.pth")

    def load(self, prefix, idx):
        self.critic_train.load_state_dict(
            torch.load(f"{prefix}_critic{idx}_train.pth",  map_location=self.device))
        self.critic_target.load_state_dict(
            torch.load(f"{prefix}_critic{idx}_target.pth", map_location=self.device))
        self.critic_optimizer.load_state_dict(
            torch.load(f"{prefix}_critic{idx}_opt.pth",    map_location=self.device))


class DdpgActor:
    """
    Per-agent actor. Owns its own actor networks + OUNoise.
    Owns its own DdpgCritic (proper MATD3 — not one shared critic).
    """

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
        soft_update(self.actor_train, self.actor_target, tau=1.0)

        self.actor_optimizer = optim.Adam(
            self.actor_train.parameters(), lr=actor_lr
        )
        self.noise  = OUNoise(action_size, seed)
        self.critic = critic
        self.memory = experience_buffer

    # ------------------------------------------------------------------
    def act(self, state, add_noise=True, epsilon=1.0):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor_train.eval()
        with torch.no_grad():
            action = self.actor_train(state_t).cpu().numpy()
        self.actor_train.train()
        if add_noise:
            action += self.noise.sample() * epsilon
        return np.clip(action, -1, 1)

    # ------------------------------------------------------------------
    def save(self, prefix, idx):
        torch.save(self.actor_train.state_dict(),    f"{prefix}_actor{idx}_train.pth")
        torch.save(self.actor_target.state_dict(),   f"{prefix}_actor{idx}_target.pth")
        torch.save(self.actor_optimizer.state_dict(), f"{prefix}_actor{idx}_opt.pth")

    def load(self, prefix, idx):
        self.actor_train.load_state_dict(
            torch.load(f"{prefix}_actor{idx}_train.pth",  map_location=self.device))
        self.actor_target.load_state_dict(
            torch.load(f"{prefix}_actor{idx}_target.pth", map_location=self.device))
        self.actor_optimizer.load_state_dict(
            torch.load(f"{prefix}_actor{idx}_opt.pth",    map_location=self.device))