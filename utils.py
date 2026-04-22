"""
utils.py  —  Training utilities for Attention-MATD3
=====================================================
Includes:
  - OUNoise
  - Checkpoint save / load  (every CKPT_INTERVAL episodes)
  - step()      — calls each agent's OWN critic.learn()
  - train_agent() — full loop with epsilon decay, rich metrics, resume support
"""

import os
import json
import copy
import random
import numpy as np
from collections import deque


# ── Checkpoint config ────────────────────────────────────────────
CKPT_DIR      = "checkpoints"
CKPT_PREFIX   = os.path.join(CKPT_DIR, "ckpt")
CKPT_META     = os.path.join(CKPT_DIR, "meta.json")
CKPT_INTERVAL = 200


# ═══════════════════════════════════════════════════════════════
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.size  = size
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


# ═══════════════════════════════════════════════════════════════
def save_checkpoint(agents, episode, scores, epsilon, file_prefix):
    os.makedirs(CKPT_DIR, exist_ok=True)
    for i, agent in enumerate(agents):
        agent.save(CKPT_PREFIX, i)
        agent.critic.save(CKPT_PREFIX, i)
    meta = {"episode": episode, "scores": scores,
            "epsilon": epsilon, "file_prefix": file_prefix}
    with open(CKPT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [Checkpoint saved @ episode {episode}]")


def load_checkpoint(agents):
    if not os.path.exists(CKPT_META):
        return 1, [], 1.0
    with open(CKPT_META) as f:
        meta = json.load(f)
    try:
        for i, agent in enumerate(agents):
            agent.load(CKPT_PREFIX, i)
            agent.critic.load(CKPT_PREFIX, i)
        ep  = meta["episode"] + 1
        sc  = meta["scores"]
        eps = meta["epsilon"]
        print(f"  [Checkpoint loaded — resuming from episode {ep}]")
        return ep, sc, eps
    except Exception as e:
        print(f"  [WARNING] Checkpoint load failed ({e}). Starting fresh.")
        return 1, [], 1.0


# ═══════════════════════════════════════════════════════════════
def step(memory, agents):
    """
    Sample one batch. Each agent runs its OWN critic.learn() call.
    This is true MATD3 — per-agent critics, not one shared critic.
    """
    states, actions, rewards, next_states, dones = memory.sample()
    experiences = (states, actions, rewards, next_states, dones)
    for idx, agent in enumerate(agents):
        agent.critic.learn(experiences, agent, agents, agent_idx=idx)


# ═══════════════════════════════════════════════════════════════
def train_agent(agents, memory, env, action_size,
                batch_size, update_every, file_prefix,
                n_episodes=1000,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.9995,
                resume=True):

    brain_name   = env.brain_names[0]
    TOTAL_AGENTS = 20

    scores        = []
    scores_window = deque(maxlen=100)

    if resume:
        start_episode, scores, epsilon = load_checkpoint(agents)
        scores_window.extend(scores[-100:])
    else:
        start_episode = 1
        epsilon       = epsilon_start

    for i_episode in range(start_episode, n_episodes + 1):
        env_info   = env.reset(train_mode=True)[brain_name]
        states     = env_info.vector_observations[:2]
        state_size = states.shape[1]
        score      = np.zeros(2)

        for agent in agents:
            agent.noise.reset()

        t = 0
        while True:
            t += 1
            actions_2 = np.vstack([
                a.act(s, add_noise=True, epsilon=epsilon)
                for a, s in zip(agents, states)
            ])

            actions_full = np.zeros((TOTAL_AGENTS, action_size))
            actions_full[:2] = actions_2

            env_info    = env.step(actions_full)[brain_name]
            next_states = env_info.vector_observations[:2]
            rewards     = env_info.rewards[:2]
            dones       = env_info.local_done[:2]

            memory.add(
                states.reshape(1, state_size * 2),
                actions_2.reshape(1, action_size * 2),
                np.array(rewards, dtype=np.float32).reshape(1, 2),
                next_states.reshape(1, state_size * 2),
                np.array(dones, dtype=np.float32).reshape(1, 2),
            )

            if len(memory) > batch_size and t % update_every == 0:
                step(memory, agents)

            score  += rewards
            states  = next_states
            if np.any(dones):
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        ep_score = np.mean(score)
        scores_window.append(ep_score)
        scores.append(ep_score)

        recent = scores[-min(100, len(scores)):]
        avg_100 = np.mean(scores_window)

        # Collect per-agent critic/actor losses
        c_losses = [f"{a.critic.critic_loss:.4f}" for a in agents]
        a_losses = [f"{a.critic.actor_loss:.4f}"  for a in agents]

        print(
            f"Ep {i_episode:4d} | Score: {ep_score:.3f}"
            f" | Avg100: {avg_100:.3f}"
            f" | Max: {max(recent):.3f} | Min: {min(recent):.3f}"
            f" | ε: {epsilon:.4f}"
            f" | C-Loss: {c_losses}"
            f" | A-Loss: {a_losses}"
        )

        if i_episode % CKPT_INTERVAL == 0:
            save_checkpoint(agents, i_episode, scores, epsilon, file_prefix)

        if avg_100 >= 30.0:
            print(f"\n✅ Solved in {i_episode} episodes! Avg: {avg_100:.3f}")
            import torch
            for i, agent in enumerate(agents):
                torch.save(agent.actor_train.state_dict(), f"{file_prefix}_actor_{i}.pth")
                torch.save(agent.critic.critic_train.state_dict(), f"{file_prefix}_critic_{i}.pth")
            break

    return scores

            if np.any(dones):
                break

        scores_window.append(np.mean(score))
        scores.append(np.mean(score))

        avg_100 = np.mean(scores_window)
        print(f"Episode {i_episode:4d} | Score: {np.mean(score):.3f}"
              f" | Avg(100): {avg_100:.3f}")

        # --- save checkpoint when solved (avg ≥ 30 is the Reacher target) ---
        if avg_100 >= 30.0:
            print(f"\n✅ Environment solved in {i_episode} episodes!"
                  f"  Avg score: {avg_100:.3f}")
            import torch
            for i, agent in enumerate(agents):
                torch.save(agent.actor_train.state_dict(),
                           f"{file_prefix}_actor_{i}.pth")
            torch.save(agents[0].critic.critic_train.state_dict(),
                       f"{file_prefix}_critic.pth")
            break

    return scores