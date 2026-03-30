import random
import copy
import numpy as np
from collections import deque


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""

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


# ------------------------------------------------------------------
def step(memory, agents):
    """Sample one batch and run one learning update."""
    states, actions, rewards, next_states, dones = memory.sample()
    experiences = (states, actions, rewards, next_states, dones)

    # Both actors share the same centralised critic.
    # We update critic + actor[0] here; actor[1] is symmetrically handled
    # because both actors currently share weights (baseline simplification).
    # In the full MADDPG extension each agent will have its own learn() call.
    agents[0].critic.learn(experiences, agents[0])


# ------------------------------------------------------------------
def train_agent(agents, memory, env, action_size,
                batch_size, update_every, file_prefix,
                n_episodes=1000):

    brain_name    = env.brain_names[0]
    TOTAL_AGENTS  = 20          # Unity Reacher has 20 arms; we only control 2

    scores        = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):

        env_info   = env.reset(train_mode=True)[brain_name]
        states     = env_info.vector_observations[:2]   # (2, 33)
        state_size = states.shape[1]                    # 33
        score      = np.zeros(2)

        # Reset noise at the start of every episode
        for agent in agents:
            agent.noise.reset()

        t = 0
        while True:
            t += 1

            # --- collect actions for the 2 trained agents ---
            actions_2 = np.vstack([
                a.act(s, add_noise=True)
                for a, s in zip(agents, states)
            ])                                           # (2, 4)

            # --- pad to full 20-agent action matrix ---
            actions_full = np.zeros((TOTAL_AGENTS, action_size))
            actions_full[:2] = actions_2

            # --- step environment ---
            env_info    = env.step(actions_full)[brain_name]
            next_states = env_info.vector_observations[:2]   # (2, 33)
            rewards     = env_info.rewards[:2]               # list[2]
            dones       = env_info.local_done[:2]            # list[2]

            # --- store flat multi-agent transition ---
            # shapes stored: states (1,66), actions (1,8), rewards (1,2),
            #                next_states (1,66), dones (1,2)
            memory.add(
                states.reshape(1, state_size * 2),
                actions_2.reshape(1, action_size * 2),
                np.array(rewards, dtype=np.float32).reshape(1, 2),
                next_states.reshape(1, state_size * 2),
                np.array(dones,   dtype=np.float32).reshape(1, 2),
            )

            # --- learn ---
            if len(memory) > batch_size and t % update_every == 0:
                step(memory, agents)

            score  += rewards
            states  = next_states

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