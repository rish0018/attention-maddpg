# main.py  —  Attention-MATD3 entry point

import argparse
import numpy as np
import torch
import sys
import os

from ddpg_agent import DdpgActor, DdpgCritic
from replaybuffers import ReplayBuffer
from utils import train_agent

try:
    from unityagents import UnityEnvironment
    HAS_UNITY = True
except ImportError:
    HAS_UNITY = False
    print("Warning: unityagents not found. pip install unityagents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file',              default="maddpg_env/Reacher.exe")
    parser.add_argument('--mode',   choices=["train", "test"], default="train")
    parser.add_argument('--episodes', type=int,    default=2000)
    parser.add_argument('--resume',  action='store_true',
                        help="Resume from checkpoints/")
    parser.add_argument('--output_weights_prefix', default="model")

    args = parser.parse_args()

    train_mode = args.mode == "train"

    if not HAS_UNITY:
        print("Error: install unityagents first."); sys.exit(1)

    if not os.path.exists(args.env_file):
        print(f"Error: env not found at {args.env_file}"); sys.exit(1)

    env        = UnityEnvironment(file_name=args.env_file, no_graphics=not train_mode)
    brain_name = env.brain_names[0]
    brain      = env.brains[brain_name]

    env_info   = env.reset(train_mode=train_mode)[brain_name]
    print("Agents in env :", len(env_info.agents))

    action_size = brain.vector_action_space_size
    state_size  = len(env_info.vector_observations[0])
    print(f"State: {state_size}  Action: {action_size}")

    # ── Hyperparameters ──────────────────────────────────────────
    BUFFER_SIZE  = int(1e5)
    BATCH_SIZE   = 128
    GAMMA        = 0.99
    TAU          = 1e-3
    LR_ACTOR     = 1e-4
    LR_CRITIC    = 1e-3
    UPDATE_EVERY = 1
    EPS_START    = 1.0
    EPS_END      = 0.01
    EPS_DECAY    = 0.9995
    N_AGENTS     = 2
    N_HEADS      = 4
    ATTN_DIM     = 64

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Shared replay buffer ─────────────────────────────────────
    memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed=0, device=device)

    # ── Per-agent critics (MATD3: each agent gets its own) ───────
    critics = [
        DdpgCritic(
            state_size=state_size, action_size=action_size, seed=i,
            critic_lr=LR_CRITIC, weight_decay=0, tau=TAU,
            update_every=UPDATE_EVERY, gamma=GAMMA, device=device,
            attn_dim=ATTN_DIM, n_heads=N_HEADS, n_agents=N_AGENTS
        )
        for i in range(N_AGENTS)
    ]

    # ── Actors ───────────────────────────────────────────────────
    agents = [
        DdpgActor(
            state_size=state_size, action_size=action_size, seed=i,
            batch_size=BATCH_SIZE, actor_lr=LR_ACTOR,
            experience_buffer=memory, critic=critics[i],
            weight_decay=0, tau=TAU, update_every=UPDATE_EVERY,
            gamma=GAMMA, device=device
        )
        for i in range(N_AGENTS)
    ]

    if train_mode:
        train_agent(
            agents, memory, env, action_size, BATCH_SIZE, UPDATE_EVERY,
            file_prefix=args.output_weights_prefix,
            n_episodes=args.episodes,
            epsilon_start=EPS_START, epsilon_end=EPS_END,
            epsilon_decay=EPS_DECAY, resume=args.resume,
        )
    else:
        print("Test mode — loading weights...")
        prefix = args.output_weights_prefix
        for i, agent in enumerate(agents):
            w = f"{prefix}_actor_{i}.pth"
            if os.path.exists(w):
                agent.actor_train.load_state_dict(torch.load(w, map_location=device))
                print(f"  Loaded {w}")

        env_info = env.reset(train_mode=False)[brain_name]
        states   = env_info.vector_observations[:2]
        score    = np.zeros(2)
        actions_full = np.zeros((20, action_size))
        while True:
            actions_2 = np.vstack([a.act(s, add_noise=False) for a, s in zip(agents, states)])
            actions_full[:2] = actions_2
            env_info    = env.step(actions_full)[brain_name]
            next_states = env_info.vector_observations[:2]
            rewards     = env_info.rewards[:2]
            dones       = env_info.local_done[:2]
            score      += rewards
            states      = next_states
            if np.any(dones):
                break
        print(f"Test score: {np.mean(score):.3f}")

    env.close()