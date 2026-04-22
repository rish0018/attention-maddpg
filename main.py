# main.py

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
    print("Warning: unityagents package not found. Install with: pip install unityagents")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_file', default="maddpg_env/Reacher.exe")
    parser.add_argument('--mode', choices=["train", "test"], default="train")
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--output_weights_prefix', type=str, default="model")

    args = parser.parse_args()

    train_mode = args.mode == "train"

    # -------- LOAD ENV --------
    if not HAS_UNITY:
        print("Error: UnityEnvironment is required. Please install unityagents package.")
        sys.exit(1)

    env_path = args.env_file
    if not os.path.exists(env_path):
        cwd = os.getcwd()
        print(f"Error: Environment file not found at {env_path}")
        print(f"Current working directory: {cwd}")
        print(f"Expected location: {os.path.join(cwd, env_path)}")
        print("Please provide a valid path to the Unity environment executable.")
        print("Usage: python main.py --env_file <path_to_env>")
        sys.exit(1)

    env = UnityEnvironment(
        file_name=env_path,
        no_graphics=not train_mode
    )

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    print("Resetting env...")

    env_info = env.reset(train_mode=train_mode)[brain_name]

    num_env_agents = len(env_info.agents)
    print("Number of agents in environment:", num_env_agents)

    action_size = brain.vector_action_space_size
    print("Action size:", action_size)

    state = env_info.vector_observations[0]
    state_size = len(state)
    print("State size:", state_size)

    # -------- HYPERPARAMETERS --------
    BUFFER_SIZE = int(1e5)
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 1e-3
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3
    UPDATE_EVERY = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -------- REPLAY BUFFER --------
    memory = ReplayBuffer(
        action_size,
        BUFFER_SIZE,
        BATCH_SIZE,
        seed=0,
        device=device
    )

    # -------- CRITIC --------
    critic = DdpgCritic(
        state_size=state_size,
        action_size=action_size,
        seed=0,
        critic_lr=LR_CRITIC,
        weight_decay=0,
        tau=TAU,
        update_every=UPDATE_EVERY,
        gamma=GAMMA,
        device=device
    )

    # -------- CREATE 2 AGENTS --------
    agents = [
        DdpgActor(
            state_size=state_size,
            action_size=action_size,
            seed=0,
            batch_size=BATCH_SIZE,
            actor_lr=LR_ACTOR,
            experience_buffer=memory,
            critic=critic,
            weight_decay=0,
            tau=TAU,
            update_every=UPDATE_EVERY,
            gamma=GAMMA,
            device=device
        )
        for _ in range(2)
    ]

    # -------- TRAIN --------
    if train_mode:
        train_agent(
            agents,
            memory,
            env,
            action_size,
            BATCH_SIZE,
            UPDATE_EVERY,
            file_prefix=args.output_weights_prefix,
            n_episodes=args.episodes
        )

    else:
        print("Test mode not implemented in this version.")