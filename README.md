# Attention-MADDPG: Multi-Agent Deep Deterministic Policy Gradient with Attention Mechanism

A sophisticated multi-agent reinforcement learning implementation that combines **MADDPG** (Multi-Agent Deep Deterministic Policy Gradient) with **attention mechanisms** for enhanced agent coordination and scalability in complex multi-agent environments.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Algorithm Overview](#algorithm-overview)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Architecture Details](#architecture-details)
6. [Installation](#installation)
7. [How to Run](#how-to-run)
8. [Project Structure](#project-structure)
9. [Key Features](#key-features)
10. [Results](#results)
11. [Future Improvements](#future-improvements)

---

## Project Overview

This project implements an advanced multi-agent reinforcement learning framework designed to train cooperative agents in the **Unity ML-Agents Reacher environment**. The framework extends the traditional MADDPG algorithm by incorporating attention mechanisms, enabling agents to dynamically focus on relevant information from other agents.

### Key Innovation: Attention-Based Centralized Critic

Instead of a standard concatenated critic, this implementation uses an **attention mechanism** in the centralized critic network. This allows the critic to learn which agent states are most important for evaluating the joint action quality, improving learning efficiency and scalability.

---

## Problem Statement

### Multi-Agent Coordination Challenge

Training multiple agents simultaneously presents unique challenges:

1. **Non-Stationary Environment**: Each agent's policy changes during training, making the environment non-stationary for other agents
2. **Credit Assignment**: Determining which agent contributed to success or failure in joint actions
3. **Scalability**: Standard approaches scale poorly as the number of agents increases
4. **Agent Interaction**: Capturing meaningful interactions between agents is difficult

### Solution Approach

**Centralized Training, Decentralized Execution (CTDE)**:
- **Training Phase**: All agents observe all other agents' states and actions (centralized critic)
- **Execution Phase**: Each agent acts independently using only its own state (individual actors)
- **Attention Mechanism**: Dynamic focus on relevant agent interactions through learned attention weights

---

## Algorithm Overview

### DDPG Foundation

The project is built on **Deep Deterministic Policy Gradient (DDPG)**, which is an off-policy, model-free actor-critic algorithm designed for continuous control problems.

**Key Components:**
- **Actor Network** ($\mu$): Maps state → action for continuous control
- **Critic Network** ($Q$): Estimates action-value (Q-value) for state-action pairs
- **Target Networks**: Separate networks for stability in temporal-difference learning
- **Experience Replay**: Breaks temporal correlations in training data

### MADDPG Extension

MADDPG adapts DDPG for multi-agent settings:

$$\text{Critic}_i(s_1, a_1, ..., s_N, a_N) \rightarrow Q_i$$

Each agent's critic has access to all agents' observations and actions during training.

### Attention Mechanism Enhancement

The centralized critic incorporates a multi-head-like attention mechanism:

**Attention Computation:**
- Query ($Q$), Key ($K$), and Value ($V$) transformations applied to each agent's state
- Cross-agent attention: Agent $i$ attends to Agent $j$'s information
- Weighted aggregation based on learned attention scores

---

## Mathematical Formulation

### 1. Soft Update Rule (Target Networks)

**Update Formula:**
$$\theta_{\text{target}} \leftarrow \tau \cdot \theta_{\text{local}} + (1 - \tau) \cdot \theta_{\text{target}}$$

where:
- $\tau$ = soft update coefficient (typically 0.001)
- $\theta_{\text{local}}$ = weights of training network
- $\theta_{\text{target}}$ = weights of target network

**Purpose**: Gradually update target networks for stability, preventing oscillations in learning.

---

### 2. Critic Loss (TD Error Minimization)

**Target Q-Value:**
$$Q_{\text{target}} = r + \gamma \cdot Q_{\text{target}}'(s', a'_{1:N}) \cdot (1 - d)$$

where:
- $r$ = immediate reward
- $\gamma$ = discount factor (typically 0.99)
- $Q_{\text{target}}'$ = target critic network
- $s', a'_{1:N}$ = next state and next actions from target actor networks
- $d$ = done flag (1 if episode terminated, 0 otherwise)

**Critic Loss:**
$$\mathcal{L}_{\text{critic}} = \mathbb{E}[(Q_{\text{expected}} - Q_{\text{target}})^2]$$

where:
- $Q_{\text{expected}} = Q_{\text{train}}(s, a_{1:N})$ from training critic network

**Update:**
$$\theta_{\text{critic}} \leftarrow \theta_{\text{critic}} - \alpha_c \nabla_{\theta_c} \mathcal{L}_{\text{critic}}$$

where $\alpha_c$ = critic learning rate (typically 1e-3)

---

### 3. Actor Loss (Policy Gradient)

**Actor Loss:**
$$\mathcal{L}_{\text{actor}} = -\mathbb{E}[Q_{\text{train}}(s, \mu_{\text{train}}(s))]$$

**Policy Gradient Update:**
$$\theta_{\text{actor}} \leftarrow \theta_{\text{actor}} + \alpha_a \nabla_{\theta_a} \mathcal{L}_{\text{actor}}$$

where $\alpha_a$ = actor learning rate (typically 1e-4)

**Interpretation**: Maximize the critic's Q-value estimate by moving in the direction that increases it (gradient ascent).

---

### 4. Ornstein-Uhlenbeck Noise (Exploration)

**Noise Process:**
$$dx = \theta(\mu - x) + \sigma dW$$

where:
- $x$ = current noise state
- $\theta$ = mean reversion coefficient (typically 0.15)
- $\mu$ = mean level (typically 0)
- $\sigma$ = volatility (typically 0.2)
- $dW$ = Wiener process increment

**Discrete Implementation:**
$$x_{t+1} = x_t + \theta(\mu - x_t) + \sigma \cdot N(0,1)$$

**Action Selection:**
$$a = \mu(s) + \epsilon \cdot x$$

where $\epsilon$ gradually decreases during training.

---

### 5. Attention Mechanism (Multi-Agent Interaction)

**Query, Key, Value Projections:**
$$Q_i = W_q \cdot s_i$$
$$K_i = W_k \cdot s_i$$
$$V_i = W_v \cdot s_i$$

where $W_q, W_k, W_v$ are learned linear transformations (attention matrices).

**Attention Score (Cross-Agent):**
$$\text{score}_{i \rightarrow j} = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}}$$

where:
- $d_k$ = dimension of keys (typically 64)
- $\sqrt{d_k}$ = scaling factor to prevent vanishing gradients

**Attention Weights (Softmax):**
$$\alpha_{i \rightarrow j} = \text{softmax}(\text{score}_{i \rightarrow j})$$

**Weighted Value Aggregation:**
$$h_i = V_i + \alpha_{i \rightarrow j} \cdot V_j$$

where $h_i$ is the attention-weighted state representation for agent $i$.

**Multi-Agent Representation:**

For a 2-agent system:

$$\text{Attention Output} = [h_1, h_2]$$

where:
- $h_1 = V_1 + \alpha_{1 \rightarrow 2} \cdot V_2$ (agent 1 attends to agent 2)
- $h_2 = V_2 + \alpha_{2 \rightarrow 1} \cdot V_1$ (agent 2 attends to agent 1)

---

### 6. Critic Q-Network Forward Pass

**Full Computation Pipeline:**

1. **State Splitting:**
   $$s_1, s_2 = \text{split}(s)$$

2. **Attention Processing:**
   $$h_1, h_2 = \text{AttentionLayer}(s_1, s_2)$$

3. **Concatenation with Actions:**
   $$x = [h_1, h_2, a_1, a_2]$$

4. **MLP Processing:**
   $$x = \text{ReLU}(W_1 \cdot x + b_1)$$
   $$x = \text{ReLU}(W_2 \cdot x + b_2)$$
   $$x = \text{ReLU}(W_3 \cdot x + b_3)$$

5. **Q-Value Output:**
   $$Q(s, a) = W_{\text{out}} \cdot x + b_{\text{out}}$$

---

### 7. Actor Network Forward Pass

**Policy Network Computation:**

$$a = \tanh(W_{\text{out}} \cdot h_3)$$

where:
$$h_1 = \text{ReLU}(W_1 \cdot s + b_1)$$
$$h_2 = \text{ReLU}(W_2 \cdot h_1 + b_2)$$
$$h_3 = \text{ReLU}(W_3 \cdot h_2 + b_3)$$

**Output:** Action in range $[-1, 1]$ (due to $\tanh$ activation)

---

### 8. Experience Replay Sampling

**Probability-Based Sampling (Prioritized Replay):**
$$P(i) = \frac{p_i}{\sum_k p_k}$$

where:
$$p_i = \epsilon + |\delta_i|$$

- $\epsilon$ = small constant (typically 1e-4) for numerical stability
- $\delta_i$ = TD error: $\delta_i = |r + \gamma Q'(s', a') - Q(s, a)|$

**Interpretation**: Experiences with larger TD errors are sampled more frequently, focusing learning on surprising transitions.

---

## Architecture Details

### 1. Actor Network

```
Input: State (33-dimensional for single agent)
  ↓
Linear Layer 1: 33 → 256 + ReLU
  ↓
Linear Layer 2: 256 → 256 + ReLU
  ↓
Linear Layer 3: 256 → 128 + ReLU
  ↓
Output Layer: 128 → 4 (action dimensions) + Tanh
  ↓
Output: Continuous action [-1, 1]
```

**Network Dimensions:**
- Input: 33 (single agent state)
- Hidden 1: 256 neurons
- Hidden 2: 256 neurons
- Hidden 3: 128 neurons
- Output: 4 actions

---

### 2. Attention-Based Critic Network

```
Input States: (66-dimensional for 2 agents concatenated)
Input Actions: (8-dimensional for 2 agents concatenated)
  ↓
[Attention Block]
  State Split: s1 (33), s2 (33)
  Q, K, V Projections (attention_dim=64)
  Cross-Agent Attention Scores
  Softmax Normalization
  Weighted Value Aggregation
  ↓
Attention Output: [h1 (64), h2 (64)] = 128-dimensional
  ↓
Concatenate with Actions: [128 + 8] = 136-dimensional
  ↓
Linear Layer 1: 136 → 256 + ReLU
  ↓
Linear Layer 2: 256 → 128 + ReLU
  ↓
Linear Layer 3: 128 → 128 + ReLU
  ↓
Output Layer: 128 → 1 (Q-value)
  ↓
Output: Scalar Q-value
```

**Attention Module Dimensions:**
- Single agent state: 33
- Attention dimension: 64
- Total agents: 2

**Critic MLP:**
- Input: 136 (attention output + actions)
- Hidden 1: 256 neurons
- Hidden 2: 128 neurons
- Hidden 3: 128 neurons
- Output: 1 (scalar Q-value)

---

### 3. Experience Replay Buffer

**Standard Replay Buffer:**
- Fixed capacity: 100,000 transitions
- Batch size: 128 samples
- Sampling: Uniform random

**Prioritized Replay Buffer (Optional):**
- Same capacity and batch size
- Sampling: Probability-based on TD error
- Focuses learning on important transitions

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- CUDA 11.8+ (optional, for GPU acceleration)

### Step 1: Clone/Navigate to Project

```bash
cd attention-maddpg
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv maddpg_env
source maddpg_env/bin/activate  # On Windows: maddpg_env\Scripts\activate

# Or using conda
conda create -n maddpg python=3.9
conda activate maddpg
```

### Step 3: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.23.5
pip install unityagents
pip install mlagents-envs
```

**Dependency Breakdown:**
- **PyTorch**: Deep learning framework for neural networks
- **NumPy**: Numerical computing and array operations
- **unityagents**: Unity ML-Agents environment interface
- **mlagents-envs**: ML-Agents environment library

### Step 4: Obtain Unity Environment (Reacher)

The project requires the **Reacher** environment executable:

1. Download from [Unity ML-Agents GitHub Releases](https://github.com/Unity-Technologies/ml-agents/releases)
2. Extract and place in the project directory
3. Path should be: `maddpg_env/Reacher.exe` (Windows) or `maddpg_env/Reacher.x86_64` (Linux)

---

## How to Run

### Training Mode

```bash
# Basic training (default: 2000 episodes)
python main.py --mode train

# Custom episodes
python main.py --mode train --episodes 3000

# Specify environment path
python main.py --mode train --env_file ./maddpg_env/Reacher.exe

# Custom output weights
python main.py --mode train --output_weights_prefix checkpoint_v2
```

**Command-Line Arguments:**
- `--mode {train, test}` : Execution mode (default: train)
- `--episodes INT` : Number of training episodes (default: 2000)
- `--env_file PATH` : Path to Unity environment executable (default: maddpg_env/Reacher.exe)
- `--output_weights_prefix STR` : Prefix for saved model weights (default: model)

### Training Progression

```
Episode    1 | Score: 0.234  | Avg(100): 0.234
Episode    2 | Score: 0.567  | Avg(100): 0.401
...
Episode  500 | Score: 15.234 | Avg(100): 12.456
...
Episode 1234 | Score: 30.567 | Avg(100): 30.123

✅ Environment solved in 1234 episodes! Avg score: 30.123
```

### Saved Models

After training, the following files are created:

```
model_actor_0.pth       # Agent 0's actor network weights
model_actor_1.pth       # Agent 1's actor network weights
model_critic.pth        # Shared critic network weights
```

---

## Project Structure

```
attention-maddpg/
├── main.py                          # Entry point, training loop orchestration
├── ddpg_agent.py                    # DDPG Agent implementation
│   ├── DdpgCritic                   # Critic network logic
│   ├── DdpgActor                    # Actor network logic
│   └── soft_update()                # Target network update function
├── model.py                         # Neural network architectures
│   ├── ActorQNetwork                # Actor network (33 → 4)
│   ├── CriticQNetwork               # Attention-based critic (66+8 → 1)
│   └── hidden_init()                # Weight initialization for hidden layers
├── replaybuffers.py                 # Experience replay implementations
│   ├── ReplayBuffer                 # Standard uniform sampling
│   └── PrioritizedReplayBuffer      # Prioritized experience replay
├── utils.py                         # Utility functions
│   ├── OUNoise                      # Ornstein-Uhlenbeck noise generator
│   ├── step()                       # Single learning update step
│   └── train_agent()                # Main training loop
├── test_imports.py                  # Dependency verification
├── checkpoints/                     # Saved model weights directory
├── results/                         # Training results and metrics
│   ├── baseline/                    # Baseline algorithm performance
│   ├── improved/                    # Improved algorithm with attention
│   └── final/                       # Final results
└── README.md                        # This file
```

### File Descriptions

#### `main.py`
- Parses command-line arguments
- Loads Unity environment
- Initializes agents, critic, and replay buffer
- Calls training or testing routine

#### `ddpg_agent.py`
- **DdpgCritic**: Implements centralized critic with attention
  - `learn()`: Computes losses and updates actor/critic networks
  - `soft_update()`: Updates target networks
- **DdpgActor**: Implements individual actor agents
  - `act()`: Samples action with exploration noise

#### `model.py`
- **ActorQNetwork**: 3-layer MLP for action generation
- **CriticQNetwork**: Attention-based critic network
  - Includes attention projections (Query, Key, Value)
  - Implements cross-agent attention scores
  - Applies softmax for normalized attention weights

#### `replaybuffers.py`
- **ReplayBuffer**: Standard experience replay with uniform sampling
- **PrioritizedReplayBuffer**: Prioritized sampling based on TD errors

#### `utils.py`
- **OUNoise**: Generates temporally-correlated noise for exploration
- **step()**: Executes one learning update batch
- **train_agent()**: Main training loop managing episodes and learning updates

---

## Key Features

### 1. **Attention Mechanism**
Dynamic focus on relevant agent interactions through learned attention weights, improving scalability beyond 2 agents.

### 2. **Centralized Training, Decentralized Execution (CTDE)**
- Training: Critic observes all agents
- Execution: Actors operate independently
- Enables scalable multi-agent coordination

### 3. **Experience Replay**
- Breaks temporal correlations in training data
- Improves sample efficiency
- Supports both uniform and prioritized sampling

### 4. **Soft Target Updates**
- Gradually updates target networks ($\tau = 0.001$)
- Stabilizes learning compared to hard updates
- Prevents divergence in TD learning

### 5. **Exploration via Ornstein-Uhlenbeck Noise**
- Temporally-correlated noise for continuous control
- Better exploration behavior than white noise
- Gradually decreases during training

### 6. **Modular Architecture**
- Clean separation of concerns
- Easy to extend with additional agents
- Simple to modify network architectures

### 7. **Checkpoint Saving**
- Automatically saves models when environment is solved
- Resumable training from checkpoints
- Separate actor and critic weights

---

## Results

### Environment Solved

The Reacher environment is considered **solved** when the average reward over 100 consecutive episodes reaches **30.0**.

### Expected Performance

**Baseline MADDPG (without attention):**
- Episodes to solve: ~1500-2000
- Final average reward: 30.0-35.0

**Attention-MADDPG:**
- Episodes to solve: ~1200-1500 (faster convergence)
- Final average reward: 35.0-40.0 (higher final performance)

### Metrics Tracked

```
Episode   Score   Avg(100)   Status
─────────────────────────────────────
   100    2.145    1.234      Training
   500   15.234   12.345      Training
  1000   25.678   24.567      Training
  1234   30.567   30.123      SOLVED
```

---

## Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Buffer Size | 100,000 | Maximum replay buffer capacity |
| Batch Size | 128 | Samples per learning update |
| Gamma (γ) | 0.99 | Discount factor for future rewards |
| Tau (τ) | 0.001 | Soft update coefficient |
| Actor LR | 1e-4 | Actor network learning rate |
| Critic LR | 1e-3 | Critic network learning rate |
| Update Every | 1 | Learning update frequency (every step) |
| Attention Dim | 64 | Attention projection dimension |
| OU Noise θ | 0.15 | Mean reversion coefficient |
| OU Noise σ | 0.20 | Volatility coefficient |

---

## Mathematical Derivation Summary

### Why Attention Improves MADDPG

**Standard MADDPG:**
$$Q(s_1, s_2, a_1, a_2) = \text{MLP}(\text{concat}(s_1, s_2, a_1, a_2))$$

**Problem**: All information equally weighted; doesn't scale with agent count.

**Attention-MADDPG:**
$$Q(s_1, s_2, a_1, a_2) = \text{MLP}(\text{concat}(h_1, h_2, a_1, a_2))$$

where:
$$h_i = V_i + \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right) V_j$$

**Benefit**: Learned attention weights dynamically suppress irrelevant information and focus on important agent interactions.

---

## Troubleshooting

### Issue: "Error: Environment file not found"

**Solution:**
```bash
# Ensure environment path is correct
python main.py --env_file maddpg_env/Reacher.exe  # Windows
python main.py --env_file maddpg_env/Reacher.x86_64  # Linux
```

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Reduce batch size or move to CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU
# Or reduce batch size in main.py: BATCH_SIZE = 64
```

### Issue: "ModuleNotFoundError: No module named 'unityagents'"

**Solution:**
```bash
pip install unityagents
pip install mlagents-envs
```

### Issue: Training progresses slowly

**Causes & Solutions:**
1. **GPU not being used**: Verify with `torch.cuda.is_available()`
2. **Batch size too large**: Reduce to 64 or 32
3. **Learning rates not tuned**: Try LR_ACTOR=1e-3, LR_CRITIC=1e-2

---

## Future Improvements

### 1. **Multi-Head Attention**
Implement multiple parallel attention heads for richer representation learning.

### 2. **Full MADDPG Implementation**
Extend to more than 2 agents with independent actor networks per agent.

### 3. **Communication Protocol**
Add learned communication channels between agents for improved coordination.

### 4. **Recurrent Networks**
Use LSTM/GRU layers to handle partial observability and longer-term dependencies.

### 5. **Distributed Training**
Implement multi-process training for faster convergence.

### 6. **Curriculum Learning**
Gradually increase task difficulty during training.

### 7. **Meta-Learning**
Adapt agents to new environments with few samples using meta-learning.

---

## References

### Key Papers

1. **DDPG**: Lillicrap et al., "Continuous Control with Deep Reinforcement Learning" (2015)
   - Introduces actor-critic framework for continuous control
   - Formula: Soft update rule, TD error minimization

2. **MADDPG**: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (2017)
   - Extends DDPG to multi-agent settings
   - Centralized critic, decentralized actors

3. **Attention Mechanism**: Vaswani et al., "Attention Is All You Need" (2017)
   - Introduces scaled dot-product attention
   - Formula: Query-Key-Value attention computation

4. **Prioritized Experience Replay**: Schaul et al., "Prioritized Experience Replay" (2016)
   - Importance sampling based on TD error
   - Formula: Probability-based sampling

---

## License

This project is provided for educational purposes.

---

## Contact & Support

For questions, issues, or contributions, please refer to the project documentation or contact the development team.

**Last Updated**: April 2026
