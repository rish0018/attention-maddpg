# Attention-MADDPG: Multi-Agent Deep Deterministic Policy Gradient with Attention Mechanism

This project implements a multi-agent reinforcement learning algorithm that combines **MADDPG** (Multi-Agent Deep Deterministic Policy Gradient) with **attention mechanisms** for improved coordination and scalability in multi-agent environments.

## Project Overview

The Attention-MADDPG framework extends the traditional MADDPG algorithm by incorporating attention mechanisms to enable agents to focus on relevant information from other agents. This approach improves learning efficiency and scalability in complex multi-agent scenarios.

## Features

- **Multi-Agent Learning**: Support for training multiple cooperative/competitive agents simultaneously
- **Attention Mechanism**: Dynamic attention weights for agent-to-agent communication
- **Experience Replay**: Efficient replay buffer management for sample efficiency
- **Centralized Training, Decentralized Execution**: Centralized critic networks during training with decentralized actor networks during execution
- **Checkpointing**: Model saving and loading for long training sessions
- **Modular Design**: Clean separation of concerns with dedicated modules for agents, models, and utilities

## Project Structure

```
attention-maddpg/
├── main.py                 # Entry point and training loop
├── ddpg_agent.py          # DDPG agent implementation with attention
├── model.py               # Neural network models (Actor & Critic)
├── replaybuffers.py       # Experience replay buffer management
├── utils.py               # Utility functions and helpers
├── results/               # Training results and metrics
│   ├── baseline/          # Baseline algorithm results
│   ├── improved/          # Improved algorithm results
│   └── final/             # Final model results
├── checkpoints/           # Saved model checkpoints
└── README.md              # This file
```

## Installation

### Requirements
- Python 3.8+
- PyTorch
- NumPy
- OpenAI Gym (or appropriate environment)

### Setup

```bash
# Clone or navigate to the project directory
cd attention-maddpg

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy gym
```

## Usage

### Training

```bash
# Run the training script
python main.py
```

### Configuration

Edit the configuration parameters in `main.py`:
- `num_agents`: Number of agents in the environment
- `state_size`: Dimension of state space
- `action_size`: Dimension of action space
- `learning_rate`: Learning rate for network optimization
- `batch_size`: Batch size for experience replay
- `episodes`: Number of training episodes

### Loading Checkpoints

```python
from ddpg_agent import Agent

agent = Agent(...)
agent.load_checkpoint('checkpoints/model.pth')
```

## Key Components

### 1. **ddpg_agent.py**
Implements the DDPG agent with attention mechanism:
- Actor network: Decides actions based on state
- Critic network: Evaluates state-action pairs
- Attention layers: Weight agent interactions

### 2. **model.py**
Neural network architectures:
- `Actor`: Policy network for action selection
- `Critic`: Value network for state-action evaluation
- `AttentionLayer`: Multi-head attention for agent coordination

### 3. **replaybuffers.py**
Experience replay buffer:
- Stores transitions (state, action, reward, next_state, done)
- Efficient sampling for mini-batch training
- Priority-based sampling (optional)

### 4. **utils.py**
Utility functions:
- Environment wrappers
- Reward normalization
- Logging and visualization helpers

## Algorithm Overview

### Training Process
1. **Collect Experiences**: Agents interact with environment using current policies
2. **Update Critics**: Learn state-action values using centralized critics
3. **Update Actors**: Improve policies based on critic feedback
4. **Attention Update**: Refine attention weights for inter-agent communication

## Results

- **Baseline**: Results using standard MADDPG
- **Improved**: Results with attention mechanism
- **Final**: Best performing model configuration

See `results/` directory for detailed metrics and plots.

## References

- [MADDPG Paper](https://arxiv.org/abs/1706.02275) - Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
- [Transformer Architecture](https://arxiv.org/abs/1706.03762) - Attention Is All You Need
- PyTorch Documentation: https://pytorch.org/docs/

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or inquiries, please contact the project maintainers.

---

**Last Updated**: March 2026
