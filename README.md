# AgroTrack: Reinforcement Learning for Post-Harvest Food Loss Prevention

A custom Gymnasium environment specifically modeled for post-harvest food loss prevention using Reinforcement Learning (RL). This project implements and trains four different RL algorithms (DQN, REINFORCE, PPO, and A2C) to optimize agricultural supply chain decisions.

## Overview

AgroTrack simulates real-world agricultural dynamics where an agent must make decisions about storage conditions, handling, and transportation to minimize food loss while managing costs. The environment models key factors such as temperature, humidity, product quality, and storage costs.

## Features

- **Custom Gymnasium Environment**: Fully compliant with Gymnasium API standards
- **Realistic Agricultural Dynamics**: Models temperature, humidity, quality degradation, and costs
- **Multiple RL Algorithms**: Implementations of DQN, REINFORCE, PPO, and A2C
- **Comprehensive Training Scripts**: Ready-to-use scripts for training each algorithm
- **Visualization Tools**: Compare and analyze model performance
- **Modular Architecture**: Clean separation of environment, training, and visualization code

## Project Structure

```
Agrotrack_summatives/
├── agrotrack_env/              # Custom Gymnasium environment
│   ├── __init__.py
│   └── agrotrack_env.py        # Environment implementation
├── training_scripts/           # Training scripts for each algorithm
│   ├── train_dqn.py           # Deep Q-Network
│   ├── train_ppo.py           # Proximal Policy Optimization
│   ├── train_a2c.py           # Advantage Actor-Critic
│   └── train_reinforce.py     # REINFORCE (Policy Gradient)
├── visualization/              # Visualization and comparison tools
│   └── compare_models.py      # Compare trained models
├── models/                     # Saved trained models (created during training)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Environment Details

### Observation Space

The environment provides a 6-dimensional continuous observation space:

1. **Temperature (°C)**: 0-40 - Current storage temperature
2. **Humidity (%)**: 0-100 - Current storage humidity
3. **Days Since Harvest**: 0-30 - Time elapsed since harvest
4. **Quality Index**: 0-100 - Current product quality (100 = perfect)
5. **Storage Cost**: 0-1000 - Accumulated storage costs
6. **Product Quantity (kg)**: 0-1000 - Amount of produce

### Action Space

5 discrete actions representing different interventions:

- **Action 0**: No intervention (low cost, natural degradation)
- **Action 1**: Basic cooling (moderate cost, slows degradation)
- **Action 2**: Advanced cooling (high cost, minimal degradation)
- **Action 3**: Humidity control (moderate cost, prevents moisture damage)
- **Action 4**: Rush to market (high cost, saves remaining quality)

### Reward Structure

The reward function balances multiple objectives:

- **Quality Maintenance**: Positive reward for maintaining high quality
- **Quality Loss Penalty**: Penalty proportional to quality degradation
- **Cost Penalty**: Penalty for intervention costs
- **Success Bonus**: Large bonus for delivering high-quality produce
- **Spoilage Penalty**: Severe penalty for total product loss

### Terminal States

Episodes terminate when:

1. Product quality drops below 10% (spoilage)
2. Maximum time steps reached (30 days)
3. Agent chooses to rush to market (Action 4)

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nitekar/Agrotrack_summatives.git
cd Agrotrack_summatives
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

Each algorithm has its own training script. Navigate to the project root and run:

#### Train DQN
```bash
python training_scripts/train_dqn.py
```

#### Train PPO
```bash
python training_scripts/train_ppo.py
```

#### Train A2C
```bash
python training_scripts/train_a2c.py
```

#### Train REINFORCE
```bash
python training_scripts/train_reinforce.py
```

Training parameters can be modified directly in the respective training scripts.

### Comparing Models

After training all models, compare their performance:

```bash
python visualization/compare_models.py
```

This will:
- Load all trained models
- Evaluate each model over 20 episodes
- Generate comparison plots
- Print summary statistics

### Using the Environment Standalone

```python
from agrotrack_env import AgroTrackEnv

# Create environment
env = AgroTrackEnv(render_mode="human")

# Reset environment
obs, info = env.reset()

# Run episode
terminated = False
truncated = False
total_reward = 0

while not (terminated or truncated):
    # Random action for demonstration
    action = env.action_space.sample()
    
    # Take action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    # Render environment
    env.render()

print(f"Episode finished with total reward: {total_reward}")
```

## Algorithms Implemented

### 1. DQN (Deep Q-Network)
- **Type**: Value-based
- **Features**: Experience replay, target network
- **Best for**: Discrete action spaces, sample efficiency

### 2. REINFORCE
- **Type**: Policy gradient (Monte Carlo)
- **Features**: Direct policy optimization, episodic updates
- **Best for**: Simplicity, policy-based learning

### 3. PPO (Proximal Policy Optimization)
- **Type**: Policy gradient (Actor-Critic)
- **Features**: Clipped objective, stable updates
- **Best for**: Stability, continuous action spaces

### 4. A2C (Advantage Actor-Critic)
- **Type**: Policy gradient (Actor-Critic)
- **Features**: Advantage estimation, synchronous updates
- **Best for**: Fast training, parallel environments

## Model Performance

After training, models are evaluated on:
- **Average Episode Reward**: Total reward per episode
- **Final Product Quality**: Quality index at episode end
- **Storage Costs**: Total costs incurred
- **Episode Length**: Days until termination or market delivery

Results are visualized in comparison plots showing:
- Episode reward trends
- Reward distributions (box plots)
- Final quality distributions
- Storage cost comparisons

## Customization

### Modifying the Environment

Edit `agrotrack_env/agrotrack_env.py` to customize:
- Action costs
- Quality degradation rates
- Environmental stress factors
- Reward weights
- Maximum episode length

### Tuning Hyperparameters

Each training script contains hyperparameters that can be adjusted:
- Learning rate
- Batch size
- Network architecture
- Discount factor (gamma)
- Exploration parameters

## Integration with Stable Baselines3

The environment is fully compatible with Stable Baselines3, allowing you to:
- Use built-in callbacks (evaluation, checkpointing)
- Leverage tensorboard logging
- Apply vectorized environments
- Use pre-trained models

Example:
```python
from stable_baselines3 import PPO
from agrotrack_env import AgroTrackEnv

env = AgroTrackEnv()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_agrotrack")
```

## Future Enhancements

Potential improvements and extensions:
- Multi-agent scenarios (multiple storage facilities)
- Stochastic events (weather, market fluctuations)
- Continuous action spaces (temperature/humidity control)
- Real-world data integration
- Transfer learning from related tasks

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is available for educational and research purposes.

## Acknowledgments

- Built using [Gymnasium](https://gymnasium.farama.org/)
- RL algorithms from [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- Inspired by real-world agricultural challenges in post-harvest management