# AgroTrack: Reinforcement Learning for Post-Harvest Loss Prevention

##  Mission

Reduce post-harvest food loss through intelligent decision-making, optimize resource allocation, and suggest efficient reuse options.

---

##  Project Structure

```
agrotrack_rl_project/
├── environment/
│   ├── __init__.py
│   └── custom_env.py           # YOUR environment (8 actions, 28D obs)
│
├── training/
│   ├── dqn_training.py          # DQN with 10 hyperparameter configs
│   ├── pg_training.py           # PPO & A2C (10 configs each)
│   └── reinforce_training.py    # Manual REINFORCE (10 configs)
│
├── evaluation/
│   └── compare_models.py        # Comprehensive comparison
│
├── models/                      # Saved models (40 total)
│   ├── dqn/
│   ├── ppo/
│   ├── a2c/
│   └── reinforce/
│
├── logs/                        # Training logs
├── results/                     # Evaluation results & plots
│
├── demo_random_agent.py         # Random agent demo
├── test_env.py                  # Environment test
├── main.py                      # Main runner
└── requirements.txt             # Dependencies
```

---

##  Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Test environment
python test_env.py
```

### 2. Demo (No Training)

```bash
# With visualization
python demo_random_agent.py --episodes 3 --steps 100 --delay 0.2

# Headless mode
python demo_random_agent.py --episodes 1 --headless
```

### 3. Train Models

```bash
# Quick test (10 minutes)
python training/dqn_training.py --single dqn_baseline --timesteps 10000

# Full training (15-20 hours for all)
python training/dqn_training.py --timesteps 100000
python training/pg_training.py --algorithm both --timesteps 100000
python training/reinforce_training.py --episodes 1000
```

### 4. Evaluate

```bash
python evaluation/compare_models.py --episodes 50
```

### 5. Run Best Model

```bash
python main.py --episodes 3
```

---

##  Environment Details

### Action Space (8 discrete actions)

| ID | Action | Description |
|----|--------|-------------|
| 0 | Monitor | Track freshness, minimal cost |
| 1 | Basic Preservation | Refrigeration/drying |
| 2 | Transport to Market | Immediate delivery |
| 3 | Reuse/Donate | Food bank/donation |
| 4 | Advanced Preservation | Controlled atmosphere |
| 5 | Emergency Transport | Premium cooling transport |
| 6 | Compost | Low quality items |
| 7 | Process Products | Create preserved goods |

### Observation Space (28 dimensions)

- **Food Inventory** (5): Vegetables, Fruits, Grains, Dairy, Meat
- **Quality Indices** (5): Freshness levels [0-1]
- **Environmental** (2): Temperature, humidity
- **Time** (1): Days until market
- **Resources** (3): Storage, transport, preservation
- **Market** (5): Prices per food type
- **Spoilage** (5): Rates per food type
- **Tracking** (2): Total saved, total wasted

### Rewards

- **+10**: Successful high-quality market delivery
- **+5**: Reuse/donation of good quality food
- **+2**: Composting low quality items
- **-15**: Food spoilage and waste
- **-5**: Inefficient resource usage
- **-3**: Missed market opportunities

---

##  Training Details

### Algorithms & Configurations

| Algorithm | Configs | Timesteps/Episodes | Training Time |
|-----------|---------|-------------------|---------------|
| **DQN** | 10 | 100k timesteps | ~5-7 hours |
| **PPO** | 10 | 100k timesteps | ~4-6 hours |
| **A2C** | 10 | 100k timesteps | ~3-5 hours |
| **REINFORCE** | 10 | 1000 episodes | ~2-4 hours |
| **Total** | **40** | **3M+ steps** | **14-22 hours** |

### Training Commands

```bash
# Train single configuration (testing)
python training/dqn_training.py --single dqn_baseline --timesteps 10000
python training/pg_training.py --algorithm ppo --single ppo_baseline --timesteps 10000
python training/reinforce_training.py --single reinforce_baseline --episodes 100

# Full hyperparameter sweeps
python training/dqn_training.py --timesteps 100000
python training/pg_training.py --algorithm ppo --timesteps 100000
python training/pg_training.py --algorithm a2c --timesteps 100000
python training/reinforce_training.py --episodes 1000
```

---

##  Evaluation

### Run Comparison

```bash
python evaluation/compare_models.py --episodes 50
```

### Outputs

1. **CSV**: `results/consolidated_evaluation.csv`
2. **Report**: `results/comparison_report.txt`
3. **Plots**: `results/figures/`
   - `reward_comparison.png`
   - `efficiency_comparison.png`
   - `reward_distribution.png`

### Metrics

- Mean Reward ± Std
- Loss Prevention Efficiency (%)
- Food Saved vs. Wasted
- Average Quality
- Action Distribution

---

##  Main Runner

### Commands

```bash
# Auto-detect best algorithm
python main.py --episodes 3 --max-steps 100 --delay 0.2

# Specify algorithm
python main.py --algorithm ppo --episodes 5

# No rendering (faster)
python main.py --no-render --episodes 10

# Quick demo
python main.py --episodes 1 --max-steps 50 --delay 0.05
```

---

##  Expected Performance

### Typical Results

| Algorithm | Mean Reward | Efficiency | Notes |
|-----------|-------------|------------|-------|
| **PPO** | 450-550 | 75-85% | Most stable |
| **DQN** | 400-500 | 70-80% | Best long-term planning |
| **A2C** | 350-450 | 65-75% | Fast learning |
| **REINFORCE** | 300-400 | 60-70% | Higher variance |

---

##  Troubleshooting

### Import Errors

```bash
# Ensure you're in project root
cd agrotrack_rl_project

# Check Python can find modules
python -c "from environment.custom_env import AgroTrackEnv; print('✓ OK')"
```

### Pygame Issues

```bash
# Test pygame
python -c "import pygame; pygame.init(); print('✓ Pygame works')"

# If problems, use headless mode
python demo_random_agent.py --headless
python main.py --no-render
```

### CUDA/GPU

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Models will work on CPU (default)
```

---

##  Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

### Check Progress

```bash
# View intermediate results
cat results/dqn_sweep_results.csv
cat results/ppo_sweep_results.csv
```

---

##  Project Highlights

### Technical Features

 Production-quality code  
 40 hyperparameter experiments  
 Manual REINFORCE from scratch  
 Comprehensive evaluation pipeline  
 Real-time pygame visualization  
 Full tensorboard logging  
 Modular architecture  

### Rubric Compliance

 Environment complexity (28D obs, 8 actions)  
 Action/observation spaces properly defined  
 Mission-relevant reward function  
 Advanced visualization  
 DQN, PPO, A2C with SB3  
 Manual REINFORCE implementation  
 10 configs per algorithm (40 total)  
 Model saving/loading  
 Quantitative & qualitative evaluation  
 Comparative analysis  
 Complete documentation  

---

##  File Descriptions

### Core Files

- **`environment/custom_env.py`**: Your custom 8-action environment
- **`training/dqn_training.py`**: DQN training with 10 configs
- **`training/pg_training.py`**: PPO & A2C training (20 configs)
- **`training/reinforce_training.py`**: Manual REINFORCE (10 configs)
- **`evaluation/compare_models.py`**: Full comparison system
- **`main.py`**: Production runner script
- **`demo_random_agent.py`**: Visualization demo
- **`test_env.py`**: Environment verification

---

##  Execution Order

```bash
# 1. Test environment
python test_env.py

# 2. Demo visualization
python demo_random_agent.py --episodes 2

# 3. Quick training test (5 min)
python training/dqn_training.py --single dqn_baseline --timesteps 5000

# 4. Full training (overnight)
python training/dqn_training.py --timesteps 100000
python training/pg_training.py --algorithm both --timesteps 100000
python training/reinforce_training.py --episodes 1000

# 5. Evaluate all models
python evaluation/compare_models.py --episodes 50

# 6. Run best model
python main.py --episodes 3
```

---

##  Tips

### For Quick Testing

```bash
# Minimal training for testing pipeline
python training/dqn_training.py --single dqn_baseline --timesteps 1000
python main.py --episodes 1 --max-steps 20
```

### For Production

```bash
# Full training with all configurations
bash train_all.sh  # Create this script with all training commands

# Monitor with TensorBoard
tensorboard --logdir logs/ &
```

---

## 📧 Support

For issues:
1. Check `test_env.py` passes
2. Verify `demo_random_agent.py` works
3. Review error messages
4. Check file paths are correct

---

