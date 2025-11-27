"""
REINFORCE (Manual Policy Gradient Implementation)
From scratch using PyTorch for AgroTrack (8 actions)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from datetime import datetime
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import AgroTrackEnv

os.makedirs("models/reinforce", exist_ok=True)
os.makedirs("logs/reinforce", exist_ok=True)
os.makedirs("results", exist_ok=True)

class PolicyNetwork(nn.Module):
    """Policy network"""
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 128]):
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(input_dim, hidden_size), nn.ReLU()])
            input_dim = hidden_size
        layers.extend([nn.Linear(input_dim, action_dim), nn.Softmax(dim=-1)])
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

class REINFORCE:
    """REINFORCE algorithm"""
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 128], 
                 learning_rate=1e-3, gamma=0.99, device='cpu'):
        self.device = torch.device(device)
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        action, log_prob = self.policy.get_action(state)
        self.log_probs.append(log_prob)
        return action
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def update(self):
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        self.log_probs = []
        self.rewards = []
        return loss.item()
    
    def save(self, path):
        torch.save({'policy': self.policy.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

def evaluate_reinforce(agent, env, n_episodes=20):
    """Evaluate agent"""
    episode_rewards = []
    food_saved_list = []
    food_wasted_list = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                probs = agent.policy(state_tensor)
                action = torch.argmax(probs).item()
            
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        food_saved_list.append(info['total_saved'])
        food_wasted_list.append(info['total_wasted'])
    
    efficiency_list = []
    for saved, wasted in zip(food_saved_list, food_wasted_list):
        if saved + wasted > 0:
            efficiency_list.append((saved / (saved + wasted)) * 100)
        else:
            efficiency_list.append(0)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_food_saved": np.mean(food_saved_list),
        "mean_food_wasted": np.mean(food_wasted_list),
        "mean_efficiency": np.mean(efficiency_list),
    }

REINFORCE_CONFIGS = [
    {"name": "reinforce_baseline", "learning_rate": 1e-3, "gamma": 0.99, "hidden_sizes": [128, 128]},
    {"name": "reinforce_high_lr", "learning_rate": 5e-3, "gamma": 0.99, "hidden_sizes": [128, 128]},
    {"name": "reinforce_low_lr", "learning_rate": 1e-4, "gamma": 0.99, "hidden_sizes": [128, 128]},
    {"name": "reinforce_deep", "learning_rate": 1e-3, "gamma": 0.99, "hidden_sizes": [256, 256, 128]},
    {"name": "reinforce_wide", "learning_rate": 1e-3, "gamma": 0.99, "hidden_sizes": [256, 256]},
    {"name": "reinforce_high_gamma", "learning_rate": 1e-3, "gamma": 0.995, "hidden_sizes": [128, 128]},
    {"name": "reinforce_low_gamma", "learning_rate": 1e-3, "gamma": 0.95, "hidden_sizes": [128, 128]},
    {"name": "reinforce_small", "learning_rate": 1e-3, "gamma": 0.99, "hidden_sizes": [64, 64]},
    {"name": "reinforce_large", "learning_rate": 5e-4, "gamma": 0.995, "hidden_sizes": [256, 256]},
    {"name": "reinforce_aggressive", "learning_rate": 5e-3, "gamma": 0.995, "hidden_sizes": [256, 256, 128]},
]

def train_reinforce_single(config, num_episodes=1000, max_steps=100):
    """Train REINFORCE with single config"""
    print(f"\n{'='*80}\nTraining REINFORCE: {config['name']}\n{'='*80}\n")
    
    env = AgroTrackEnv(max_steps=max_steps)
    agent = REINFORCE(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                      hidden_sizes=config['hidden_sizes'], learning_rate=config['learning_rate'],
                      gamma=config['gamma'], device='cpu')
    
    episode_rewards = []
    running_reward = deque(maxlen=100)
    
    start_time = datetime.now()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.store_reward(reward)
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        loss = agent.update()
        episode_rewards.append(episode_reward)
        running_reward.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(list(running_reward))
            print(f"Episode {episode + 1:4d} | Reward: {episode_reward:7.2f} | "
                  f"Avg (100): {avg_reward:7.2f} | Loss: {loss:7.4f}")
            
            checkpoint_path = f"models/reinforce/{config['name']}/checkpoint_ep{episode+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            agent.save(checkpoint_path)
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    final_path = f"models/reinforce/{config['name']}/final_model.pth"
    agent.save(final_path)
    
    eval_results = evaluate_reinforce(agent, env, n_episodes=20)
    env.close()
    
    return {
        "config_name": config['name'], "algorithm": "REINFORCE",
        "training_time": training_time, "num_episodes": num_episodes,
        **eval_results
    }

def run_reinforce_sweep(num_episodes=1000):
    """Run REINFORCE sweep"""
    print("\n" + "="*80 + "\nREINFORCE HYPERPARAMETER SWEEP\n" + "="*80)
    all_results = []
    
    for i, config in enumerate(REINFORCE_CONFIGS):
        print(f"\n{'#'*80}\n# REINFORCE Config {i+1}/{len(REINFORCE_CONFIGS)}\n{'#'*80}\n")
        try:
            results = train_reinforce_single(config, num_episodes=num_episodes)
            all_results.append(results)
            pd.DataFrame(all_results).to_csv("results/reinforce_sweep_results.csv", index=False)
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()
    
    if all_results:
        df = pd.DataFrame(all_results).sort_values("mean_reward", ascending=False)
        df.to_csv("results/reinforce_sweep_results_final.csv", index=False)
        
        with open("models/reinforce/best_model_info.json", "w") as f:
            json.dump({
                "best_config": df.iloc[0]['config_name'],
                "mean_reward": float(df.iloc[0]['mean_reward']),
                "std_reward": float(df.iloc[0]['std_reward']),
                "mean_efficiency": float(df.iloc[0]['mean_efficiency']),
                "model_path": f"models/reinforce/{df.iloc[0]['config_name']}/final_model.pth"
            }, f, indent=2)
    
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--single", type=str, default=None)
    args = parser.parse_args()
    
    if args.single:
        config = next((c for c in REINFORCE_CONFIGS if c['name'] == args.single), None)
        if config: train_reinforce_single(config, num_episodes=args.episodes)
    else:
        run_reinforce_sweep(num_episodes=args.episodes)