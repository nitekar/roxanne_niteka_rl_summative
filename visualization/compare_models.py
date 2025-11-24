"""
Visualization script for comparing trained RL models on AgroTrack environment
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
import torch
from agrotrack_env import AgroTrackEnv
from training_scripts.train_reinforce import PolicyNetwork


def load_models(models_dir='../models'):
    """Load all trained models."""
    models = {}
    
    # Load DQN
    dqn_path = f"{models_dir}/dqn_agrotrack/dqn_final.zip"
    if os.path.exists(dqn_path):
        models['DQN'] = DQN.load(dqn_path)
        print(f"Loaded DQN model from {dqn_path}")
    else:
        print(f"DQN model not found at {dqn_path}")
    
    # Load PPO
    ppo_path = f"{models_dir}/ppo_agrotrack/ppo_final.zip"
    if os.path.exists(ppo_path):
        models['PPO'] = PPO.load(ppo_path)
        print(f"Loaded PPO model from {ppo_path}")
    else:
        print(f"PPO model not found at {ppo_path}")
    
    # Load A2C
    a2c_path = f"{models_dir}/a2c_agrotrack/a2c_final.zip"
    if os.path.exists(a2c_path):
        models['A2C'] = A2C.load(a2c_path)
        print(f"Loaded A2C model from {a2c_path}")
    else:
        print(f"A2C model not found at {a2c_path}")
    
    # Load REINFORCE
    reinforce_path = f"{models_dir}/reinforce_agrotrack/reinforce_final.pt"
    if os.path.exists(reinforce_path):
        checkpoint = torch.load(reinforce_path)
        env = AgroTrackEnv()
        policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()
        models['REINFORCE'] = policy
        print(f"Loaded REINFORCE model from {reinforce_path}")
    else:
        print(f"REINFORCE model not found at {reinforce_path}")
    
    return models


def evaluate_model(model, model_name, num_episodes=20):
    """Evaluate a single model."""
    env = AgroTrackEnv()
    
    episode_rewards = []
    episode_lengths = []
    final_qualities = []
    total_costs = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            if model_name == 'REINFORCE':
                # REINFORCE uses PyTorch
                state = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    probs = model(state)
                action = torch.argmax(probs, dim=1).item()
            else:
                # Stable Baselines3 models
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(info['days_since_harvest'])
        final_qualities.append(info['quality_index'])
        total_costs.append(info['storage_cost'])
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'qualities': final_qualities,
        'costs': total_costs
    }


def compare_models(models, num_episodes=20, save_path='../visualization'):
    """Compare all models and create visualizations."""
    print("\n" + "=" * 50)
    print("Evaluating Models")
    print("=" * 50)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        results[model_name] = evaluate_model(model, model_name, num_episodes)
    
    # Create comparison plots
    os.makedirs(save_path, exist_ok=True)
    plot_comparisons(results, save_path)
    
    # Print summary statistics
    print_summary(results)
    
    return results


def plot_comparisons(results, save_path):
    """Create comparison plots for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    model_names = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    for i, model_name in enumerate(model_names):
        rewards = results[model_name]['rewards']
        ax.plot(rewards, label=model_name, color=colors[i], alpha=0.7, marker='o', markersize=3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Box plot of rewards
    ax = axes[0, 1]
    reward_data = [results[model_name]['rewards'] for model_name in model_names]
    bp = ax.boxplot(reward_data, labels=model_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Final Quality Index
    ax = axes[1, 0]
    quality_data = [results[model_name]['qualities'] for model_name in model_names]
    bp = ax.boxplot(quality_data, labels=model_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Final Quality Index')
    ax.set_title('Final Product Quality')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Storage Costs
    ax = axes[1, 1]
    cost_data = [results[model_name]['costs'] for model_name in model_names]
    bp = ax.boxplot(cost_data, labels=model_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Total Storage Cost ($)')
    ax.set_title('Storage Costs')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = f"{save_path}/model_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nComparison plot saved to {plot_path}")
    plt.close()
    
    # Create bar chart for average metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Average rewards
    ax = axes[0]
    avg_rewards = [np.mean(results[model_name]['rewards']) for model_name in model_names]
    bars = ax.bar(model_names, avg_rewards, color=colors, alpha=0.7)
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Episode Reward')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    # Average quality
    ax = axes[1]
    avg_quality = [np.mean(results[model_name]['qualities']) for model_name in model_names]
    bars = ax.bar(model_names, avg_quality, color=colors, alpha=0.7)
    ax.set_ylabel('Average Quality Index')
    ax.set_title('Average Final Quality')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    # Average costs
    ax = axes[2]
    avg_costs = [np.mean(results[model_name]['costs']) for model_name in model_names]
    bars = ax.bar(model_names, avg_costs, color=colors, alpha=0.7)
    ax.set_ylabel('Average Storage Cost ($)')
    ax.set_title('Average Storage Cost')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    bar_path = f"{save_path}/average_metrics.png"
    plt.savefig(bar_path, dpi=300)
    print(f"Average metrics plot saved to {bar_path}")
    plt.close()


def print_summary(results):
    """Print summary statistics for all models."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':<15} {'Avg Reward':<15} {'Avg Quality':<15} {'Avg Cost':<15}")
    print("-" * 70)
    
    for model_name in results.keys():
        avg_reward = np.mean(results[model_name]['rewards'])
        std_reward = np.std(results[model_name]['rewards'])
        avg_quality = np.mean(results[model_name]['qualities'])
        std_quality = np.std(results[model_name]['qualities'])
        avg_cost = np.mean(results[model_name]['costs'])
        std_cost = np.std(results[model_name]['costs'])
        
        print(f"{model_name:<15} "
              f"{avg_reward:>6.2f}±{std_reward:<5.2f} "
              f"{avg_quality:>6.2f}±{std_quality:<5.2f} "
              f"{avg_cost:>6.2f}±{std_cost:<5.2f}")
    
    print("=" * 70)
    
    # Find best model for each metric
    best_reward_model = max(results.keys(), key=lambda x: np.mean(results[x]['rewards']))
    best_quality_model = max(results.keys(), key=lambda x: np.mean(results[x]['qualities']))
    best_cost_model = min(results.keys(), key=lambda x: np.mean(results[x]['costs']))
    
    print(f"\nBest Performance:")
    print(f"  Highest Reward: {best_reward_model}")
    print(f"  Highest Quality: {best_quality_model}")
    print(f"  Lowest Cost: {best_cost_model}")
    print("=" * 70)


if __name__ == "__main__":
    # Load all trained models
    models = load_models()
    
    if not models:
        print("\nNo trained models found. Please train models first using the training scripts.")
        sys.exit(1)
    
    # Compare models
    results = compare_models(models, num_episodes=20)
