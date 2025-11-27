"""
Model Comparison and Evaluation Script
Compare all trained models (DQN, PPO, A2C, REINFORCE) and generate comprehensive reports
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN, PPO, A2C
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import AgroTrackEnv
from training.reinforce_training import REINFORCE, PolicyNetwork

# Create results directory
os.makedirs("results/figures", exist_ok=True)

def load_best_models():
    """Load best models from all algorithms"""
    
    models = {}
    
    # Load DQN
    try:
        with open("models/dqn/best_model_info.json", "r") as f:
            dqn_info = json.load(f)
        models['DQN'] = {
            'model': DQN.load(dqn_info['model_path']),
            'info': dqn_info,
            'type': 'sb3'
        }
        print(f"✓ Loaded DQN: {dqn_info['best_config']}")
    except Exception as e:
        print(f"✗ Could not load DQN: {e}")
    
    # Load PPO
    try:
        with open("models/ppo/best_model_info.json", "r") as f:
            ppo_info = json.load(f)
        models['PPO'] = {
            'model': PPO.load(ppo_info['model_path']),
            'info': ppo_info,
            'type': 'sb3'
        }
        print(f"✓ Loaded PPO: {ppo_info['best_config']}")
    except Exception as e:
        print(f"✗ Could not load PPO: {e}")
    
    # Load A2C
    try:
        with open("models/a2c/best_model_info.json", "r") as f:
            a2c_info = json.load(f)
        models['A2C'] = {
            'model': A2C.load(a2c_info['model_path']),
            'info': a2c_info,
            'type': 'sb3'
        }
        print(f"✓ Loaded A2C: {a2c_info['best_config']}")
    except Exception as e:
        print(f"✗ Could not load A2C: {e}")
    
    # Load REINFORCE
    try:
        with open("models/reinforce/best_model_info.json", "r") as f:
            reinforce_info = json.load(f)
        
        # Reconstruct REINFORCE agent
        env_temp = AgroTrackEnv()
        agent = REINFORCE(
            state_dim=env_temp.observation_space.shape[0],
            action_dim=env_temp.action_space.n,
            hidden_sizes=[128, 128]  # Default, will be overridden by loaded weights
        )
        agent.load(reinforce_info['model_path'])
        env_temp.close()
        
        models['REINFORCE'] = {
            'model': agent,
            'info': reinforce_info,
            'type': 'reinforce'
        }
        print(f"✓ Loaded REINFORCE: {reinforce_info['best_config']}")
    except Exception as e:
        print(f"✗ Could not load REINFORCE: {e}")
    
    return models

def evaluate_single_model(model_name, model_data, env, n_episodes=50):
    """Evaluate a single model"""
    
    print(f"\nEvaluating {model_name}...")
    
    model = model_data['model']
    model_type = model_data['type']
    
    episode_rewards = []
    episode_lengths = []
    food_saved_list = []
    food_spoiled_list = []
    food_reused_list = []
    revenue_list = []
    efficiency_list = []
    action_distributions = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        # Track all actions according to environment action space
        actions_taken = {i: 0 for i in range(env.action_space.n)}
        
        while not done:
            # Get action based on model type
            if model_type == 'sb3':
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):
                    action = action.item()
            else:  # REINFORCE
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    probs = model.policy(state_tensor)
                    action = torch.argmax(probs).item()
            
            actions_taken[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        # Use environment-provided info keys (total_saved / total_wasted)
        episode_rewards.append(episode_reward)
        episode_lengths.append(info.get('step', 0))
        food_saved_list.append(info.get('total_saved', 0))
        food_spoiled_list.append(info.get('total_wasted', 0))
        # Environment doesn't track reused or revenue separately; set to 0 or reuse total_saved
        food_reused_list.append(0)
        revenue_list.append(0)

        saved = info.get('total_saved', 0)
        wasted = info.get('total_wasted', 0)
        if saved + wasted > 0:
            efficiency = (saved / (saved + wasted)) * 100
        else:
            efficiency = 0
        efficiency_list.append(efficiency)
        action_distributions.append(actions_taken)
    
    # Aggregate action distribution across all actions
    n_actions = env.action_space.n
    avg_action_dist = {i: np.mean([d[i] for d in action_distributions]) for i in range(n_actions)}
    
    results = {
        'algorithm': model_name,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'median_reward': np.median(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_food_saved': np.mean(food_saved_list),
        'std_food_saved': np.std(food_saved_list),
        'mean_food_spoiled': np.mean(food_spoiled_list),
        'std_food_spoiled': np.std(food_spoiled_list),
        'mean_food_reused': np.mean(food_reused_list),
        'std_food_reused': np.std(food_reused_list),
        'mean_revenue': np.mean(revenue_list),
        'std_revenue': np.std(revenue_list),
        'mean_efficiency': np.mean(efficiency_list),
        'std_efficiency': np.std(efficiency_list),
    }

    # Add action distribution per action (0..n-1) with readable keys
    action_names = [
        'Monitor', 'Basic Preservation', 'Transport to Market', 'Reuse/Donate',
        'Advanced Preservation', 'Emergency Transport', 'Compost', 'Process Products'
    ]
    # Ensure action_names length matches n_actions
    if len(action_names) < n_actions:
        action_names = [f"Action_{i}" for i in range(n_actions)]

    for i in range(n_actions):
        key = f"action_dist_{i}"
        results[key] = avg_action_dist.get(i, 0)
        # also add human-readable alias if available
        alias_key = f"action_name_{i}"
        results[alias_key] = action_names[i] if i < len(action_names) else f"Action_{i}"
    
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Efficiency: {results['mean_efficiency']:.2f}%")
    print(f"  Mean Food Saved: {results['mean_food_saved']:.2f}")
    print(f"  Mean Revenue: ${results['mean_revenue']:.2f}")
    
    return results, episode_rewards, efficiency_list

def create_comparison_plots(all_results, all_rewards, all_efficiencies):
    """Create comparison visualizations"""
    
    sns.set_style("whitegrid")
    
    # 1. Mean Reward Comparison
    plt.figure(figsize=(12, 6))
    algorithms = [r['algorithm'] for r in all_results]
    means = [r['mean_reward'] for r in all_results]
    stds = [r['std_reward'] for r in all_results]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = plt.bar(algorithms, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    plt.ylabel('Mean Reward', fontsize=12)
    plt.title('Algorithm Performance Comparison - Mean Reward', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/reward_comparison.png', dpi=300)
    plt.close()
    
    # 2. Efficiency Comparison
    plt.figure(figsize=(12, 6))
    efficiency_means = [r['mean_efficiency'] for r in all_results]
    efficiency_stds = [r['std_efficiency'] for r in all_results]
    
    bars = plt.bar(algorithms, efficiency_means, yerr=efficiency_stds, capsize=5, color=colors, alpha=0.7)
    plt.ylabel('Loss Prevention Efficiency (%)', fontsize=12)
    plt.title('Algorithm Performance - Loss Prevention Efficiency', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('results/figures/efficiency_comparison.png', dpi=300)
    plt.close()
    
    # 3. Box plots for reward distribution
    plt.figure(figsize=(12, 6))
    reward_data = [all_rewards[alg] for alg in algorithms]
    bp = plt.boxplot(reward_data, labels=algorithms, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title('Reward Distribution Across Algorithms', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/reward_distribution.png', dpi=300)
    plt.close()
    
    # 4. Multiple metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Food Saved
    axes[0, 0].bar(algorithms, [r['mean_food_saved'] for r in all_results], color=colors, alpha=0.7)
    axes[0, 0].set_ylabel('Mean Food Saved')
    axes[0, 0].set_title('Food Saved Comparison')
    
    # Food Spoiled
    axes[0, 1].bar(algorithms, [r['mean_food_spoiled'] for r in all_results], color=colors, alpha=0.7)
    axes[0, 1].set_ylabel('Mean Food Spoiled')
    axes[0, 1].set_title('Food Spoiled Comparison')
    
    # Revenue
    axes[1, 0].bar(algorithms, [r['mean_revenue'] for r in all_results], color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('Mean Revenue ($)')
    axes[1, 0].set_title('Revenue Generation Comparison')
    
    # Food Reused
    axes[1, 1].bar(algorithms, [r['mean_food_reused'] for r in all_results], color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Mean Food Reused')
    axes[1, 1].set_title('Food Reuse Comparison')
    
    plt.tight_layout()
    plt.savefig('results/figures/metrics_comparison.png', dpi=300)
    plt.close()
    
    # 5. Action distribution heatmap (support variable number of actions)
    # Determine how many actions are present from the first result
    n_actions = 0
    if all_results:
        keys = [k for k in all_results[0].keys() if k.startswith('action_dist_')]
        n_actions = len(keys)

    if n_actions > 0:
        # Use readable names if present
        action_names = []
        for i in range(n_actions):
            name_key = f'action_name_{i}'
            action_names.append(all_results[0].get(name_key, f'Action_{i}'))

        action_data = np.array([[r.get(f'action_dist_{i}', 0) for i in range(n_actions)] for r in all_results])

        plt.figure(figsize=(max(8, n_actions * 1.2), 6))
        sns.heatmap(action_data, annot=True, fmt='.1f', cmap='YlOrRd',
                    xticklabels=action_names, yticklabels=algorithms,
                    cbar_kws={'label': 'Average Actions per Episode'})
        plt.title('Action Distribution Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/figures/action_distribution.png', dpi=300)
        plt.close()
    
    print("\n Plots saved to results/figures/")

def generate_comparison_report(all_results):
    """Generate detailed comparison report"""
    
    report = []
    report.append("="*80)
    report.append("AGROTRACK RL ALGORITHMS COMPARISON REPORT")
    report.append("="*80)
    report.append("")
    report.append("Performance Summary")
    report.append("-"*80)
    report.append("")
    
    # Create summary table
    df = pd.DataFrame(all_results)
    
    # Sort by mean reward
    df_sorted = df.sort_values('mean_reward', ascending=False)
    
    report.append("Rankings by Mean Reward:")
    report.append("")
    for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
        report.append(f"{idx}. {row['algorithm']:12s} - {row['mean_reward']:8.2f} ± {row['std_reward']:6.2f}")
    
    report.append("")
    report.append("="*80)
    report.append("Detailed Metrics")
    report.append("="*80)
    report.append("")
    
    for _, row in df_sorted.iterrows():
        report.append(f"Algorithm: {row['algorithm']}")
        report.append("-"*80)
        report.append(f"  Reward:          {row['mean_reward']:8.2f} ± {row['std_reward']:6.2f}")
        report.append(f"  Efficiency:      {row['mean_efficiency']:7.2f}% ± {row['std_efficiency']:5.2f}%")
        report.append(f"  Food Saved:      {row['mean_food_saved']:8.2f} ± {row['std_food_saved']:6.2f}")
        report.append(f"  Food Spoiled:    {row['mean_food_spoiled']:8.2f} ± {row['std_food_spoiled']:6.2f}")
        report.append(f"  Food Reused:     {row['mean_food_reused']:8.2f} ± {row['std_food_reused']:6.2f}")
        report.append(f"  Revenue:        ${row['mean_revenue']:8.2f} ± ${row['std_revenue']:6.2f}")
        report.append("")
        report.append(f"  Action Distribution:")
        # Print all action distributions dynamically
        action_keys = [k for k in row.index if str(k).startswith('action_dist_')]
        action_keys = sorted(action_keys, key=lambda x: int(x.split('_')[-1]))
        for k in action_keys:
            idx = int(k.split('_')[-1])
            name = row.get(f'action_name_{idx}', f'Action_{idx}')
            report.append(f"    {name:12s}: {row[k]:6.2f}")
        report.append("")
    
    report.append("="*80)
    report.append("Key Insights")
    report.append("="*80)
    report.append("")
    
    best_algo = df_sorted.iloc[0]['algorithm']
    best_reward = df_sorted.iloc[0]['mean_reward']
    best_efficiency = df_sorted.iloc[0]['mean_efficiency']
    
    report.append(f"• Best performing algorithm: {best_algo}")
    report.append(f"• Achieved mean reward: {best_reward:.2f}")
    report.append(f"• Loss prevention efficiency: {best_efficiency:.2f}%")
    report.append("")
    
    # Find best in each category
    best_saver = df.loc[df['mean_food_saved'].idxmax(), 'algorithm']
    best_revenue = df.loc[df['mean_revenue'].idxmax(), 'algorithm']
    best_reuser = df.loc[df['mean_food_reused'].idxmax(), 'algorithm']
    
    report.append(f"• Best at saving food: {best_saver}")
    report.append(f"• Best at generating revenue: {best_revenue}")
    report.append(f"• Best at food reuse: {best_reuser}")
    report.append("")
    report.append("="*80)
    
    # Save report
    with open("results/comparison_report.txt", "w") as f:
        f.write("\n".join(report))
    
    # Print to console
    print("\n" + "\n".join(report))

def run_comprehensive_evaluation(n_episodes=50):
    """Run comprehensive evaluation of all models"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    print(f"\nEvaluating each model over {n_episodes} episodes...")
    print("="*80 + "\n")
    
    # Load models
    print("Loading trained models...")
    print("-"*80)
    models = load_best_models()
    
    if not models:
        print("\n No models found. Please train models first.")
        return
    
    print(f"\n Loaded {len(models)} models\n")
    print("="*80)
    
    # Create evaluation environment
    env = AgroTrackEnv(max_steps=100)
    
    # Evaluate each model
    all_results = []
    all_rewards = {}
    all_efficiencies = {}
    
    for model_name, model_data in models.items():
        results, rewards, efficiencies = evaluate_single_model(model_name, model_data, env, n_episodes)
        all_results.append(results)
        all_rewards[model_name] = rewards
        all_efficiencies[model_name] = efficiencies
    
    env.close()
    
    # Save consolidated results
    df = pd.DataFrame(all_results)
    df.to_csv("results/consolidated_evaluation.csv", index=False)
    print("\n Results saved to results/consolidated_evaluation.csv")
    
    # Create visualizations
    print("\nGenerating comparison plots...")
    create_comparison_plots(all_results, all_rewards, all_efficiencies)
    
    # Generate report
    print("\nGenerating comparison report...")
    generate_comparison_report(all_results)
    print(" Report saved to results/comparison_report.txt")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare all trained models")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    run_comprehensive_evaluation(n_episodes=args.episodes)