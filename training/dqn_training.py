import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import AgroTrackEnv

# Create directories
os.makedirs("models/dqn", exist_ok=True)
os.makedirs("logs/dqn", exist_ok=True)
os.makedirs("results", exist_ok=True)

# DQN Hyperparameter configurations (10 combinations)
HYPERPARAMETER_CONFIGS = [
    {
        "name": "dqn_baseline",
        "learning_rate": 1e-4,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    {
        "name": "dqn_high_lr",
        "learning_rate": 5e-4,
        "buffer_size": 50000,
        "learning_starts": 500,
        "batch_size": 128,
        "gamma": 0.99,
        "target_update_interval": 500,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    {
        "name": "dqn_deep_network",
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [256, 256, 128]},
    },
    {
        "name": "dqn_large_buffer",
        "learning_rate": 1e-4,
        "buffer_size": 200000,
        "learning_starts": 2000,
        "batch_size": 128,
        "gamma": 0.995,
        "target_update_interval": 2000,
        "exploration_fraction": 0.4,
        "exploration_final_eps": 0.02,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    {
        "name": "dqn_high_gamma",
        "learning_rate": 1e-4,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.995,
        "target_update_interval": 1000,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    {
        "name": "dqn_fast_update",
        "learning_rate": 2e-4,
        "buffer_size": 50000,
        "learning_starts": 500,
        "batch_size": 64,
        "gamma": 0.99,
        "target_update_interval": 500,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.1,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    {
        "name": "dqn_wide_network",
        "learning_rate": 1e-4,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [256, 256]},
    },
    {
        "name": "dqn_large_batch",
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "learning_starts": 2000,
        "batch_size": 256,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    {
        "name": "dqn_long_exploration",
        "learning_rate": 1e-4,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.5,
        "exploration_final_eps": 0.1,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    {
        "name": "dqn_aggressive",
        "learning_rate": 5e-4,
        "buffer_size": 100000,
        "learning_starts": 500,
        "batch_size": 128,
        "gamma": 0.995,
        "target_update_interval": 500,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.02,
        "policy_kwargs": {"net_arch": [256, 256]},
    },
]

def train_dqn_single_config(config, total_timesteps=100000, eval_freq=5000):
    """Train DQN with a single configuration"""
    
    print(f"\n{'='*80}")
    print(f"Training DQN: {config['name']}")
    print(f"{'='*80}")
    print("Configuration:")
    for key, value in config.items():
        if key != "name":
            print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Create environment
    env = AgroTrackEnv(max_steps=100)
    env = Monitor(env, f"logs/dqn/{config['name']}")
    
    # Create eval environment
    eval_env = AgroTrackEnv(max_steps=100)
    eval_env = Monitor(eval_env, f"logs/dqn/{config['name']}_eval")
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/dqn/{config['name']}",
        log_path=f"logs/dqn/{config['name']}",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"models/dqn/{config['name']}/checkpoints",
        name_prefix="dqn_checkpoint",
    )
    
    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        target_update_interval=config["target_update_interval"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        policy_kwargs=config["policy_kwargs"],
        verbose=1,
        tensorboard_log=f"logs/dqn/{config['name']}/tensorboard",
    )
    
    # Train
    print(f"\nStarting training for {total_timesteps} timesteps...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Save final model
    model.save(f"models/dqn/{config['name']}/final_model")
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate
    print("\nEvaluating trained model...")
    eval_results = evaluate_model(model, eval_env, n_episodes=20)
    
    # Cleanup
    env.close()
    eval_env.close()
    
    # Return results
    results = {
        "config_name": config['name'],
        "training_time": training_time,
        **eval_results,
        **{f"hp_{k}": v for k, v in config.items() if k != "name" and k != "policy_kwargs"}
    }
    
    return results

def evaluate_model(model, env, n_episodes=20):
    """Evaluate trained model"""
    
    episode_rewards = []
    episode_lengths = []
    food_saved_list = []
    food_wasted_list = []
    avg_quality_list = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(info['step'])
        food_saved_list.append(info['total_saved'])
        food_wasted_list.append(info['total_wasted'])
        avg_quality_list.append(info['avg_quality'])
    
    # Calculate efficiency
    efficiency_list = []
    for saved, wasted in zip(food_saved_list, food_wasted_list):
        if saved + wasted > 0:
            efficiency_list.append((saved / (saved + wasted)) * 100)
        else:
            efficiency_list.append(0)
    
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "mean_food_saved": np.mean(food_saved_list),
        "mean_food_wasted": np.mean(food_wasted_list),
        "mean_avg_quality": np.mean(avg_quality_list),
        "mean_efficiency": np.mean(efficiency_list),
    }
    
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Efficiency: {results['mean_efficiency']:.2f}%")
    print(f"  Mean Food Saved: {results['mean_food_saved']:.2f}")
    print(f"  Mean Food Wasted: {results['mean_food_wasted']:.2f}")
    
    return results

def run_hyperparameter_sweep(total_timesteps=100000, save_results=True):
    """Run complete hyperparameter sweep"""
    
    print("\n" + "="*80)
    print("DQN HYPERPARAMETER SWEEP")
    print("="*80)
    print(f"Total configurations: {len(HYPERPARAMETER_CONFIGS)}")
    print(f"Timesteps per config: {total_timesteps}")
    print(f"Environment: AgroTrack (8 actions, 28D obs)")
    print("="*80 + "\n")
    
    all_results = []
    
    for i, config in enumerate(HYPERPARAMETER_CONFIGS):
        print(f"\n{'#'*80}")
        print(f"# Configuration {i+1}/{len(HYPERPARAMETER_CONFIGS)}")
        print(f"{'#'*80}\n")
        
        try:
            results = train_dqn_single_config(config, total_timesteps)
            all_results.append(results)
            
            # Save intermediate results
            if save_results:
                df = pd.DataFrame(all_results)
                df.to_csv("results/dqn_sweep_results.csv", index=False)
                print(f"\nIntermediate results saved")
                
        except Exception as e:
            print(f"\nError training {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP COMPLETE")
    print("="*80)
    
    if all_results:
        df = pd.DataFrame(all_results)
        df_sorted = df.sort_values("mean_reward", ascending=False)
        
        print("\nTop 5 Configurations by Mean Reward:")
        print("-" * 80)
        for idx, row in df_sorted.head(5).iterrows():
            print(f"{row['config_name']:25s}: {row['mean_reward']:8.2f} ± {row['std_reward']:6.2f} "
                  f"(Efficiency: {row['mean_efficiency']:5.1f}%)")
        
        print("\n" + "="*80)
        
        # Save final results
        if save_results:
            df_sorted.to_csv("results/dqn_sweep_results_final.csv", index=False)
            
            # Save best model info
            best_config = df_sorted.iloc[0]['config_name']
            with open("models/dqn/best_model_info.json", "w") as f:
                json.dump({
                    "best_config": best_config,
                    "mean_reward": float(df_sorted.iloc[0]['mean_reward']),
                    "std_reward": float(df_sorted.iloc[0]['std_reward']),
                    "mean_efficiency": float(df_sorted.iloc[0]['mean_efficiency']),
                    "model_path": f"models/dqn/{best_config}/final_model.zip"
                }, f, indent=2)
            
            print(f"Results saved to results/dqn_sweep_results_final.csv")
            print(f"Best model info saved to models/dqn/best_model_info.json")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DQN Training for AgroTrack")
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps per config")
    parser.add_argument("--single", type=str, default=None, help="Train single config")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")
    
    args = parser.parse_args()
    
    if args.single:
        config = next((c for c in HYPERPARAMETER_CONFIGS if c['name'] == args.single), None)
        if config:
            train_dqn_single_config(config, total_timesteps=args.timesteps, eval_freq=args.eval_freq)
        else:
            print(f"Configuration '{args.single}' not found")
            print("Available configurations:")
            for c in HYPERPARAMETER_CONFIGS:
                print(f"  - {c['name']}")
    else:
        run_hyperparameter_sweep(total_timesteps=args.timesteps)
