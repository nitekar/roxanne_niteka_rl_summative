"""
Policy Gradient Training (PPO & A2C) for AgroTrack
Uses Stable-Baselines3 with 10 configs each
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import AgroTrackEnv

# Create directories
os.makedirs("models/ppo", exist_ok=True)
os.makedirs("models/a2c", exist_ok=True)
os.makedirs("logs/ppo", exist_ok=True)
os.makedirs("logs/a2c", exist_ok=True)
os.makedirs("results", exist_ok=True)

# PPO Configs (10)
PPO_CONFIGS = [
    {"name": "ppo_baseline", "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "ppo_high_lr", "learning_rate": 1e-3, "n_steps": 2048, "batch_size": 128, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "ppo_deep", "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "policy_kwargs": {"net_arch": [256, 256, 128]}},
    {"name": "ppo_long_horizon", "learning_rate": 3e-4, "n_steps": 4096, "batch_size": 128, "n_epochs": 15, "gamma": 0.995, "gae_lambda": 0.98, "clip_range": 0.2, "ent_coef": 0.0, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "ppo_entropy", "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.05, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "ppo_large_batch", "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 256, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "ppo_many_epochs", "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 20, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "ppo_wide", "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "policy_kwargs": {"net_arch": [256, 256]}},
    {"name": "ppo_tight_clip", "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.1, "ent_coef": 0.0, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "ppo_aggressive", "learning_rate": 1e-3, "n_steps": 4096, "batch_size": 256, "n_epochs": 15, "gamma": 0.995, "gae_lambda": 0.98, "clip_range": 0.3, "ent_coef": 0.01, "policy_kwargs": {"net_arch": [256, 256]}},
]

# A2C Configs (10)
A2C_CONFIGS = [
    {"name": "a2c_baseline", "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.0, "vf_coef": 0.5, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "a2c_high_lr", "learning_rate": 1e-3, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.01, "vf_coef": 0.5, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "a2c_deep", "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.0, "vf_coef": 0.5, "policy_kwargs": {"net_arch": [256, 256, 128]}},
    {"name": "a2c_long_rollout", "learning_rate": 7e-4, "n_steps": 20, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.0, "vf_coef": 0.5, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "a2c_entropy", "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.05, "vf_coef": 0.5, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "a2c_high_vf", "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.0, "vf_coef": 1.0, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "a2c_wide", "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.0, "vf_coef": 0.5, "policy_kwargs": {"net_arch": [256, 256]}},
    {"name": "a2c_medium_rollout", "learning_rate": 7e-4, "n_steps": 10, "gamma": 0.99, "gae_lambda": 0.98, "ent_coef": 0.0, "vf_coef": 0.5, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "a2c_balanced", "learning_rate": 5e-4, "n_steps": 10, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.7, "policy_kwargs": {"net_arch": [128, 128]}},
    {"name": "a2c_aggressive", "learning_rate": 1e-3, "n_steps": 20, "gamma": 0.995, "gae_lambda": 0.98, "ent_coef": 0.01, "vf_coef": 0.8, "policy_kwargs": {"net_arch": [256, 256]}},
]

def evaluate_model(model, env, n_episodes=20):
    """Evaluate model"""
    episode_rewards = []
    food_saved_list = []
    food_wasted_list = []
    avg_quality_list = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        food_saved_list.append(info['total_saved'])
        food_wasted_list.append(info['total_wasted'])
        avg_quality_list.append(info['avg_quality'])
    
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
        "mean_avg_quality": np.mean(avg_quality_list),
        "mean_efficiency": np.mean(efficiency_list),
    }

def train_ppo_single(config, total_timesteps=100000):
    """Train single PPO config"""
    print(f"\n{'='*80}\nTraining PPO: {config['name']}\n{'='*80}\n")
    
    env = AgroTrackEnv(max_steps=100)
    env = Monitor(env, f"logs/ppo/{config['name']}")
    
    eval_env = AgroTrackEnv(max_steps=100)
    eval_env = Monitor(eval_env, f"logs/ppo/{config['name']}_eval")
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"models/ppo/{config['name']}", 
                                  log_path=f"logs/ppo/{config['name']}", eval_freq=5000, 
                                  deterministic=True, n_eval_episodes=10)
    
    model = PPO("MlpPolicy", env, learning_rate=config["learning_rate"], n_steps=config["n_steps"], 
                batch_size=config["batch_size"], n_epochs=config["n_epochs"], gamma=config["gamma"], 
                gae_lambda=config["gae_lambda"], clip_range=config["clip_range"], 
                ent_coef=config["ent_coef"], policy_kwargs=config["policy_kwargs"], verbose=1,
                tensorboard_log=f"logs/ppo/{config['name']}/tensorboard")
    
    start_time = datetime.now()
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback], progress_bar=True)
    training_time = (datetime.now() - start_time).total_seconds()
    
    model.save(f"models/ppo/{config['name']}/final_model")
    
    eval_results = evaluate_model(model, eval_env, n_episodes=20)
    
    env.close()
    eval_env.close()
    
    return {"config_name": config['name'], "algorithm": "PPO", "training_time": training_time, **eval_results}

def train_a2c_single(config, total_timesteps=100000):
    """Train single A2C config"""
    print(f"\n{'='*80}\nTraining A2C: {config['name']}\n{'='*80}\n")
    
    env = AgroTrackEnv(max_steps=100)
    env = Monitor(env, f"logs/a2c/{config['name']}")
    
    eval_env = AgroTrackEnv(max_steps=100)
    eval_env = Monitor(eval_env, f"logs/a2c/{config['name']}_eval")
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"models/a2c/{config['name']}", 
                                  log_path=f"logs/a2c/{config['name']}", eval_freq=5000,
                                  deterministic=True, n_eval_episodes=10)
    
    model = A2C("MlpPolicy", env, learning_rate=config["learning_rate"], n_steps=config["n_steps"],
                gamma=config["gamma"], gae_lambda=config["gae_lambda"], ent_coef=config["ent_coef"],
                vf_coef=config["vf_coef"], policy_kwargs=config["policy_kwargs"], verbose=1,
                tensorboard_log=f"logs/a2c/{config['name']}/tensorboard")
    
    start_time = datetime.now()
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback], progress_bar=True)
    training_time = (datetime.now() - start_time).total_seconds()
    
    model.save(f"models/a2c/{config['name']}/final_model")
    
    eval_results = evaluate_model(model, eval_env, n_episodes=20)
    
    env.close()
    eval_env.close()
    
    return {"config_name": config['name'], "algorithm": "A2C", "training_time": training_time, **eval_results}

def run_ppo_sweep(total_timesteps=100000):
    """Run PPO sweep"""
    print("\n" + "="*80 + "\nPPO HYPERPARAMETER SWEEP\n" + "="*80)
    all_results = []
    
    for i, config in enumerate(PPO_CONFIGS):
        print(f"\n{'#'*80}\n# PPO Config {i+1}/{len(PPO_CONFIGS)}\n{'#'*80}\n")
        try:
            results = train_ppo_single(config, total_timesteps)
            all_results.append(results)
            pd.DataFrame(all_results).to_csv("results/ppo_sweep_results.csv", index=False)
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()
    
    if all_results:
        df = pd.DataFrame(all_results).sort_values("mean_reward", ascending=False)
        df.to_csv("results/ppo_sweep_results_final.csv", index=False)
        
        with open("models/ppo/best_model_info.json", "w") as f:
            json.dump({
                "best_config": df.iloc[0]['config_name'],
                "mean_reward": float(df.iloc[0]['mean_reward']),
                "std_reward": float(df.iloc[0]['std_reward']),
                "mean_efficiency": float(df.iloc[0]['mean_efficiency']),
                "model_path": f"models/ppo/{df.iloc[0]['config_name']}/final_model.zip"
            }, f, indent=2)
    
    return all_results

def run_a2c_sweep(total_timesteps=100000):
    """Run A2C sweep"""
    print("\n" + "="*80 + "\nA2C HYPERPARAMETER SWEEP\n" + "="*80)
    all_results = []
    
    for i, config in enumerate(A2C_CONFIGS):
        print(f"\n{'#'*80}\n# A2C Config {i+1}/{len(A2C_CONFIGS)}\n{'#'*80}\n")
        try:
            results = train_a2c_single(config, total_timesteps)
            all_results.append(results)
            pd.DataFrame(all_results).to_csv("results/a2c_sweep_results.csv", index=False)
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()
    
    if all_results:
        df = pd.DataFrame(all_results).sort_values("mean_reward", ascending=False)
        df.to_csv("results/a2c_sweep_results_final.csv", index=False)
        
        with open("models/a2c/best_model_info.json", "w") as f:
            json.dump({
                "best_config": df.iloc[0]['config_name'],
                "mean_reward": float(df.iloc[0]['mean_reward']),
                "std_reward": float(df.iloc[0]['std_reward']),
                "mean_efficiency": float(df.iloc[0]['mean_efficiency']),
                "model_path": f"models/a2c/{df.iloc[0]['config_name']}/final_model.zip"
            }, f, indent=2)
    
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["ppo", "a2c", "both"], default="both")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--single", type=str, default=None)
    args = parser.parse_args()
    
    if args.single:
        if args.algorithm in ["ppo", "both"]:
            config = next((c for c in PPO_CONFIGS if c['name'] == args.single), None)
            if config: train_ppo_single(config, args.timesteps)
        if args.algorithm in ["a2c", "both"]:
            config = next((c for c in A2C_CONFIGS if c['name'] == args.single), None)
            if config: train_a2c_single(config, args.timesteps)
    else:
        if args.algorithm in ["ppo", "both"]: run_ppo_sweep(args.timesteps)
        if args.algorithm in ["a2c", "both"]: run_a2c_sweep(args.timesteps)