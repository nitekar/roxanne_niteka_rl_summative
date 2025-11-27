import sys
import os
import json
import time
import argparse
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
import torch

from environment.custom_env import AgroTrackEnv
from training.reinforce_training import REINFORCE

def load_best_model(algorithm=None):
    """Load the best performing model"""
    
    if algorithm is None:
        # Automatically select best algorithm based on saved results
        print("Determining best algorithm...")
        
        best_reward = -float('inf')
        best_algo = None
        best_info = None
        
        for algo in ['dqn', 'ppo', 'a2c', 'reinforce']:
            try:
                with open(f"models/{algo}/best_model_info.json", "r") as f:
                    info = json.load(f)
                    if info['mean_reward'] > best_reward:
                        best_reward = info['mean_reward']
                        best_algo = algo
                        best_info = info
            except:
                continue
        
        if best_algo is None:
            raise FileNotFoundError("No trained models found. Please train models first.")
        
        algorithm = best_algo
        print(f" Best algorithm: {algorithm.upper()} (reward: {best_reward:.2f})")
    else:
        algorithm = algorithm.lower()
        with open(f"models/{algorithm}/best_model_info.json", "r") as f:
            best_info = json.load(f)
    
    # Load model
    print(f"\nLoading {algorithm.upper()} model...")
    print(f"Configuration: {best_info['best_config']}")
    print(f"Mean Reward: {best_info['mean_reward']:.2f} ± {best_info['std_reward']:.2f}")
    print(f"Efficiency: {best_info['mean_efficiency']:.2f}%\n")
    
    if algorithm == 'reinforce':
        # Load REINFORCE
        env_temp = AgroTrackEnv()
        model = REINFORCE(
            state_dim=env_temp.observation_space.shape[0],
            action_dim=env_temp.action_space.n,
            hidden_sizes=[128, 128]
        )
        model.load(best_info['model_path'])
        env_temp.close()
        model_type = 'reinforce'
    else:
        # Load stable-baselines3 model
        if algorithm == 'dqn':
            model = DQN.load(best_info['model_path'])
        elif algorithm == 'ppo':
            model = PPO.load(best_info['model_path'])
        elif algorithm == 'a2c':
            model = A2C.load(best_info['model_path'])
        model_type = 'sb3'
    
    return model, model_type, algorithm.upper()

def run_episode(env, model, model_type, render_delay=0.1, verbose=True):
    """Run a single episode with the model"""
    
    observation, info = env.reset()
    episode_reward = 0
    done = False
    step = 0
    
    # Full action names for the AgroTrack environment (8 actions)
    action_names = [
        "Monitor", "Basic Preservation", "Transport to Market", "Reuse/Donate",
        "Advanced Preservation", "Emergency Transport", "Compost", "Process Products"
    ]
    # Ensure we track exactly the number of actions in the env
    n_actions = env.action_space.n
    if len(action_names) < n_actions:
        # Fallback generic names if env has more actions
        action_names = [f"Action_{i}" for i in range(n_actions)]
    actions_taken = {i: 0 for i in range(n_actions)}
    
    if verbose:
        print("\nStarting episode...")
        print(f"Initial freshness: {info['avg_freshness']:.2f}")
        print(f"Initial inventory: {info['total_inventory']:.2f}\n")
    
    while not done:
        # Get action
        if model_type == 'sb3':
            action, _ = model.predict(observation, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.item()
        else:  # REINFORCE
            with torch.no_grad():
                state_tensor = torch.FloatTensor(observation).unsqueeze(0)
                probs = model.policy(state_tensor)
                action = torch.argmax(probs).item()
        
        actions_taken[action] += 1
        
        # Take step
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # Render
        env.render()
        
        # Print step info
        if verbose and (step % 10 == 0 or step < 5):
            act_name = action_names[action] if action < len(action_names) else f"Action_{action}"
            print(f"Step {step:3d}: {act_name:10s} | "
                f"Reward: {reward:7.2f} | "
                f"Cumulative: {episode_reward:8.2f} | "
                f"Freshness: {info['avg_freshness']:.2f}")
        
        step += 1
        done = terminated or truncated
        
        # Delay for visualization
        time.sleep(render_delay)
    
    return episode_reward, info, actions_taken, step

def print_episode_summary(algo_name, episode_num, episode_reward, info, actions_taken, steps):
    """Print episode summary"""
    
    # Derive action names; default to AgroTrack action labels if possible
    default_names = [
        "Monitor", "Basic Preservation", "Transport to Market", "Reuse/Donate",
        "Advanced Preservation", "Emergency Transport", "Compost", "Process Products"
    ]
    n_actions = len(actions_taken)
    action_names = default_names[:n_actions] if len(default_names) >= n_actions else [f"Action_{i}" for i in range(n_actions)]
    
    print("\n" + "="*80)
    print(f"EPISODE {episode_num} SUMMARY - {algo_name}")
    print("="*80)
    print(f"\nTotal Steps: {steps}")
    print(f"Total Reward: {episode_reward:.2f}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Food Saved:    {info.get('food_saved', 0.0):.2f}")
    print(f"  Food Spoiled:  {info.get('food_spoiled', 0.0):.2f}")
    print(f"  Food Reused:   {info.get('food_reused', 0.0):.2f}")
    print(f"  Revenue:       ${info.get('revenue', 0.0):.2f}")
    
    if info['food_saved'] + info['food_spoiled'] > 0:
        efficiency = (info['food_saved'] / (info['food_saved'] + info['food_spoiled'])) * 100
        print(f"  Efficiency:    {efficiency:.1f}%")
    
    print(f"\nFinal State:")
    print(f"  Avg Freshness: {info.get('avg_freshness', info.get('avg_quality', 0.0)):.2f}")
    print(f"  Total Inventory: {info.get('total_inventory', 0.0):.2f}")
    print(f"  Spoilage Risk: {info.get('spoilage_risk', 0.0):.2f}")
    
    print(f"\nAction Distribution:")
    for i, name in enumerate(action_names):
        count = actions_taken.get(i, 0)
        percentage = (count / steps) * 100 if steps > 0 else 0
        print(f"  {name:20s}: {count:3d} times ({percentage:5.1f}%)")
    
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run AgroTrack with trained model")
    parser.add_argument("--algorithm", type=str, choices=["dqn", "ppo", "a2c", "reinforce"], 
                       default=None, help="Algorithm to use (auto-detect if not specified)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--delay", type=float, default=0.1, help="Render delay (seconds)")
    parser.add_argument("--no-render", action="store_true", help="Run without rendering")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("AGROTRACK - POST-HARVEST LOSS PREVENTION")
    print("="*80)
    
    try:
        # Load model
        model, model_type, algo_name = load_best_model(args.algorithm)
        
        # Create environment
        render_mode = None if args.no_render else "human"
        env = AgroTrackEnv(render_mode=render_mode, max_steps=args.max_steps)
        
        print(f"\n Environment created (max_steps={args.max_steps})")
        print(f" Rendering: {'Enabled' if not args.no_render else 'Disabled'}")
        print("\n" + "="*80)
        
        # Run episodes
        total_rewards = []
        total_efficiency = []
        
        for episode in range(args.episodes):
            episode_reward, info, actions_taken, steps = run_episode(
                env, model, model_type, 
                render_delay=args.delay,
                verbose=args.verbose
            )
            
            total_rewards.append(episode_reward)
            
            if info['food_saved'] + info['food_spoiled'] > 0:
                eff = (info['food_saved'] / (info['food_saved'] + info['food_spoiled'])) * 100
                total_efficiency.append(eff)
            
            print_episode_summary(algo_name, episode + 1, episode_reward, info, actions_taken, steps)
            
            if episode < args.episodes - 1:
                print("\nStarting next episode in 3 seconds...")
                time.sleep(3)
        
        # Final summary
        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        print(f"\nAlgorithm: {algo_name}")
        print(f"Episodes: {args.episodes}")
        print(f"\nAverage Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        if total_efficiency:
            print(f"Average Efficiency: {np.mean(total_efficiency):.2f}%")
        print("\n" + "="*80)
        
        # Cleanup
        env.close()
        
        print("\nExecution complete\n")
        
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("\nPlease train models first using:")
        print("  python training/dqn_training.py")
        print("  python training/pg_training.py")
        print("  python training/reinforce_training.py\n")
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user\n")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()