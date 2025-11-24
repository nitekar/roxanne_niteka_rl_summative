"""
Demo script to showcase AgroTrack environment usage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agrotrack_env import AgroTrackEnv
import numpy as np


def demo_random_agent():
    """Demonstrate random agent interacting with environment."""
    print("=" * 60)
    print("AgroTrack Environment Demo - Random Agent")
    print("=" * 60)
    
    env = AgroTrackEnv(render_mode="human")
    
    # Run 3 episodes
    for episode in range(3):
        print(f"\n{'*' * 60}")
        print(f"EPISODE {episode + 1}")
        print('*' * 60)
        
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        print(f"\nInitial State:")
        env.render()
        
        while not (terminated or truncated):
            # Random action
            action = env.action_space.sample()
            
            action_names = [
                "No intervention",
                "Basic cooling",
                "Advanced cooling",
                "Humidity control",
                "Rush to market"
            ]
            
            print(f"\nStep {step + 1}: Taking action - {action_names[action]}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            print(f"  Reward: {reward:.2f}")
            env.render()
            
            if terminated:
                print("\n⚠️  Episode terminated!")
                if info['quality_index'] <= 10:
                    print("   Reason: Product spoiled")
                elif info.get('rush_to_market', False):
                    print("   Reason: Rushed to market")
                else:
                    print("   Reason: Time limit reached")
        
        print(f"\n{'=' * 60}")
        print(f"Episode {episode + 1} Summary:")
        print(f"  Total Steps: {step}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Final Quality: {info['quality_index']:.2f}%")
        print(f"  Total Cost: ${info['storage_cost']:.2f}")
        print(f"  Days Elapsed: {info['days_since_harvest']}")
        print('=' * 60)


def demo_greedy_quality_agent():
    """Demonstrate an agent that always tries to maximize quality."""
    print("\n\n" + "=" * 60)
    print("AgroTrack Environment Demo - Greedy Quality Agent")
    print("(Always chooses advanced cooling to preserve quality)")
    print("=" * 60)
    
    env = AgroTrackEnv(render_mode="human")
    
    obs, info = env.reset()
    episode_reward = 0
    terminated = False
    truncated = False
    step = 0
    
    print(f"\nInitial State:")
    env.render()
    
    while not (terminated or truncated):
        # Always use advanced cooling (action 2)
        action = 2
        
        print(f"\nStep {step + 1}: Taking action - Advanced cooling")
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step += 1
        
        print(f"  Reward: {reward:.2f}")
        env.render()
    
    print(f"\n{'=' * 60}")
    print(f"Episode Summary:")
    print(f"  Total Steps: {step}")
    print(f"  Total Reward: {episode_reward:.2f}")
    print(f"  Final Quality: {info['quality_index']:.2f}%")
    print(f"  Total Cost: ${info['storage_cost']:.2f}")
    print(f"  Days Elapsed: {info['days_since_harvest']}")
    print('=' * 60)


def demo_observation_space():
    """Demonstrate observation space structure."""
    print("\n\n" + "=" * 60)
    print("AgroTrack Environment - Observation Space Demo")
    print("=" * 60)
    
    env = AgroTrackEnv()
    
    print("\nObservation Space:")
    print(f"  Type: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Low bounds: {env.observation_space.low}")
    print(f"  High bounds: {env.observation_space.high}")
    
    print("\nObservation Features:")
    print("  [0] Temperature (°C): 0-40")
    print("  [1] Humidity (%): 0-100")
    print("  [2] Days since harvest: 0-30")
    print("  [3] Quality index: 0-100")
    print("  [4] Storage cost: 0-1000")
    print("  [5] Product quantity (kg): 0-1000")
    
    print("\nAction Space:")
    print(f"  Type: {env.action_space}")
    print(f"  Number of actions: {env.action_space.n}")
    
    print("\nAvailable Actions:")
    print("  [0] No intervention - Low cost, natural degradation")
    print("  [1] Basic cooling - Moderate cost, slows degradation")
    print("  [2] Advanced cooling - High cost, minimal degradation")
    print("  [3] Humidity control - Moderate cost, prevents moisture damage")
    print("  [4] Rush to market - High cost, saves remaining quality")
    
    # Sample a few observations
    print("\nSample Observations:")
    for i in range(3):
        obs, _ = env.reset()
        print(f"\n  Sample {i+1}:")
        print(f"    Temperature: {obs[0]:.2f}°C")
        print(f"    Humidity: {obs[1]:.2f}%")
        print(f"    Days: {obs[2]:.0f}")
        print(f"    Quality: {obs[3]:.2f}%")
        print(f"    Cost: ${obs[4]:.2f}")
        print(f"    Quantity: {obs[5]:.2f} kg")


if __name__ == "__main__":
    # Run all demos
    demo_observation_space()
    demo_random_agent()
    demo_greedy_quality_agent()
    
    print("\n\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Train models: python training_scripts/train_<model>.py")
    print("2. Compare models: python visualization/compare_models.py")
    print("3. See README.md for more information")
    print("=" * 60)
