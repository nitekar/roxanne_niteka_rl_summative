from environment.custom_env import AgroTrackEnv
import numpy as np

print("="*80)
print("Testing AgroTrack Environment")
print("="*80)

# Create environment
env = AgroTrackEnv()
print("\n Environment created successfully")

# Check spaces
print(f" Observation space: {env.observation_space}")
print(f" Action space: {env.action_space}")
print(f" Observation shape: {env.observation_space.shape}")
print(f" Number of actions: {env.action_space.n}")

# Reset
obs, info = env.reset()
print(f"\n Environment reset successful")
print(f" Observation shape: {obs.shape}")
print(f" Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
print(f"\nInitial Info:")
for key, value in info.items():
    print(f"  {key}: {value}")

# Take some steps
print("\n" + "="*80)
print("Taking 10 random steps...")
print("="*80 + "\n")

action_names = [
    'Monitor', 'Basic Preservation', 'Transport to Market',
    'Reuse/Donate', 'Advanced Preservation', 'Emergency Transport',
    'Compost', 'Process Products'
]

total_reward = 0
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    print(f"Step {i+1:2d}: {action_names[action]:20s} | "
          f"Reward: {reward:7.2f} | "
          f"Quality: {info['avg_quality']:.2f} | "
          f"Inventory: {info['total_inventory']:.2f}")
    
    if terminated or truncated:
        print(f"\nEpisode terminated at step {i+1}")
        break

print(f"\n Total reward: {total_reward:.2f}")
print(f" Food saved: {info['total_saved']:.2f}")
print(f" Food wasted: {info['total_wasted']:.2f}")

env.close()
print("\n" + "="*80)
print(" All tests passed!")
print("="*80)