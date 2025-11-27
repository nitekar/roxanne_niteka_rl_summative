import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

from environment.custom_env import AgroTrackEnv

def run_random_agent_demo(num_episodes=3, max_steps=100, render_delay=0.2):
    """
    Run random agent with visualization
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render_delay: Delay between steps (seconds)
    """
    print("="*80)
    print("AgroTrack Random Agent Demonstration")
    print("="*80)
    print("\nEnvironment Info:")
    print("  Observation Space: 28 dimensions")
    print("  Action Space: 8 discrete actions")
    print("\nActions:")
    print("  0: Monitor only")
    print("  1: Basic Preservation")
    print("  2: Transport to Market")
    print("  3: Reuse/Donate")
    print("  4: Advanced Preservation")
    print("  5: Emergency Transport")
    print("  6: Compost")
    print("  7: Process Products")
    print("\n" + "="*80 + "\n")
    
    # Create environment with rendering (human display)
    env = AgroTrackEnv(render_mode="human", max_steps=max_steps)
    
    action_names = [
        'Monitor', 'Basic Preservation', 'Transport to Market',
        'Reuse/Donate', 'Advanced Preservation', 'Emergency Transport',
        'Compost', 'Process Products'
    ]
    
    for episode in range(num_episodes):
        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*80}\n")
        
        observation, info = env.reset()
        episode_reward = 0
        episode_actions = {i: 0 for i in range(8)}
        
        print(f"Initial State:")
        print(f"  Total Inventory: {info['total_inventory']:.2f}")
        print(f"  Average Quality: {info['avg_quality']:.2f}")
        print(f"  Episode Reward: {info['episode_reward']:.2f}")
        print(f"\nStarting episode...\n")
        
        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()
            episode_actions[action] += 1
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Render
            env.render()
            
            # Print step information
            if step % 10 == 0 or step < 5:
                print(f"Step {step:3d}: {action_names[action]:20s} | "
                      f"Reward={reward:7.2f} | "
                      f"Cumulative={episode_reward:8.2f} | "
                      f"Quality={info['avg_quality']:.2f} | "
                      f"Inventory={info['total_inventory']:.2f}")
            
            # Delay for visualization
            time.sleep(render_delay)
            
            if terminated or truncated:
                break
        
        # Episode summary
        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1} SUMMARY")
        print(f"{'='*80}")
        print(f"Total Steps: {step + 1}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"\nPerformance Metrics:")
        print(f"  Food Saved: {info['total_saved']:.2f}")
        print(f"  Food Wasted: {info['total_wasted']:.2f}")
        print(f"  Average Quality: {info['avg_quality']:.2f}")
        print(f"  Final Inventory: {info['total_inventory']:.2f}")
        
        if info['total_saved'] + info['total_wasted'] > 0:
            efficiency = (info['total_saved'] / (info['total_saved'] + info['total_wasted'])) * 100
            print(f"  Loss Prevention Efficiency: {efficiency:.1f}%")
        
        print(f"\nAction Distribution:")
        for action_id, count in episode_actions.items():
            percentage = (count / (step + 1)) * 100
            print(f"  {action_names[action_id]:20s}: {count:3d} times ({percentage:5.1f}%)")
        
        print(f"{'='*80}\n")
        
        if episode < num_episodes - 1:
            print("\nPreparing next episode...\n")
            time.sleep(2)
    
    # Clean up
    env.close()
    
    print("\n" + "="*80)
    print("Random Agent Demonstration Complete")
    print("="*80)
    print("\nKey Observations:")
    print("- Random actions lead to suboptimal performance")
    print("- RL training should improve decision-making")
    print("- Environment tracks comprehensive metrics")
    print("\nNext Steps:")
    print("- Train DQN, PPO, A2C, and REINFORCE agents")
    print("- Compare trained agents vs random baseline")
    print("="*80 + "\n")

def run_headless_demo(num_episodes=1, max_steps=50):
    """Run demo without rendering (for Colab)"""
    print("="*80)
    print("AgroTrack Headless Demonstration")
    print("="*80)
    print("\nRunning without visualization\n")
    
    env = AgroTrackEnv(max_steps=max_steps)
    
    action_names = [
        'Monitor', 'Basic Preservation', 'Transport to Market',
        'Reuse/Donate', 'Advanced Preservation', 'Emergency Transport',
        'Compost', 'Process Products'
    ]
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        for step in range(max_steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Food Saved: {info['total_saved']:.2f}")
        print(f"  Food Wasted: {info['total_wasted']:.2f}")
        if info['total_saved'] + info['total_wasted'] > 0:
            efficiency = (info['total_saved'] / (info['total_saved'] + info['total_wasted']) * 100)
            print(f"  Efficiency: {efficiency:.1f}%")
    
    env.close()
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AgroTrack Random Agent Demo")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--delay", type=float, default=0.2, help="Render delay (seconds)")
    parser.add_argument("--headless", action="store_true", help="Run without rendering")
    parser.add_argument("--save-frames", action="store_true", help="Save rendered frames as PNG images")
    parser.add_argument("--out-dir", type=str, default="results/frames", help="Output directory for saved frames")
    
    args = parser.parse_args()
    
    try:
        if args.save_frames:
            # When saving frames we use rgb_array render mode to capture images
            os.makedirs(args.out_dir, exist_ok=True)

            def run_and_save():
                env = AgroTrackEnv(render_mode="rgb_array", max_steps=args.steps)
                action_names = [
                    'Monitor', 'Basic Preservation', 'Transport to Market',
                    'Reuse/Donate', 'Advanced Preservation', 'Emergency Transport',
                    'Compost', 'Process Products'
                ]

                for episode in range(args.episodes):
                    observation, info = env.reset()
                    episode_reward = 0
                    for step in range(args.steps):
                        action = env.action_space.sample()
                        observation, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward

                        frame = env.render()
                        if frame is not None:
                            # frame is a numpy array (H,W,3). Save as PNG
                            fname = os.path.join(args.out_dir, f"ep{episode+1:02d}_step{step+1:03d}.png")
                            plt.imsave(fname, frame.astype('uint8'))

                        if terminated or truncated:
                            break

                env.close()

            run_and_save()
        elif args.headless:
            run_headless_demo(num_episodes=args.episodes, max_steps=args.steps)
        else:
            run_random_agent_demo(
                num_episodes=args.episodes,
                max_steps=args.steps,
                render_delay=args.delay
            )
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()