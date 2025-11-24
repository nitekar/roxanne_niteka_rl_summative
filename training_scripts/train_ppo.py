"""
Training script for PPO (Proximal Policy Optimization) on AgroTrack environment
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from agrotrack_env import AgroTrackEnv
import numpy as np


def train_ppo(total_timesteps=100000, save_path='../models/ppo_agrotrack'):
    """
    Train PPO agent on AgroTrack environment.
    
    Args:
        total_timesteps: Total number of timesteps to train
        save_path: Path to save the trained model
    """
    print("=" * 50)
    print("Training PPO on AgroTrack Environment")
    print("=" * 50)
    
    # Create environment
    env = AgroTrackEnv()
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = AgroTrackEnv()
    eval_env = Monitor(eval_env)
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix='ppo_checkpoint'
    )
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=f"{save_path}/tensorboard/"
    )
    
    print("\nStarting training...")
    print(f"Total timesteps: {total_timesteps}")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = f"{save_path}/ppo_final"
    model.save(final_model_path)
    print(f"\nTraining completed! Model saved to {final_model_path}")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_model(model, num_episodes=5)
    
    return model


def test_model(model, num_episodes=5):
    """
    Test the trained model.
    
    Args:
        model: Trained PPO model
        num_episodes: Number of episodes to test
    """
    env = AgroTrackEnv(render_mode="human")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if episode < 2:  # Render first 2 episodes
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length}")
        print(f"  Final Quality: {info['quality_index']:.2f}")
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print("=" * 50)


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Train PPO
    model = train_ppo(total_timesteps=100000)
