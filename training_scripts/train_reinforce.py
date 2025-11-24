"""
Training script for REINFORCE (Monte Carlo Policy Gradient) on AgroTrack environment
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from agrotrack_env import AgroTrackEnv
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    """
    Neural network for policy approximation in REINFORCE.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)


class REINFORCE:
    """
    REINFORCE (Monte Carlo Policy Gradient) algorithm implementation.
    """
    def __init__(self, env, learning_rate=1e-3, gamma=0.99):
        self.env = env
        self.gamma = gamma
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state, deterministic=False):
        """Select action using current policy."""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        
        if deterministic:
            action = torch.argmax(probs, dim=1).item()
        else:
            m = Categorical(probs)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
            action = action.item()
        
        return action
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm."""
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate discounted returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        
        # Normalize returns for better stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate policy loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear memory
        del self.saved_log_probs[:]
        del self.rewards[:]
        
        return policy_loss.item()
    
    def train(self, num_episodes=1000, save_path='../models/reinforce_agrotrack'):
        """Train REINFORCE agent."""
        episode_rewards = []
        episode_lengths = []
        losses = []
        
        print("=" * 50)
        print("Training REINFORCE on AgroTrack Environment")
        print("=" * 50)
        print(f"Number of episodes: {num_episodes}")
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            terminated = False
            truncated = False
            
            # Collect trajectory
            while not (terminated or truncated):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                self.rewards.append(reward)
                episode_reward += reward
                episode_length += 1
                
                state = next_state
            
            # Update policy
            loss = self.update_policy()
            losses.append(loss)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.2f}, "
                      f"Loss: {loss:.4f}")
        
        print("\nTraining completed!")
        
        # Save model
        os.makedirs(save_path, exist_ok=True)
        model_path = f"{save_path}/reinforce_final.pt"
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")
        
        # Plot training progress
        self.plot_training_progress(episode_rewards, episode_lengths, losses, save_path)
        
        return episode_rewards, episode_lengths
    
    def plot_training_progress(self, rewards, lengths, losses, save_path):
        """Plot and save training progress."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot rewards
        axes[0].plot(rewards, alpha=0.3, label='Episode Reward')
        window = 100
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(rewards)), moving_avg, 
                        label=f'{window}-Episode Moving Average', linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Rewards')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot episode lengths
        axes[1].plot(lengths, alpha=0.3, label='Episode Length')
        if len(lengths) >= window:
            moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(lengths)), moving_avg, 
                        label=f'{window}-Episode Moving Average', linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Length')
        axes[1].set_title('Episode Lengths')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot losses
        axes[2].plot(losses, alpha=0.5)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Policy Loss')
        axes[2].grid(True)
        
        plt.tight_layout()
        plot_path = f"{save_path}/training_progress.png"
        plt.savefig(plot_path)
        print(f"Training progress plot saved to {plot_path}")
        plt.close()
    
    def test(self, num_episodes=5):
        """Test the trained agent."""
        env = AgroTrackEnv(render_mode="human")
        episode_rewards = []
        episode_lengths = []
        
        print("\nTesting trained model...")
        
        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            terminated = False
            truncated = False
            
            print(f"\n--- Episode {episode + 1} ---")
            
            while not (terminated or truncated):
                action = self.select_action(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
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


def train_reinforce(num_episodes=1000, save_path='../models/reinforce_agrotrack'):
    """
    Main function to train REINFORCE agent.
    
    Args:
        num_episodes: Number of episodes to train
        save_path: Path to save the trained model
    """
    env = AgroTrackEnv()
    agent = REINFORCE(env, learning_rate=1e-3, gamma=0.99)
    
    # Train agent
    episode_rewards, episode_lengths = agent.train(num_episodes, save_path)
    
    # Test agent
    agent.test(num_episodes=5)
    
    return agent


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Train REINFORCE
    agent = train_reinforce(num_episodes=1000)
