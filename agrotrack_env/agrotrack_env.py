"""
AgroTrack Environment - Custom Gymnasium Environment for Post-Harvest Food Loss Prevention

This environment models agricultural dynamics for post-harvest food loss prevention.
The agent must make decisions about storage conditions, handling, and transportation
to minimize food loss while managing costs.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AgroTrackEnv(gym.Env):
    """
    Custom Gymnasium environment for AgroTrack post-harvest food loss prevention.
    
    Observation Space:
        - Temperature (°C): 0-40
        - Humidity (%): 0-100
        - Days since harvest: 0-30
        - Current quality index: 0-100
        - Storage cost accumulated: 0-1000
        - Product quantity (kg): 0-1000
    
    Action Space:
        0: No intervention (low cost, natural degradation)
        1: Basic cooling (moderate cost, slows degradation)
        2: Advanced cooling (high cost, minimal degradation)
        3: Humidity control (moderate cost, prevents moisture damage)
        4: Rush to market (high cost, saves remaining quality)
    
    Rewards:
        - Positive reward for maintaining high quality
        - Penalty for quality loss
        - Penalty for costs
        - Bonus for successfully delivering high-quality produce
        - Large penalty for total spoilage
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None, max_steps=30):
        super().__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Define observation space: 6 continuous features
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([40, 100, 30, 100, 1000, 1000], dtype=np.float32),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        
        # Action costs
        self.action_costs = {
            0: 5,    # No intervention (minimal storage)
            1: 20,   # Basic cooling
            2: 50,   # Advanced cooling
            3: 30,   # Humidity control
            4: 100   # Rush to market
        }
        
        # Initial state variables
        self.temperature = None
        self.humidity = None
        self.days_since_harvest = None
        self.quality_index = None
        self.storage_cost = None
        self.product_quantity = None
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize with realistic starting conditions
        self.temperature = self.np_random.uniform(15, 25)  # Room temperature
        self.humidity = self.np_random.uniform(40, 70)     # Moderate humidity
        self.days_since_harvest = 0
        self.quality_index = 100.0  # Start with perfect quality
        self.storage_cost = 0.0
        self.product_quantity = self.np_random.uniform(500, 1000)  # kg
        
        self.current_step = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment."""
        # Convert numpy array to int if needed
        if hasattr(action, 'item'):
            action = action.item()
        
        self.current_step += 1
        self.days_since_harvest += 1
        
        # Apply action effects
        cost = self.action_costs[action]
        self.storage_cost += cost
        
        # Update environmental conditions based on action
        if action == 0:  # No intervention
            self.temperature += self.np_random.uniform(-2, 5)
            self.humidity += self.np_random.uniform(-10, 10)
            quality_loss_rate = 5.0
        elif action == 1:  # Basic cooling
            self.temperature = max(10, self.temperature - 5)
            self.humidity += self.np_random.uniform(-5, 5)
            quality_loss_rate = 2.0
        elif action == 2:  # Advanced cooling
            self.temperature = max(4, self.temperature - 10)
            self.humidity = np.clip(self.humidity, 60, 80)
            quality_loss_rate = 0.5
        elif action == 3:  # Humidity control
            self.humidity = np.clip(self.humidity, 60, 70)
            self.temperature += self.np_random.uniform(-3, 3)
            quality_loss_rate = 3.0
        elif action == 4:  # Rush to market (terminal action)
            # Immediately end episode
            terminated = True
            reward = self._calculate_final_reward()
            observation = self._get_observation()
            info = self._get_info()
            info['rush_to_market'] = True
            return observation, reward, terminated, False, info
        
        # Clip environmental variables to valid ranges
        self.temperature = np.clip(self.temperature, 0, 40)
        self.humidity = np.clip(self.humidity, 0, 100)
        
        # Calculate quality degradation based on conditions
        temp_stress = max(0, (self.temperature - 10) / 30)  # Optimal around 10°C
        humidity_stress = abs(self.humidity - 65) / 35  # Optimal around 65%
        
        environmental_loss = (temp_stress + humidity_stress) * 2
        total_quality_loss = quality_loss_rate + environmental_loss
        
        self.quality_index -= total_quality_loss
        self.quality_index = max(0, self.quality_index)
        
        # Calculate reward
        reward = self._calculate_step_reward(cost, total_quality_loss)
        
        # Check termination conditions
        terminated = False
        if self.quality_index <= 10:  # Product spoiled
            terminated = True
            reward -= 500  # Large penalty for spoilage
        elif self.current_step >= self.max_steps:  # Time limit reached
            terminated = True
            reward += self._calculate_final_reward()
        
        truncated = False
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_step_reward(self, cost, quality_loss):
        """Calculate reward for a single step."""
        # Reward components:
        # 1. Maintain quality (positive)
        quality_reward = self.quality_index / 100 * 10
        
        # 2. Penalize quality loss
        quality_loss_penalty = -quality_loss * 2
        
        # 3. Penalize costs
        cost_penalty = -cost / 10
        
        total_reward = quality_reward + quality_loss_penalty + cost_penalty
        
        return total_reward
    
    def _calculate_final_reward(self):
        """Calculate final reward when episode ends."""
        # Reward based on final quality and costs
        quality_bonus = (self.quality_index / 100) ** 2 * 200  # Quadratic bonus for high quality
        cost_penalty = -self.storage_cost / 2
        
        # Bonus for delivering high-quality produce
        if self.quality_index >= 80:
            quality_bonus += 100
        elif self.quality_index >= 60:
            quality_bonus += 50
        
        return quality_bonus + cost_penalty
    
    def _get_observation(self):
        """Get current observation."""
        return np.array([
            self.temperature,
            self.humidity,
            self.days_since_harvest,
            self.quality_index,
            self.storage_cost,
            self.product_quantity
        ], dtype=np.float32)
    
    def _get_info(self):
        """Get additional information."""
        return {
            'temperature': self.temperature,
            'humidity': self.humidity,
            'days_since_harvest': self.days_since_harvest,
            'quality_index': self.quality_index,
            'storage_cost': self.storage_cost,
            'product_quantity': self.product_quantity
        }
    
    def render(self):
        """Render the environment state."""
        if self.render_mode == "human":
            print(f"\n=== AgroTrack Environment ===")
            print(f"Day: {self.days_since_harvest}/{self.max_steps}")
            print(f"Temperature: {self.temperature:.1f}°C")
            print(f"Humidity: {self.humidity:.1f}%")
            print(f"Quality Index: {self.quality_index:.1f}/100")
            print(f"Storage Cost: ${self.storage_cost:.1f}")
            print(f"Product Quantity: {self.product_quantity:.1f} kg")
            print("=" * 30)
