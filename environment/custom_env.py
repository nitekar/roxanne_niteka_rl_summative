import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Dict, Tuple, Any

class AgroTrackEnv(gym.Env):
  
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 100):
        super().__init__()
        
        # Environment parameters
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Food types and their characteristics
        self.food_types = ['Vegetables', 'Fruits', 'Grains', 'Dairy', 'Meat']
        self.n_food_types = len(self.food_types)
        
        # Action space: 8 discrete actions
        self.action_space = spaces.Discrete(8)
        
        # Observation space dimension = inventory( n ) + quality( n )
        # + temperature, humidity, days_to_market, storage_capacity,
        # transport_availability, preservation_resources (6 scalars)
        # + market_prices( n ) + spoilage_rates( n ) = 4*n + 6
        obs_dim = (self.n_food_types * 4) + 6
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Rendering
        self.window = None
        self.clock = None
        self.window_size = 1000
        
        # Initialize state
        self.state = None
        self.current_step = 0
        self.episode_reward = 0
        self.total_saved = 0
        self.total_wasted = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.inventory = self.np_random.uniform(0.2, 0.8, self.n_food_types)
        self.quality = self.np_random.uniform(0.6, 0.95, self.n_food_types)
        self.temperature = self.np_random.uniform(0.3, 0.8)
        self.humidity = self.np_random.uniform(0.3, 0.8)
        self.days_to_market = self.np_random.uniform(0.3, 0.9)
        self.storage_capacity = 0.7
        self.transport_availability = self.np_random.uniform(0.5, 1.0)
        self.preservation_resources = 0.8
        self.market_prices = self.np_random.uniform(0.5, 1.0, self.n_food_types)
        self.spoilage_rates = self._calculate_spoilage_rates()
        
        # Reset tracking
        self.current_step = 0
        self.episode_reward = 0
        self.total_saved = 0
        self.total_wasted = 0
        
        return self._get_obs(), self._get_info()
    
    def _calculate_spoilage_rates(self) -> np.ndarray:
        """Calculate spoilage rates based on environmental conditions"""
        base_rates = np.array([0.15, 0.12, 0.05, 0.20, 0.18])
        temp_factor = 1 + (self.temperature * 2)
        humidity_factor = 1 + (self.humidity * 1.5)
        return np.clip(base_rates * temp_factor * humidity_factor, 0, 0.5)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        reward = 0
        # Start with the standard info fields (avg_quality, totals, etc.)
        info = self._get_info()
        # Add the human-readable action name
        info['action_name'] = self._get_action_name(action)
        
        # Execute action
        if action == 0:
            reward += self._action_monitor()
        elif action == 1:
            reward += self._action_preserve_basic()
        elif action == 2:
            reward += self._action_transport_market()
        elif action == 3:
            reward += self._action_reuse()
        elif action == 4:
            reward += self._action_preserve_advanced()
        elif action == 5:
            reward += self._action_emergency_transport()
        elif action == 6:
            reward += self._action_compost()
        elif action == 7:
            reward += self._action_process()
        
        # Natural degradation
        degradation_penalty = self._apply_degradation()
        reward += degradation_penalty
        
        # Update environment
        self._update_environment()
        
        # Increment step
        self.current_step += 1
        self.episode_reward += reward
        
        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        if np.sum(self.inventory) < 0.01:
            terminated = True
            
        return self._get_obs(), reward, terminated, truncated, info
    
    def _action_monitor(self) -> float:
        """Monitor: Small penalty for inaction"""
        return -0.5
    
    def _action_preserve_basic(self) -> float:
        """Apply basic preservation"""
        reward = 0
        if self.preservation_resources > 0.2:
            quality_boost = 0.05
            self.quality = np.clip(self.quality + quality_boost, 0, 1)
            self.preservation_resources -= 0.1
            reward = 3.0
        else:
            reward = -2.0
        return reward
    
    def _action_transport_market(self) -> float:
        """Transport to market"""
        reward = 0
        if self.transport_availability > 0.3:
            for i in range(self.n_food_types):
                if self.inventory[i] > 0.1 and self.quality[i] > 0.5:
                    value = self.inventory[i] * self.quality[i] * self.market_prices[i]
                    reward += value * 10
                    self.total_saved += self.inventory[i]
                    self.inventory[i] *= 0.3
                elif self.inventory[i] > 0.1 and self.quality[i] <= 0.5:
                    reward -= 5
                    self.total_wasted += self.inventory[i] * 0.3
                    self.inventory[i] *= 0.7
            self.transport_availability -= 0.2
        else:
            reward = -3.0
        return reward
    
    def _action_reuse(self) -> float:
        """Donate/reuse food"""
        reward = 0
        for i in range(self.n_food_types):
            if self.inventory[i] > 0.1 and self.quality[i] > 0.4:
                donation_amount = self.inventory[i] * 0.5
                reward += donation_amount * self.quality[i] * 5
                self.total_saved += donation_amount
                self.inventory[i] -= donation_amount
        return reward
    
    def _action_preserve_advanced(self) -> float:
        """Advanced preservation"""
        reward = 0
        if self.preservation_resources > 0.4:
            quality_boost = 0.15
            self.quality = np.clip(self.quality + quality_boost, 0, 1)
            self.preservation_resources -= 0.25
            reward = 7.0
        else:
            reward = -3.0
        return reward
    
    def _action_emergency_transport(self) -> float:
        """Emergency transport"""
        reward = 0
        if self.transport_availability > 0.5 and self.preservation_resources > 0.3:
            for i in range(self.n_food_types):
                if self.inventory[i] > 0.1:
                    value = self.inventory[i] * self.quality[i] * self.market_prices[i]
                    reward += value * 15
                    self.total_saved += self.inventory[i]
                    self.inventory[i] *= 0.2
            self.transport_availability -= 0.4
            self.preservation_resources -= 0.3
        else:
            reward = -5.0
        return reward
    
    def _action_compost(self) -> float:
        """Compost low quality items"""
        reward = 0
        for i in range(self.n_food_types):
            if self.inventory[i] > 0.1 and self.quality[i] < 0.3:
                compost_amount = self.inventory[i]
                reward += compost_amount * 2
                self.inventory[i] = 0
        return reward
    
    def _action_process(self) -> float:
        """Process into preserved products"""
        reward = 0
        if self.preservation_resources > 0.3:
            for i in range(self.n_food_types):
                if self.inventory[i] > 0.2 and self.quality[i] > 0.5:
                    processed = self.inventory[i] * 0.6
                    reward += processed * 8
                    self.total_saved += processed
                    self.inventory[i] *= 0.4
            self.preservation_resources -= 0.2
        else:
            reward = -2.0
        return reward
    
    def _apply_degradation(self) -> float:
        """Apply natural degradation"""
        penalty = 0
        
        self.quality -= self.spoilage_rates * 0.1
        self.quality = np.clip(self.quality, 0, 1)
        
        for i in range(self.n_food_types):
            if self.quality[i] < 0.15 and self.inventory[i] > 0:
                penalty -= self.inventory[i] * 15
                self.total_wasted += self.inventory[i]
                self.inventory[i] = 0
        
        return penalty
    
    def _update_environment(self):
        """Update environmental conditions"""
        self.temperature += self.np_random.uniform(-0.05, 0.05)
        self.temperature = np.clip(self.temperature, 0, 1)
        
        self.humidity += self.np_random.uniform(-0.05, 0.05)
        self.humidity = np.clip(self.humidity, 0, 1)
        
        self.market_prices += self.np_random.uniform(-0.1, 0.1, self.n_food_types)
        self.market_prices = np.clip(self.market_prices, 0.3, 1.2)
        
        self.days_to_market = max(0, self.days_to_market - 0.05)
        
        self.transport_availability = min(1.0, self.transport_availability + 0.05)
        self.preservation_resources = min(1.0, self.preservation_resources + 0.03)
        
        self.spoilage_rates = self._calculate_spoilage_rates()
    
    def _get_obs(self) -> np.ndarray:
        """Construct observation vector"""
        obs = np.concatenate([
            self.inventory,
            self.quality,
            [self.temperature],
            [self.humidity],
            [self.days_to_market],
            [self.storage_capacity],
            [self.transport_availability],
            [self.preservation_resources],
            self.market_prices,
            self.spoilage_rates
        ])
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Return additional information"""
        avg_q = float(np.mean(self.quality))
        total_inv = float(np.sum(self.inventory))
        # Map legacy keys expected by other scripts to current names
        return {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'total_saved': float(self.total_saved),
            'total_wasted': float(self.total_wasted),
            'avg_quality': avg_q,
            'avg_freshness': avg_q,  # alias for compatibility
            'total_inventory': total_inv,
            # Legacy keys some scripts expect
            'food_saved': float(self.total_saved),
            'food_spoiled': float(self.total_wasted),
            'food_reused': 0.0,
            'revenue': 0.0,
            'spoilage_risk': float(np.mean(self.spoilage_rates))
        }
    
    def _get_action_name(self, action: int) -> str:
        """Get human-readable action name"""
        actions = [
            'Monitor', 'Basic Preservation', 'Transport to Market',
            'Reuse/Donate', 'Advanced Preservation', 'Emergency Transport',
            'Compost', 'Process Products'
        ]
        return actions[action]
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            return self._render_frame()
    
    def _render_frame(self):
        """Render frame using Pygame"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("AgroTrack Environment")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((240, 240, 240))
        
        # Fonts
        font_large = pygame.font.Font(None, 42)
        font_med = pygame.font.Font(None, 28)
        font_small = pygame.font.Font(None, 22)
        
        # Title
        title = font_large.render("AgroTrack Dashboard", True, (20, 20, 100))
        canvas.blit(title, (20, 20))
        
        # Inventory bars
        y_offset = 80
        for i, food_type in enumerate(self.food_types):
            label = font_med.render(food_type, True, (0, 0, 0))
            canvas.blit(label, (20, y_offset))
            
            bar_x, bar_y = 200, y_offset
            bar_width, bar_height = 300, 25
            pygame.draw.rect(canvas, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height))
            inv_width = int(self.inventory[i] * bar_width)
            color = self._get_quality_color(self.quality[i])
            pygame.draw.rect(canvas, color, (bar_x, bar_y, inv_width, bar_height))
            
            quality_text = font_small.render(f"Q: {self.quality[i]:.2f}", True, (0, 0, 0))
            canvas.blit(quality_text, (bar_x + bar_width + 10, bar_y + 3))
            
            y_offset += 40
        
        # Environmental conditions
        env_y = 350
        env_info = [
            f"Temperature: {self.temperature:.2f}",
            f"Humidity: {self.humidity:.2f}",
            f"Days to Market: {self.days_to_market:.2f}",
            f"Storage Capacity: {self.storage_capacity:.2f}",
            f"Transport Avail: {self.transport_availability:.2f}",
            f"Preservation Res: {self.preservation_resources:.2f}"
        ]
        
        for i, info in enumerate(env_info):
            text = font_small.render(info, True, (0, 0, 0))
            canvas.blit(text, (20, env_y + i * 25))
        
        # Statistics
        stats_y = 550
        stats = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Episode Reward: {self.episode_reward:.2f}",
            f"Food Saved: {self.total_saved:.2f}",
            f"Food Wasted: {self.total_wasted:.2f}",
            f"Avg Quality: {np.mean(self.quality):.2f}"
        ]
        
        stats_title = font_med.render("Statistics", True, (20, 20, 100))
        canvas.blit(stats_title, (20, stats_y - 30))
        
        for i, stat in enumerate(stats):
            text = font_small.render(stat, True, (0, 0, 0))
            canvas.blit(text, (20, stats_y + i * 25))
        
        # Market prices
        market_y = 700
        market_title = font_med.render("Market Prices", True, (20, 20, 100))
        canvas.blit(market_title, (20, market_y - 30))
        
        for i, (food, price) in enumerate(zip(self.food_types, self.market_prices)):
            text = font_small.render(f"{food}: ${price:.2f}", True, (0, 0, 0))
            canvas.blit(text, (20, market_y + i * 25))
        
        # Spoilage rates
        spoilage_y = 700
        spoilage_title = font_med.render("Spoilage Rates", True, (20, 20, 100))
        canvas.blit(spoilage_title, (400, spoilage_y - 30))
        
        for i, (food, rate) in enumerate(zip(self.food_types, self.spoilage_rates)):
            text = font_small.render(f"{food}: {rate:.3f}", True, (0, 0, 0))
            canvas.blit(text, (400, spoilage_y + i * 25))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def _get_quality_color(self, quality: float) -> Tuple[int, int, int]:
        """Get color based on quality level"""
        if quality > 0.7:
            return (50, 200, 50)
        elif quality > 0.4:
            return (255, 200, 50)
        else:
            return (255, 50, 50)
    
    def close(self):
        """Close rendering window"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()