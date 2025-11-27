import pygame
import numpy as np
from typing import Tuple, List, Dict
class AgroTrackRenderer:
    """Handles all rendering operations for the AgroTrack environment"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.window = None
        self.clock = None
        self.fonts = {}
        
        # Color palette
        self.colors = {
            'background': (240, 240, 240),
            'title': (20, 20, 100),
            'text': (0, 0, 0),
            'bar_empty': (200, 200, 200),
            'quality_high': (50, 200, 50),
            'quality_medium': (255, 200, 50),
            'quality_low': (255, 50, 50),
            'panel_bg': (255, 255, 255),
            'border': (100, 100, 100)
        }
    
    def initialize(self):
        """Initialize Pygame window and fonts"""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("AgroTrack Environment")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Initialize fonts
        self.fonts = {
            'large': pygame.font.Font(None, 42),
            'medium': pygame.font.Font(None, 28),
            'small': pygame.font.Font(None, 22)
        }
    
    def render_frame(self, env_state: Dict) -> np.ndarray:
      
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.colors['background'])
        
        # Render components
        self._render_title(canvas)
        self._render_inventory_section(canvas, env_state)
        self._render_environment_section(canvas, env_state)
        self._render_statistics_section(canvas, env_state)
        self._render_market_section(canvas, env_state)
        self._render_spoilage_section(canvas, env_state)
        
        # Display on window
        if self.window is not None:
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            if self.clock:
                self.clock.tick(4)  # 4 FPS
        
        # Return as numpy array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
    
    def _render_title(self, canvas: pygame.Surface):
        """Render title bar"""
        title = self.fonts['large'].render("AgroTrack Dashboard", True, self.colors['title'])
        canvas.blit(title, (20, 20))
    
    def _render_inventory_section(self, canvas: pygame.Surface, env_state: Dict):
        """Render inventory bars with quality indicators"""
        y_offset = 80
        
        for i, food_type in enumerate(env_state['food_types']):
            # Food type label
            label = self.fonts['medium'].render(food_type, True, self.colors['text'])
            canvas.blit(label, (20, y_offset))
            
            # Inventory bar background
            bar_x, bar_y = 200, y_offset
            bar_width, bar_height = 300, 25
            pygame.draw.rect(canvas, self.colors['bar_empty'], 
                           (bar_x, bar_y, bar_width, bar_height))
            
            # Inventory bar (colored by quality)
            inv_width = int(env_state['inventory'][i] * bar_width)
            color = self._get_quality_color(env_state['quality'][i])
            pygame.draw.rect(canvas, color, (bar_x, bar_y, inv_width, bar_height))
            
            # Border
            pygame.draw.rect(canvas, self.colors['border'], 
                           (bar_x, bar_y, bar_width, bar_height), 2)
            
            # Quality text
            quality_text = self.fonts['small'].render(
                f"Q: {env_state['quality'][i]:.2f}", True, self.colors['text']
            )
            canvas.blit(quality_text, (bar_x + bar_width + 10, bar_y + 3))
            
            # Inventory value
            inv_text = self.fonts['small'].render(
                f"{env_state['inventory'][i]:.2f}", True, self.colors['text']
            )
            canvas.blit(inv_text, (bar_x + bar_width + 100, bar_y + 3))
            
            y_offset += 40
    
    def _render_environment_section(self, canvas: pygame.Surface, env_state: Dict):
        """Render environmental conditions"""
        env_y = 350
        
        # Section title
        title = self.fonts['medium'].render("Environment", True, self.colors['title'])
        canvas.blit(title, (20, env_y - 30))
        
        # Environmental data
        env_info = [
            f"Temperature: {env_state['temperature']:.2f}",
            f"Humidity: {env_state['humidity']:.2f}",
            f"Days to Market: {env_state['days_to_market']:.2f}",
            f"Storage Capacity: {env_state['storage_capacity']:.2f}",
            f"Transport Avail: {env_state['transport_availability']:.2f}",
            f"Preservation Res: {env_state['preservation_resources']:.2f}"
        ]
        
        for i, info in enumerate(env_info):
            text = self.fonts['small'].render(info, True, self.colors['text'])
            canvas.blit(text, (20, env_y + i * 25))
    
    def _render_statistics_section(self, canvas: pygame.Surface, env_state: Dict):
        """Render episode statistics"""
        stats_y = 550
        
        # Section title
        title = self.fonts['medium'].render("Statistics", True, self.colors['title'])
        canvas.blit(title, (20, stats_y - 30))
        
        # Statistics
        stats = [
            f"Step: {env_state['current_step']}/{env_state['max_steps']}",
            f"Episode Reward: {env_state['episode_reward']:.2f}",
            f"Food Saved: {env_state['total_saved']:.3f}",
            f"Food Wasted: {env_state['total_wasted']:.3f}",
            f"Avg Quality: {np.mean(env_state['quality']):.3f}",
            f"Total Inventory: {np.sum(env_state['inventory']):.3f}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.fonts['small'].render(stat, True, self.colors['text'])
            canvas.blit(text, (20, stats_y + i * 25))
    
    def _render_market_section(self, canvas: pygame.Surface, env_state: Dict):
        """Render market prices"""
        market_y = 700
        
        # Section title
        title = self.fonts['medium'].render("Market Prices", True, self.colors['title'])
        canvas.blit(title, (20, market_y - 30))
        
        # Market prices
        for i, (food, price) in enumerate(zip(env_state['food_types'], 
                                              env_state['market_prices'])):
            text = self.fonts['small'].render(
                f"{food}: ${price:.2f}", True, self.colors['text']
            )
            canvas.blit(text, (20, market_y + i * 25))
    
    def _render_spoilage_section(self, canvas: pygame.Surface, env_state: Dict):
        """Render spoilage rates"""
        spoilage_y = 700
        
        # Section title
        title = self.fonts['medium'].render("Spoilage Rates", True, self.colors['title'])
        canvas.blit(title, (400, spoilage_y - 30))
        
        # Spoilage rates
        for i, (food, rate) in enumerate(zip(env_state['food_types'], 
                                             env_state['spoilage_rates'])):
            text = self.fonts['small'].render(
                f"{food}: {rate:.3f}", True, self.colors['text']
            )
            canvas.blit(text, (400, spoilage_y + i * 25))
    
    def _get_quality_color(self, quality: float) -> Tuple[int, int, int]:
        """Get color based on quality level"""
        if quality > 0.7:
            return self.colors['quality_high']
        elif quality > 0.4:
            return self.colors['quality_medium']
        else:
            return self.colors['quality_low']
    
    def close(self):
        """Close the rendering window"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


def create_visualization_data(env) -> Dict:
 
    return {
        'inventory': env.inventory,
        'quality': env.quality,
        'food_types': env.food_types,
        'temperature': env.temperature,
        'humidity': env.humidity,
        'days_to_market': env.days_to_market,
        'storage_capacity': env.storage_capacity,
        'transport_availability': env.transport_availability,
        'preservation_resources': env.preservation_resources,
        'market_prices': env.market_prices,
        'spoilage_rates': env.spoilage_rates,
        'current_step': env.current_step,
        'max_steps': env.max_steps,
        'episode_reward': env.episode_reward,
        'total_saved': env.total_saved,
        'total_wasted': env.total_wasted
    }


def save_frame_as_image(frame: np.ndarray, filepath: str):
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 10))
    plt.imshow(frame)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Frame saved to {filepath}")