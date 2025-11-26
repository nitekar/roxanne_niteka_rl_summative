import pygame
import numpy as np
from typing import Tuple, List, Dict

class AgroRenderer:
    """
    Helper class to render AgroTrackEnv in a more flexible way.
    Can be used for both human and rgb_array rendering modes.
    """
    def __init__(self, grid_size: int, cell_size: int = 60):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window = None
        self.clock = None
        self.metadata = {"render_fps": 4}

    def init_window(self, render_mode: str):
        if self.window is None and render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size + 100)
            )
            pygame.display.set_caption("AgroTrack RL Environment")
        if self.clock is None and render_mode == "human":
            self.clock = pygame.time.Clock()

    def draw(self, agent_pos: np.ndarray, inventory: List[Dict], step: int, max_steps: int,
             total_loss: float, total_saved: float, temperature: float, humidity: float, render_mode: str):
        self.init_window(render_mode)
        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size + 100))
        canvas.fill((245, 245, 240))

        # Draw grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(canvas, (200, 200, 200), (x * self.cell_size, 0), 
                             (x * self.cell_size, self.grid_size * self.cell_size), 1)
        for y in range(self.grid_size + 1):
            pygame.draw.line(canvas, (200, 200, 200), (0, y * self.cell_size),
                             (self.grid_size * self.cell_size, y * self.cell_size), 1)

        # Draw inventory items
        for item in inventory:
            color = (74, 222, 128) if item['freshness'] > 70 else \
                    (250, 204, 21) if item['freshness'] > 40 else (239, 68, 68)
            pygame.draw.rect(canvas, color,
                             (item['x'] * self.cell_size + 5, item['y'] * self.cell_size + 5,
                              self.cell_size - 10, self.cell_size - 10))
            font = pygame.font.Font(None, 20)
            canvas.blit(font.render(f"{int(item['freshness'])}%", True, (0, 0, 0)),
                        (item['x'] * self.cell_size + 12, item['y'] * self.cell_size + 25))

        # Draw agent
        pygame.draw.circle(canvas, (59, 130, 246),
                           (agent_pos[0] * self.cell_size + self.cell_size // 2,
                            agent_pos[1] * self.cell_size + self.cell_size // 2),
                           self.cell_size // 3)

        # Draw info panel
        info_y = self.grid_size * self.cell_size + 10
        font = pygame.font.Font(None, 24)
        info_texts = [
            f"Step: {step}/{max_steps}",
            f"Items: {len(inventory)}",
            f"Loss: {total_loss:.1f}",
            f"Saved: {total_saved:.1f}",
            f"Temp: {temperature:.1f}°C",
            f"Humidity: {humidity:.1f}%"
        ]
        x_offset = 10
        for text_str in info_texts:
            text = font.render(text_str, True, (0, 0, 0))
            canvas.blit(text, (x_offset, info_y))
            x_offset += 120

        if render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
