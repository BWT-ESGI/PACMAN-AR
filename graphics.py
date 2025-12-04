import pygame
import math
import matplotlib.pyplot as plt
import numpy as np
from config import *

class GameRenderer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.map_w = env.grid_shape[1] * TILE_SIZE
        self.map_h = env.grid_shape[0] * TILE_SIZE
        self.width = self.map_w
        self.height = self.map_h + 60
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pac-Man")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.title_font = pygame.font.SysFont("Arial", 30, bold=True)
        
        self.anim_timer = 0

    def render_game(self, info_dict={}):
        self.render(info_dict)

    def render(self, info_dict={}):
        self.screen.fill(NAVY)
        self.anim_timer += 1
        
        for r in range(self.env.grid_shape[0]):
            for c in range(self.env.grid_shape[1]):
                x, y = c * TILE_SIZE, r * TILE_SIZE
                cell = self.env.grid[r][c]
                
                if cell == WALL:
                    rect = (x+2, y+2, TILE_SIZE-4, TILE_SIZE-4)
                    pygame.draw.rect(self.screen, WALL_COLOR, rect, border_radius=4)
                elif cell == DOT:
                    pygame.draw.circle(self.screen, DOT_COLOR, (x + TILE_SIZE//2, y + TILE_SIZE//2), 3)
                elif cell == POWER:
                    if (self.anim_timer // 10) % 2 == 0:
                        pygame.draw.circle(self.screen, DOT_COLOR, (x + TILE_SIZE//2, y + TILE_SIZE//2), 7)

        pr, pc = self.env.pacman_pos
        px, py = pc * TILE_SIZE + TILE_SIZE//2, pr * TILE_SIZE + TILE_SIZE//2
        
        mouth_angle = 0.2 + 0.2 * math.sin(self.anim_timer * 0.2) 
        pygame.draw.circle(self.screen, PACMAN_COLOR, (px, py), TILE_SIZE//2 - 2)
        pygame.draw.polygon(self.screen, NAVY, [
            (px, py),
            (px + (TILE_SIZE//2)*math.cos(mouth_angle), py + (TILE_SIZE//2)*math.sin(mouth_angle)),
            (px + (TILE_SIZE//2)*math.cos(-mouth_angle), py + (TILE_SIZE//2)*math.sin(-mouth_angle))
        ])

        for i, g in enumerate(self.env.ghosts):
            gr, gc = g['pos']
            gx, gy = gc * TILE_SIZE + 2, gr * TILE_SIZE + 2
            g_size = TILE_SIZE - 4
            
            color = SCARED_COLOR if g['scared'] > 0 else g['color']
            
            rect_top = (gx, gy, g_size, g_size//2)
            rect_bot = (gx, gy + g_size//2, g_size, g_size//2)
            pygame.draw.ellipse(self.screen, color, (gx, gy, g_size, g_size)) 
            pygame.draw.rect(self.screen, color, (gx, gy + g_size//2, g_size, g_size//2)) 
            
            pygame.draw.circle(self.screen, WHITE, (gx + g_size//3, gy + g_size//3), 3)
            pygame.draw.circle(self.screen, WHITE, (gx + 2*g_size//3, gy + g_size//3), 3)

        pygame.draw.rect(self.screen, BLACK, (0, self.height - 60, self.width, 60))
        pygame.draw.line(self.screen, WHITE, (0, self.height - 60), (self.width, self.height - 60), 2)
        
        score_txt = self.font.render(f"SCORE: {self.env.score}", True, WHITE)
        lives_txt = self.font.render(f"LIVES: {self.env.lives}", True, WHITE)
        level_txt = self.font.render(f"LEVEL: {self.env.level}", True, ORANGE)
        
        ep_txt = self.font.render(f"EP: {info_dict.get('episode', 0)}", True, YELLOW)
        eps_txt = self.font.render(f"EPS: {info_dict.get('epsilon', 0):.2f}", True, CYAN)
        
        self.screen.blit(score_txt, (10, self.height - 50))
        self.screen.blit(lives_txt, (120, self.height - 50))
        self.screen.blit(level_txt, (220, self.height - 50))
        
        self.screen.blit(ep_txt, (10, self.height - 25))
        self.screen.blit(eps_txt, (120, self.height - 25))

        pygame.display.flip()
        self.clock.tick(FPS)

    def render_menu(self, selected_idx, options):
        self.screen.fill(BLACK)
        title = self.title_font.render("MODELE PAC-MAN", True, YELLOW)
        self.screen.blit(title, (self.width//2 - title.get_width()//2, 50))
        
        for i, opt in enumerate(options):
            color = YELLOW if i == selected_idx else WHITE
            txt = self.font.render(opt, True, color)
            self.screen.blit(txt, (self.width//2 - txt.get_width()//2, 150 + i * 40))
            
        pygame.display.flip()
        
    def close(self):
        pygame.quit()

class TrainingChart:
    def __init__(self, visual=True):
        self.visual = visual
        
        if self.visual:
            plt.ion()
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
            self.fig.canvas.manager.set_window_title("Metrics")
        else:
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        
        self.episodes = []
        self.scores = []
        self.epsilons = []
        self.ghosts_eaten = []
        
        self.avg_scores = []
        self.avg_ghosts = []
        self.avg_levels = []
        self.max_levels = []
        
    def update(self, episode, score, epsilon, ghosts_eaten, level):
        self.episodes.append(episode)
        self.scores.append(score)
        self.epsilons.append(epsilon)
        self.ghosts_eaten.append(ghosts_eaten)
        self.max_levels.append(level)
        
        self.avg_scores.append(np.mean(self.scores[-50:]))
        self.avg_ghosts.append(np.mean(self.ghosts_eaten[-50:]))
        self.avg_levels.append(np.mean(self.max_levels[-50:]))
        
        if self.visual and episode % 10 == 0:
            self._draw()
            plt.pause(0.001)
            
    def _draw(self):
        self.ax1.clear()
        l1 = self.ax1.plot(self.episodes, self.scores, color='lightgray', alpha=0.5, label='Raw Score')
        l2 = self.ax1.plot(self.episodes, self.avg_scores, color='blue', linewidth=2, label='Avg Score')
        self.ax1.set_ylabel('Score')
        self.ax1.legend(loc='upper left')
        
        for ax in self.ax1.figure.axes:
            if ax is not self.ax1 and ax is not self.ax2 and ax is not self.ax3:
                ax.remove()
                
        ax1_twin = self.ax1.twinx()
        l3 = ax1_twin.plot(self.episodes, self.epsilons, color='orange', linestyle='--', label='Epsilon', alpha=0.7)
        ax1_twin.set_ylabel('Epsilon', color='orange')
        ax1_twin.set_ylim(0, 1.1)
        
        self.ax2.clear()
        self.ax2.plot(self.episodes, self.ghosts_eaten, color='lightgray', alpha=0.5)
        self.ax2.plot(self.episodes, self.avg_ghosts, color='red', linewidth=2, label='Avg Ghosts')
        self.ax2.set_ylabel('Ghosts Eaten')
        self.ax2.legend(loc='upper left')
        
        self.ax3.clear()
        self.ax3.plot(self.episodes, self.max_levels, color='lightgray', alpha=0.5)
        self.ax3.plot(self.episodes, self.avg_levels, color='green', linewidth=2, label='Avg Level')
        self.ax3.set_ylabel('Max Level')
        self.ax3.set_xlabel('Episode')
        self.ax3.legend(loc='upper left')
        
        plt.tight_layout()
    
    def save_plot(self, filename="models/training_plot.png"):
        if not self.episodes: return
        
        if not self.visual:
            fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
            fig.suptitle('Entraînement Pac-Man')
            
            ax1 = axs[0]
            ax1.plot(self.episodes, self.scores, color='lightgray', alpha=0.5, label='Raw Score')
            ax1.plot(self.episodes, self.avg_scores, color='blue', label='Avg Score')
            ax1.set_ylabel('Score')
            ax1.legend(loc='upper left')
            
            ax2 = ax1.twinx()
            ax2.plot(self.episodes, self.epsilons, color='orange', linestyle='--', alpha=0.7, label='Epsilon')
            ax2.set_ylabel('Epsilon', color='orange')
            
            axs[1].plot(self.episodes, self.ghosts_eaten, color='lightgray', alpha=0.5)
            axs[1].plot(self.episodes, self.avg_ghosts, color='red', label='Avg Ghosts')
            axs[1].set_ylabel('Ghosts Eaten')
            axs[1].legend(loc='upper left')
            
            axs[2].plot(self.episodes, self.max_levels, color='lightgray', alpha=0.5)
            axs[2].plot(self.episodes, self.avg_levels, color='green', label='Avg Level')
            axs[2].set_ylabel('Max Level')
            axs[2].set_xlabel('Episode')
            axs[2].legend(loc='upper left')
            
            fig.savefig(filename)
            plt.close(fig)
        else:
            self.fig.savefig(filename)
            
    def close(self):
        plt.close()
