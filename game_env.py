import numpy as np
import random
from config import *

class PacmanEnv:
    def __init__(self):
        self.grid_shape = (len(GAME_MAP), len(GAME_MAP[0]))
        self.reset()

    def reset(self):
        self.level = 1
        self.score = 0
        self.lives = INITIAL_LIVES
        self.ghosts_eaten = 0 
        self.last_action = 4 
        return self._start_level()

    def next_level(self):
        self.level += 1
        return self._start_level()

    def _start_level(self):
        self.grid = [row[:] for row in GAME_MAP]
        self.done = False
        self.steps = 0
        self.max_steps = 2000 
        self.total_dots = sum(row.count(DOT) + row.count(POWER) for row in self.grid)
        self._reset_positions()
        self.last_action = 4
        
        self.ghost_move_prob = min(MAX_GHOST_SPEED, 
                                 LEVEL_ONE_GHOST_SPEED + (self.level - 1) * GHOST_SPEED_INC_PER_LEVEL)
        self.power_duration = max(MIN_POWER_DURATION, 
                                POWER_DURATION_BASE - (self.level - 1) * POWER_DURATION_DEC_PER_LEVEL)
        
        return self.get_state()

    def _reset_positions(self):
        self.pacman_pos = (5, 5) 
        
        self.ghosts = [
            {'pos': (3, 5), 'start': (3, 5), 'scared': 0, 'color': GHOST_COLORS[0]}, 
            {'pos': (5, 7), 'start': (5, 7), 'scared': 0, 'color': GHOST_COLORS[1]}
        ]

    def step(self, action):
        if self.done: return self.get_state(), 0, True, {}
        
        self.steps += 1
        reward = R_STEP 
        info = {"level_cleared": False, "ghosts": self.ghosts_eaten, "game_won": False}
        
        r, c = self.pacman_pos
        dr, dc = {UP: (-1,0), DOWN: (1,0), LEFT: (0,-1), RIGHT: (0,1)}[action]
        nr, nc = r + dr, c + dc
        
        if not (0 <= nr < self.grid_shape[0] and 0 <= nc < self.grid_shape[1]) or self.grid[nr][nc] == WALL:
            reward += R_WALL 
        else:
            self.pacman_pos = (nr, nc)
            self.last_action = action
        
        cell = self.grid[self.pacman_pos[0]][self.pacman_pos[1]]
        if cell == DOT:
            reward += R_DOT
            self.grid[self.pacman_pos[0]][self.pacman_pos[1]] = EMPTY
            self.total_dots -= 1
            self.score += 10
        elif cell == POWER:
            reward += R_POWER
            self.grid[self.pacman_pos[0]][self.pacman_pos[1]] = EMPTY
            self.total_dots -= 1
            self.score += 50
            for g in self.ghosts: g['scared'] = self.power_duration 

        if self.total_dots == 0:
            if self.level >= MAX_LEVEL:
                self.done = True
                info["game_won"] = True
                return self.get_state(), R_GAME_WIN, True, info
            else:
                self.done = True
                info["level_cleared"] = True
                return self.get_state(), R_WIN, True, info

        for g in self.ghosts:
            if g['scared'] > 0: g['scared'] -= 1
            
            should_move = True if g['scared'] > 0 else (random.random() < self.ghost_move_prob)
            
            if should_move:
                moves = []
                gr, gc = g['pos']
                for m_dr, m_dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    if self.grid[gr+m_dr][gc+m_dc] != WALL: moves.append((gr+m_dr, gc+m_dc))
                
                if moves:
                    if g['scared'] > 0:
                        if random.random() < 0.2: 
                            g['pos'] = random.choice(moves)
                        else:
                            g['pos'] = max(moves, key=lambda p: abs(p[0]-self.pacman_pos[0]) + abs(p[1]-self.pacman_pos[1]))
                    else:
                        if random.random() < 0.3: 
                            g['pos'] = min(moves, key=lambda p: abs(p[0]-self.pacman_pos[0]) + abs(p[1]-self.pacman_pos[1]))
                        else:
                            g['pos'] = random.choice(moves)

            if g['pos'] == self.pacman_pos:
                if g['scared'] > 0:
                    reward += R_GHOST_EAT
                    self.score += 200
                    self.ghosts_eaten += 1
                    g['pos'] = g['start']
                    g['scared'] = 0
                else:
                    self.lives -= 1
                    if self.lives == 0:
                        return self.get_state(), R_DEATH, True, {"result": "die", "ghosts": self.ghosts_eaten}
                    else:
                        reward += R_DEATH / 4 
                        self._reset_positions()
                        self.last_action = 4 
                        return self.get_state(), reward, False, {"result": "hit", "ghosts": self.ghosts_eaten}

        if self.steps >= self.max_steps: self.done = True
        
        return self.get_state(), reward, self.done, info

    def get_state(self):
        head_r, head_c = self.pacman_pos
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        radar = []

        nearest_dist = float('inf')
        target = None
        for r in range(self.grid_shape[0]):
            for c in range(self.grid_shape[1]):
                if self.grid[r][c] in [DOT, POWER]:
                    dist = abs(r - head_r) + abs(c - head_c)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        target = (r, c)

        for delta_row, delta_col in deltas:
            next_r, next_c = head_r + delta_row, head_c + delta_col
            is_wall = 0
            if not (0 <= next_r < self.grid_shape[0] and 0 <= next_c < self.grid_shape[1]) or self.grid[next_r][next_c] == WALL:
                is_wall = 1
            
            ghost_status = 0 
            for i in range(1, 8): # regarder jusqu'a 7 cellules devant
                r, c = head_r + (delta_row*i), head_c + (delta_col*i)
                for g in self.ghosts:
                    if g['pos'] == (r, c):
                        if g['scared'] > 0: 
                            ghost_status = 2 
                        else:
                            if i <= 2 and ghost_status != 1: ghost_status = 1 
                        break 
                if ghost_status == 1: break 
            
            food_dir = 0
            
            if target:
                dist_current = abs(target[0] - head_r) + abs(target[1] - head_c)
                dist_next = abs(target[0] - (head_r+delta_row)) + abs(target[1] - (head_c+delta_col))
                if dist_next < dist_current:
                    food_dir = 1

            radar.append((is_wall, ghost_status, food_dir)) # 2 x 3 x 2 = 12 états possibles par direction, 12^4 = 20736
            
        any_scared = 1 if any(g['scared'] > 0 for g in self.ghosts) else 0
        
        difficulty = 0
        if self.level >= (MAX_LEVEL // 3):
            difficulty = 1
        if self.level >= (MAX_LEVEL * 2 // 3):
            difficulty = 2

        return tuple(radar + [any_scared, self.last_action]) # 20736 x [2 x 5] = 207 360 états possibles