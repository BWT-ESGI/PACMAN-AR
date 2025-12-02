import os

# --- Configuration Jeu ---
TILE_SIZE = 24       
FPS = 60             
GAME_SPEED_VISUAL = 20 

# Entités
EMPTY = 0
WALL = 1
DOT = 2
POWER = 3

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

# --- Difficulté & Niveaux ---
MAX_LEVEL = 10                  # Niveau maximum
LEVEL_ONE_GHOST_SPEED = 0.5     # 50% de chance de bouger par frame (simplifié)
GHOST_SPEED_INC_PER_LEVEL = 0.05 # +5% par niveau
MAX_GHOST_SPEED = 0.9           # Plafond
POWER_DURATION_BASE = 30        # Pas de temps
POWER_DURATION_DEC_PER_LEVEL = 2
MIN_POWER_DURATION = 10

# --- Hyperparamètres RL ---
ALGORITHM = "QLEARNING" # "QLEARNING", "SARSA", "DOUBLE_Q"
ALPHA = 0.1             
GAMMA = 0.95            
EPSILON_START = 1.0
EPSILON_MIN = 0.02      
EPSILON_DECAY_TYPE = "exponential" 
EPSILON_DECAY_RATE = 0.9996 
TOTAL_EPISODES = 5000   
INITIAL_LIVES = 3

# --- Récompenses (Reward Shaping) ---
R_STEP = -5             
R_WALL = -10            
R_DOT = 10              
R_POWER = 50            
R_GHOST_EAT = 800       
R_DEATH = -500          
R_WIN = 1000            # Finir le niveau (Level Clear)
R_GAME_WIN = 5000       # Finir le jeu (Tous les niveaux)

# --- Couleurs & Style ---
BLACK = (0, 0, 0)
NAVY = (0, 0, 50)       
WHITE = (255, 255, 255)
WALL_COLOR = (33, 33, 255) 
PACMAN_COLOR = (255, 255, 0)
DOT_COLOR = (255, 183, 174)
TEXT_COLOR = (255, 255, 255)
YELLOW = (255, 255, 0) 
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0) # Ajouté pour le HUD

# Couleurs Fantômes
GHOST_COLORS = [
    (255, 0, 0),    # Blinky
    (255, 182, 193),# Pinky
    (0, 255, 255),  # Inky
    (255, 165, 0)   # Clyde
]
SCARED_COLOR = (0, 0, 255) 

# --- Chemins ---
MODEL_FILE = "models/qtable.pkl"
LOG_FILE = "models/training_log.csv"
if not os.path.exists("models"): os.makedirs("models")

# Carte (Ta version plus dense)
GAME_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1],
    [1, 3, 1, 2, 2, 2, 2, 2, 2, 2, 1, 3, 1],
    [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1],
    [1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1],
    [1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1],
    [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
