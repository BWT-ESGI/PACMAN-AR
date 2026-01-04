import os

# --- Configuration PyGame ---
TILE_SIZE = 54       
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
MAX_LEVEL = 10                  # Niveau max
LEVEL_ONE_GHOST_SPEED = 0.5     # 50% de chance de bouger par frame
GHOST_SPEED_INC_PER_LEVEL = 0.05 # +5% vitesse par niveau
MAX_GHOST_SPEED = 0.9           # Plafond vitesse
POWER_DURATION_BASE = 30        # Pas de temps
POWER_DURATION_DEC_PER_LEVEL = 2
MIN_POWER_DURATION = 10

# --- Hyperparamètres Modèle ---
ALGORITHM = "QLEARNING" # "QLEARNING" / "SARSA"
ALPHA_START = 0.2
ALPHA_MIN = 0.05
ALPHA_DECAY_RATE = 0.999995
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY_RATE = 0.99998
TOTAL_EPISODES = 5000   
INITIAL_LIVES = 3

# --- Récompenses  ---
R_STEP = -1
R_WALL = -50
R_DOT = 10
R_POWER = 50
R_GHOST_EAT = 900
R_DEATH = -500
R_WIN = 1000
R_GAME_WIN = 5000

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
ORANGE = (255, 165, 0)

# Couleurs Fantômes
GHOST_COLORS = [
    (255, 0, 0),
    (255, 182, 193),
    (0, 255, 255),
    (255, 165, 0)
]
SCARED_COLOR = (0, 0, 255) 

# --- Chemins ---
MODEL_FILE = "models/qtable.pkl"
LOG_FILE = "models/training_log.csv"
if not os.path.exists("models"): os.makedirs("models")

# Carte
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
