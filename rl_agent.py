import pickle
import random
import os
import numpy as np
from config import *

class QLearningAgent:
    def __init__(self):
        self.q_table = {} 
        # Pour Double Q-Learning (Optionnel)
        self.q_table_2 = {}
        
        self.epsilon = EPSILON_START
        self.alpha = ALPHA
        self.gamma = GAMMA
        
    def get_q(self, state, table=1):
        target_table = self.q_table if table == 1 else self.q_table_2
        if state not in target_table:
            target_table[state] = np.zeros(len(ACTIONS))
        return target_table[state]

    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(ACTIONS)
        
        # Exploitation
        q1 = self.get_q(state, 1)
        
        if ALGORITHM == "DOUBLE_Q":
            q2 = self.get_q(state, 2)
            q_vals = q1 + q2 # On somme pour la décision
        else:
            q_vals = q1
            
        # Break ties randomly
        max_q = np.max(q_vals)
        actions_with_max_q = np.where(q_vals == max_q)[0]
        return int(random.choice(actions_with_max_q))

    def update(self, state, action, reward, next_state, next_action=None):
        """
        Mise à jour selon l'algorithme choisi dans config.py
        """
        if ALGORITHM == "QLEARNING":
            q_vals = self.get_q(state)
            old_q = q_vals[action]
            next_q_vals = self.get_q(next_state)
            target = reward + self.gamma * np.max(next_q_vals)
            self.q_table[state][action] += self.alpha * (target - old_q)
            
        elif ALGORITHM == "SARSA":
            if next_action is None: return # SARSA a besoin de l'action suivante
            q_vals = self.get_q(state)
            old_q = q_vals[action]
            next_q_vals = self.get_q(next_state)
            target = reward + self.gamma * next_q_vals[next_action]
            self.q_table[state][action] += self.alpha * (target - old_q)
            
        elif ALGORITHM == "DOUBLE_Q":
            # Pile ou face pour savoir quelle table mettre à jour
            if random.random() < 0.5:
                # Update Q1
                q1 = self.get_q(state, 1)
                q2 = self.get_q(next_state, 2)
                # On utilise Q1 pour choisir la meilleure action, mais Q2 pour sa valeur
                best_action = np.argmax(self.get_q(next_state, 1))
                target = reward + self.gamma * q2[best_action]
                q1[action] += self.alpha * (target - q1[action])
            else:
                # Update Q2
                q2 = self.get_q(state, 2)
                q1 = self.get_q(next_state, 1)
                best_action = np.argmax(self.get_q(next_state, 2))
                target = reward + self.gamma * q1[best_action]
                q2[action] += self.alpha * (target - q2[action])

    def decay_epsilon(self, episode):
        if EPSILON_DECAY_TYPE == "exponential":
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY_RATE)
        else:
            # Linear decay
            decay_step = (EPSILON_START - EPSILON_MIN) / (TOTAL_EPISODES * 0.8)
            self.epsilon = max(EPSILON_MIN, self.epsilon - decay_step)

    def save_model(self, filename=MODEL_FILE):
        # On sauvegarde tout (si double Q, on sauvegarde Q1 qui est la principale)
        with open(filename, 'wb') as f: pickle.dump(self.q_table, f)
        print(f"Modèle sauvegardé : {filename}")
    
    def load_model(self, filename=MODEL_FILE):
        if os.path.exists(filename):
            with open(filename, 'rb') as f: self.q_table = pickle.load(f)
            print(f"Modèle chargé: {len(self.q_table)} états.")
        else:
            print("Aucun modèle trouvé, départ à zéro.")
