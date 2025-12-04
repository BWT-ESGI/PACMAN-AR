import pickle
import random
import os
import numpy as np
from config import *

class QLearningAgent:
    def __init__(self):
        self.q_table = {} 
        self.q_table_2 = {}
        
        self.epsilon = EPSILON_START
        self.alpha = ALPHA_START
        self.gamma = GAMMA
        
    def get_q(self, state, table=1):
        target_table = self.q_table if table == 1 else self.q_table_2
        if state not in target_table:
            target_table[state] = np.zeros(len(ACTIONS))
        return target_table[state]

    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(ACTIONS)
        
        q1 = self.get_q(state, 1)
        
        if ALGORITHM == "DOUBLE_Q":
            q2 = self.get_q(state, 2)
            q_vals = q1 + q2 
        else:
            q_vals = q1
            
        max_q = np.max(q_vals)
        actions_with_max_q = np.where(q_vals == max_q)[0]
        return int(random.choice(actions_with_max_q))

    def update(self, state, action, reward, next_state, next_action=None):
        if ALGORITHM == "QLEARNING":
            q_vals = self.get_q(state)
            old_q = q_vals[action]
            next_q_vals = self.get_q(next_state)
            target = reward + self.gamma * np.max(next_q_vals)
            self.q_table[state][action] += self.alpha * (target - old_q)
            
        elif ALGORITHM == "SARSA":
            if next_action is None: return
            q_vals = self.get_q(state)
            old_q = q_vals[action]
            next_q_vals = self.get_q(next_state)
            target = reward + self.gamma * next_q_vals[next_action]
            self.q_table[state][action] += self.alpha * (target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(0.0, self.epsilon * EPSILON_DECAY_RATE)

        self.alpha = max(ALPHA_MIN, self.alpha * ALPHA_DECAY_RATE)

    def save_model(self, filename=MODEL_FILE, history=None):
        data = {
            "q_table": self.q_table,
            "q_table_2": self.q_table_2,
            "epsilon": self.epsilon,
            "history": history
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Modèle et historique sauvegardés dans {filename}")
    
    def load_model(self, filename=MODEL_FILE):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                
            self.q_table = data["q_table"]
            self.q_table_2 = data.get("q_table_2", {})
            self.epsilon = data.get("epsilon", self.epsilon)
            
            print(f"Modèle chargé depuis {filename}")
            return data.get("history", None) 
        else:
            print("Aucun modèle trouvé.")
            return None