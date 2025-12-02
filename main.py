import pygame
import csv
import argparse
import time
import sys
from game_env import PacmanEnv
from rl_agent import QLearningAgent
from graphics import GameRenderer, TrainingChart
from config import *

# Initialisation des logs
with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Score", "Steps", "Epsilon", "GhostsEaten"])

def log_episode(episode, score, steps, epsilon, ghosts):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, score, steps, epsilon, ghosts])

def train(episodes, visual=False, graphics=False, load=False):
    env = PacmanEnv()
    agent = QLearningAgent()
    
    if load:
        agent.load_model()

    renderer = GameRenderer(env) if visual else None
    chart = TrainingChart() if graphics else None

    print(f"Démarrage de l'entraînement ({ALGORITHM}) pour {episodes} épisodes...")
    
    try:
        for ep in range(1, episodes + 1):
            # Reset Episode (Niveau 1, Score 0)
            state = env.reset()
            
            # Pour SARSA, on doit choisir la première action avant la boucle
            action = agent.choose_action(state)
            
            done = False
            total_reward = 0
            
            # Boucle Épisode (Peut traverser plusieurs niveaux)
            while not done:
                if visual and renderer:
                    renderer.render(info_dict={'episode': ep, 'epsilon': agent.epsilon})
                
                # 1. Exécuter l'action
                next_state, reward, episode_done, info = env.step(action)
                
                # GESTION NIVEAUX
                if info.get("level_cleared", False):
                    # On ne finit pas l'épisode, on passe au niveau suivant
                    # L'agent reçoit sa récompense de victoire mais continue de jouer
                    # On doit réinitialiser l'état car la grille change
                    next_state = env.next_level()
                    # On ne change pas episode_done ici, car on veut continuer
                    episode_done = False 
                
                # 2. Choisir l'action suivante (nécessaire pour SARSA)
                next_action = agent.choose_action(next_state) if not episode_done else None
                
                # 3. Mise à jour (Q-Learning ignorera next_action, SARSA l'utilisera)
                agent.update(state, action, reward, next_state, next_action)
                
                state = next_state
                action = next_action
                total_reward += reward
                done = episode_done
                
            agent.decay_epsilon(ep)
            
            # Logging
            ghosts_count = info.get('ghosts', 0)
            log_episode(ep, env.score, env.steps, agent.epsilon, ghosts_count)
            
            if chart:
                # On passe le niveau max atteint dans cet épisode
                chart.update(ep, env.score, agent.epsilon, ghosts_count, env.level)
            
            if ep % 100 == 0 or ep == 1:
                print(f"Ep {ep}/{episodes} | Score: {env.score} | Lvl: {env.level} | Eps: {agent.epsilon:.3f}")
                
            if ep % 500 == 0:
                agent.save_model()
                
    except KeyboardInterrupt:
        print("\nArrêt manuel détecté. Sauvegarde...")
    
    agent.save_model()
    if renderer: renderer.close()
    if chart: 
        chart.save_plot()
        chart.close()

def play():
    env = PacmanEnv()
    agent = QLearningAgent()
    agent.load_model()
    
    renderer = GameRenderer(env)
    clock = pygame.time.Clock()
    
    print("Lancement en mode DÉMO...")
    
    ep = 0
    running = True
    while running:
        ep += 1
        state = env.reset()
        action = agent.choose_action(state, training=False)
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                        running = False

            renderer.render(info_dict={'episode': ep, 'epsilon': 0.0})
            
            next_state, _, episode_done, info = env.step(action)
            
            if info.get("level_cleared", False):
                print(f"Niveau {env.level} terminé ! Passage au niveau {env.level+1}...")
                time.sleep(1)
                next_state = env.next_level()
                episode_done = False

            action = agent.choose_action(next_state, training=False)
            done = episode_done
            
            clock.tick(10) 
            
        print(f"Fin Épisode {ep} | Score: {env.score} | Niveau Atteint: {env.level}")
        time.sleep(0.5)
    
    renderer.close()

def main_menu():
    pygame.init()
    env = PacmanEnv() 
    renderer = GameRenderer(env)
    
    options = [
        f"Train {ALGORITHM} (Headless)", 
        f"Train {ALGORITHM} (Visual)", 
        "Play (Demo)", 
        "Quit"
    ]
    selected = 0
    
    while True:
        renderer.render_menu(selected, options)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    renderer.close() 
                    if selected == 0:
                        train(TOTAL_EPISODES, visual=False, graphics=True)
                    elif selected == 1:
                        train(TOTAL_EPISODES, visual=True, graphics=True)
                    elif selected == 2:
                        play()
                    elif selected == 3:
                        sys.exit()
                    return 

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=["train", "play"], required=True)
        parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
        parser.add_argument("--visual", action="store_true")
        parser.add_argument("--graphics", action="store_true")
        parser.add_argument("--load", action="store_true")
        args = parser.parse_args()
        
        if args.mode == "train":
            train(args.episodes, args.visual, args.graphics, args.load)
        else:
            play()
    else:
        main_menu()
