import pygame
import csv
import argparse
import time
import sys
from game_env import PacmanEnv
from agent import QLearningAgent
from graphics import GameRenderer, TrainingChart
from config import *

if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Score", "Steps", "Epsilon", "GhostsEaten"])

def train(episodes, visual=False, graphics=False, load=False):
    env = PacmanEnv()
    agent = QLearningAgent()

    loaded_history = None
    start_episode = 0
    
    if load:
        loaded_history = agent.load_model()
        if loaded_history:
            start_episode = loaded_history['episodes'][-1] + 1
            print(f"Reprise de l'entraînement à l'épisode {start_episode}")


    renderer = GameRenderer(env) if visual else None
    chart = TrainingChart(visual=graphics) 

    if loaded_history:
        chart.episodes = loaded_history.get('episodes', [])
        chart.scores = loaded_history.get('scores', [])
        chart.avg_scores = loaded_history.get('avg_scores', [])
        chart.epsilons = loaded_history.get('epsilons', [])
        
        chart.ghosts_eaten = loaded_history.get('ghosts_eaten', []) 
        chart.avg_ghosts = loaded_history.get('avg_ghosts', [])
        
        chart.max_levels = loaded_history.get('max_levels', []) 
        chart.avg_levels = loaded_history.get('avg_levels', [])
     
    print(f"Démarrage de l'entraînement ({ALGORITHM}) pour {episodes} épisodes...")
    
    with open(LOG_FILE, 'a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        
        try:
            for ep in range(start_episode, start_episode + episodes):
                state = env.reset()
                
                action = agent.choose_action(state)
                
                done = False
                total_reward = 0
                
                while not done:
                    if visual and renderer:
                        renderer.render(info_dict={'episode': ep, 'epsilon': agent.epsilon})
                    
                    next_state, reward, episode_done, info = env.step(action)
                    
                    if info.get("level_cleared", False):
                        next_state = env.next_level()
                        episode_done = False 
                    
                    next_action = agent.choose_action(next_state) if not episode_done else None
                    
                    agent.update(state, action, reward, next_state, next_action)
                    
                    state = next_state
                    action = next_action
                    total_reward += reward
                    done = episode_done
                    
                agent.decay_epsilon(ep)
                
                ghosts_count = info.get('ghosts', 0)
                log_writer.writerow([ep, env.score, env.steps, agent.epsilon, ghosts_count])
                
                if chart:
                    chart.update(ep, env.score, agent.epsilon, ghosts_count, env.level)
                
                if ep % 100 == 0 or ep == 1:
                    print(f"Ep {ep}/{start_episode + episodes} | Score: {env.score} | Lvl: {env.level} | Eps: {agent.epsilon:.3f} | Alpha: {agent.alpha:.3f} | Taille Q-Table: {len(agent.q_table)}")
                if ep % 500 == 0:
                    pass 
                    
        except KeyboardInterrupt:
            print("\nArrêt manuel détecté. Sauvegarde...")

    history_data = {
        'episodes': chart.episodes,
        'scores': chart.scores,
        'avg_scores': chart.avg_scores,
        'epsilons': chart.epsilons,
        'ghosts_eaten': chart.ghosts_eaten,
        'avg_ghosts': chart.avg_ghosts,
        'avg_levels': chart.avg_levels,
        'max_levels': chart.max_levels
    }
    
    agent.save_model(history=history_data)
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
