# PAC-MAN AR - Reinforcement Learning

Ce projet est une implémentation d'un agent Pac-Man apprenant par renforcement (Q-Learning, SARSA, Double Q-Learning). L'environnement est construit avec Pygame et permet de visualiser l'apprentissage en temps réel ou via des graphiques de performance.

## Prérequis

Assurez-vous d'avoir Python installé. Les dépendances nécessaires sont :

- `pygame`
- `numpy`
- `matplotlib`

## Installation

1.  Clonez ce dépôt :
    ```bash
    git clone https://github.com/BWT-ESGI/PACMAN-AR.git
    cd PACMAN-AR
    ```

2.  Installez les dépendances :
    ```bash
    pip install pygame numpy matplotlib
    ```

## Utilisation

Le point d'entrée du projet est le fichier `main.py`. Vous pouvez l'utiliser via un menu interactif ou en ligne de commande.

### 1. Menu Interactif
Lancez simplement le script sans arguments pour accéder au menu :
```bash
python main.py
```
Utilisez les flèches **HAUT/BAS** pour naviguer et **ENTRÉE** pour valider.
- **Train (Headless)** : Entraînement rapide sans rendu graphique du jeu.
- **Train (Visual)** : Entraînement avec visualisation du jeu (plus lent).
- **Play (Demo)** : Regarder l'agent jouer avec le modèle sauvegardé.

### 2. Ligne de Commande (CLI)
Vous pouvez lancer des tâches spécifiques avec des arguments.

#### Entraînement
Pour lancer un entraînement :
```bash
python main.py --mode train --episodes 5000
```

**Options disponibles :**
- `--visual` : Active le rendu du jeu pendant l'entraînement.
- `--graphics` : Génère et sauvegarde les courbes d'apprentissage (Score, Epsilon, etc.) à la fin.
- `--load` : Charge le modèle existant (`models/qtable.pkl`) avant de commencer (utile pour continuer un entraînement).

**Exemple complet :**
```bash
python main.py --mode train --episodes 1000 --visual --graphics --load
```

#### Mode Démo (Play)
Pour voir l'agent jouer en utilisant le modèle entraîné :
```bash
python main.py --mode play
```

## Configuration

Le fichier `config.py` contient tous les paramètres ajustables du projet :

- **Paramètres de Jeu** : Vitesse (`FPS`), taille des tuiles, carte (`GAME_MAP`).
- **Hyperparamètres RL** :
    - `ALGORITHM` : Choix de l'algo (`"QLEARNING"`, `"SARSA"`, `"DOUBLE_Q"`).
    - `ALPHA` (Taux d'apprentissage), `GAMMA` (Facteur d'actualisation).
    - `EPSILON` (Exploration vs Exploitation).
- **Récompenses (Reward Shaping)** : Modifiez `R_DOT`, `R_DEATH`, `R_WIN`, etc. pour influencer le comportement de l'agent.

## Structure du Projet

- `main.py` : Script principal, gestion du menu et des boucles d'entraînement/jeu.
- `game_env.py` : Environnement Pac-Man (règles, déplacements, gestion des fantômes).
- `agent.py` : Logique de l'agent (Q-Table, choix d'action, mise à jour Q-Learning/SARSA).
- `graphics.py` : Gestion de l'affichage Pygame et des graphiques Matplotlib.
- `config.py` : Fichier de configuration centralisé.
- `models/` : Dossier de sauvegarde pour le modèle (`qtable.pkl`), les logs (`training_log.csv`) et les graphiques.
