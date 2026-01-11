# Ball Sort Puzzle - Reinforcement Learning Project

Entraînement d'agents RL (PPO, A2C, DQN) sur le puzzle Ball Sort (Color Sort) avec Gymnasium et Stable-Baselines3, incluant analyses XAI (Explainable AI).

## 🎯 Caractéristiques

- **Environnement Gymnasium** complet avec action masking
- **Curriculum Learning** automatique basé sur le taux de succès
- **Algorithmes RL**: PPO (avec MaskablePPO), A2C, DQN
- **Génération de niveaux solvables** via Reverse Shuffle
- **Visualisation PyGame** en temps réel
- **Analyses XAI** complètes:
  - Distributions de probabilités d'actions
  - Visualisations Q-values (DQN)
  - Integrated Gradients (Captum)
  - Décomposition des récompenses
  - Trajectoires d'épisodes
- **Configuration flexible** via YAML
- **Reproductibilité** avec seeds et sauvegarde de configs

## 📁 Structure du Projet

```
projetRL/
├── envs/                    # Environnement Ball Sort
│   └── ball_sort_env.py
├── agents/                  # Wrappers pour algorithmes RL
│   └── agent_factory.py
├── utils/                   # Utilitaires
│   └── config.py
├── configs/                 # Configurations par défaut
│   └── default_config.yaml
├── result/                  # Tous les artefacts
│   ├── models/             # Modèles entraînés + checkpoints
│   ├── logs/               # TensorBoard logs
│   ├── evaluations/        # Métriques d'évaluation (CSV)
│   ├── xai/                # Analyses XAI (figures)
│   ├── episodes/           # Enregistrements d'épisodes
│   └── configs/            # Configs sauvegardées par run
├── train.py                # Script d'entraînement
├── eval.py                 # Script d'évaluation
├── demo.py                 # Démo avec visualisation PyGame
├── xai_analysis.py         # Analyses XAI
├── requirements.txt        # Dépendances Python
└── README.md              # Ce fichier
```

## 🚀 Installation

### 1. Environnement Conda (recommandé)

```powershell
# L'environnement 'colorball' est déjà créé avec les dépendances de base
conda activate colorball

# Installer les dépendances supplémentaires du projet
pip install -r requirements.txt
```

### 2. Environnement virtuel Python (alternative)

```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

## 📝 Configuration

La configuration par défaut se trouve dans `configs/default_config.yaml`. Vous pouvez:
- Modifier ce fichier directement
- Créer un fichier custom et le passer via `--config`
- Surcharger des paramètres via arguments CLI

Paramètres clés:
- **Environnement**: niveaux, curriculum, max_steps
- **Training**: timesteps, n_envs, seed, save_freq
- **Algorithmes**: hyperparamètres PPO/A2C/DQN
- **XAI**: échantillons, Integrated Gradients steps

## 🎮 Utilisation

### Entraînement

```powershell
# PPO (recommandé, avec action masking)
python train.py --algo ppo --timesteps 100000 --n-envs 8

# A2C
python train.py --algo a2c --timesteps 100000 --n-envs 8

# DQN (single env)
python train.py --algo dqn --timesteps 100000

# Avec niveau initial et seed personnalisés
python train.py --algo ppo --level 3 --seed 123 --timesteps 200000
```

Les modèles sont sauvegardés dans `result/models/[algo]/`.

### Évaluation

```powershell
# Évaluer un modèle entraîné
python eval.py --model result/models/ppo/ppo_level1_..._final.zip --algo ppo --n-episodes 100 --level 1

# Avec visualisation (plus lent)
python eval.py --model result/models/ppo/ppo_level1_..._final.zip --algo ppo --render
```

Résultats CSV dans `result/evaluations/`.

### Démonstration avec PyGame

```powershell
# Jouer 5 épisodes avec visualisation
python demo.py --model result/models/ppo/ppo_level1_..._final.zip --algo ppo --n-episodes 5 --fps 2

# Enregistrer les épisodes en images
python demo.py --model result/models/ppo/ppo_level1_..._final.zip --algo ppo --record

# Enregistrer en GIF
python demo.py --model result/models/ppo/ppo_level1_..._final.zip --algo ppo --record-gif
```

Enregistrements dans `result/episodes/`.

### Analyses XAI

```powershell
# Analyse complète (sans Integrated Gradients)
python xai_analysis.py --model result/models/ppo/ppo_level1_..._final.zip --algo ppo --n-samples 10

# Avec Integrated Gradients (plus lent, nécessite Captum)
python xai_analysis.py --model result/models/ppo/ppo_level1_..._final.zip --algo ppo --use-ig --n-steps-ig 50

# Pour DQN (visualise Q-values)
python xai_analysis.py --model result/models/dqn/dqn_level1_..._final.zip --algo dqn --n-samples 10
```

Figures générées dans `result/xai/`.

## 📊 Monitoring avec TensorBoard

```powershell
# Lancer TensorBoard
tensorboard --logdir result/logs

# Accéder à http://localhost:6006
```

Métriques disponibles:
- Récompense par épisode
- Longueur d'épisode
- Loss (policy, value, entropy)
- Niveau de curriculum

## 🔬 Environnement Ball Sort

### Description

Puzzle de tri de balles colorées dans des tubes. Objectif: trier toutes les couleurs pour que chaque tube soit homogène.

### Espace d'observation

Matrice `(N_max=14, H=4)` de type `int32`:
- `-1`: padding (tube inexistant)
- `0`: cellule vide
- `1-12`: ID de couleur

### Espace d'actions

`Discrete(N_max * N_max)` décodé en:
- `source = action // N_max`
- `dest = action % N_max`

**Action masking** pour empêcher les actions invalides.

### Récompenses

- `step_time = -0.01` (pénalité par étape)
- `progress = Δ(purity_score)` (progrès vers solution)
- `bonus_tube_complete = +5` (tube complété)
- `bonus_win = +100 + 5×level` (victoire)
- `penalty_blocked = -10` (bloqué, aucun coup valide)

### Curriculum Learning

- Augmente automatiquement le niveau quand `success_rate ≥ 0.8` sur 100 épisodes
- Niveau 1-14 (3-14 tubes, 1-12 couleurs)

## 🧪 Tests et Validation

```powershell
# Installer dépendances de test
pip install pytest pytest-cov

# Lancer les tests (à créer)
pytest tests/ -v --cov=.
```

## 🛠️ Développement

### Formattage du code

```powershell
pip install black flake8
black .
flake8 .
```

### Type checking

```powershell
pip install mypy
mypy train.py eval.py demo.py xai_analysis.py
```

## 📈 Résultats Attendus

- **PPO avec masking**: meilleure performance (>80% success rate niveau 1-3)
- **A2C**: bon compromis vitesse/performance
- **DQN**: plus lent à converger mais stable

## 🐛 Troubleshooting

### Erreur "sb3-contrib not found"
```powershell
pip install sb3-contrib
```

### Erreur pygame/render
- Assurez-vous que pygame est installé
- Utilisez `--render` uniquement pour démo/eval, pas training

### Erreur Captum
```powershell
pip install captum
```

### Performance lente
- Réduire `n_envs` pour PPO/A2C
- Utiliser CPU si GPU non disponible
- Réduire `timesteps` pour tests rapides

## 📚 Références

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [sb3-contrib (MaskablePPO)](https://sb3-contrib.readthedocs.io/)
- [Captum (XAI)](https://captum.ai/)

## 📄 Licence

Projet éducatif - usage libre.

## 👨‍💻 Auteur

Projet ML Senior Engineer - Ball Sort Puzzle RL

---

**Note**: Ce projet est structuré pour être professionnel, reproductible et extensible. Toutes les commandes sont testées sur Windows PowerShell.
