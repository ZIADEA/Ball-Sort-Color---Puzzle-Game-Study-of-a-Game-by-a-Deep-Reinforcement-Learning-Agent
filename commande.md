# Commandes Ball Sort Puzzle RL - Windows PowerShell

Ce fichier contient toutes les commandes prêtes à copier-coller pour Windows PowerShell.

## 📦 Installation

### Option 1: Environnement Conda (Recommandé)

```powershell
# Activer l'environnement conda existant
conda activate colorball

# Installer les dépendances du projet
pip install -r requirements.txt

# Vérifier l'installation
python -c "import stable_baselines3; import gymnasium; import torch; print('✓ Installation OK')"
```

### Option 2: Environnement Python virtuel

```powershell
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
.\venv\Scripts\Activate

# Installer les dépendances
pip install -r requirements.txt

# Vérifier l'installation
python -c "import stable_baselines3; import gymnasium; import torch; print('✓ Installation OK')"
```

---

## 🎓 Entraînement

### PPO (Recommandé - avec action masking)

```powershell
# Training rapide (test)
python train.py --algo ppo --timesteps 10000 --n-envs 4 --seed 42

# Training complet (100k steps)
python train.py --algo ppo --timesteps 100000 --n-envs 8 --seed 42

# Training long (500k steps)
python train.py --algo ppo --timesteps 1000000 --n-envs 8 --seed 42

# Avec niveau initial personnalisé
python train.py --algo ppo --timesteps 100000 --n-envs 8 --level 3 --seed 42

# Avec configuration custom
python train.py --algo ppo --config configs\custom_config.yaml --timesteps 100000
```

### A2C

```powershell
# Training rapide (test)
python train.py --algo a2c --timesteps 10000 --n-envs 4 --seed 42

# Training complet
python train.py --algo a2c --timesteps 100000 --n-envs 8 --seed 42

# Training long
python train.py --algo a2c --timesteps 500000 --n-envs 8 --seed 42
```

### DQN

```powershell
# Training rapide (test)
python train.py --algo dqn --timesteps 10000 --seed 42

# Training complet
python train.py --algo dqn --timesteps 100000 --seed 42

# Training long
python train.py --algo dqn --timesteps 500000 --seed 42
```

### Notes:
- Les modèles sont sauvegardés dans `result\models\[algo]\`
- Les logs TensorBoard dans `result\logs\[algo]\`
- Les checkpoints tous les 100k steps
- La config est sauvegardée dans `result\configs\`

---

## 📊 Évaluation

### Évaluation standard

```powershell
# Remplacer [MODEL_PATH] par le chemin réel, par exemple:
# result\models\ppo\ppo_level1_20260110_143000_final.zip

# PPO - 100 épisodes
python eval.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 100 --level 1

# A2C - 100 épisodes
python eval.py --model result\models\a2c\a2c_level1_YYYYMMDD_HHMMSS_final.zip --algo a2c --n-episodes 100 --level 1

# DQN - 100 épisodes
python eval.py --model result\models\dqn\dqn_level1_YYYYMMDD_HHMMSS_final.zip --algo dqn --n-episodes 100 --level 1
```

### Évaluation avec visualisation

```powershell
# ATTENTION: Beaucoup plus lent!
python eval.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 10 --render --level 1
```

### Évaluation sur différents niveaux

```powershell
# Niveau 1 (facile)
python eval.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 100 --level 1

# Niveau 3 (moyen)
python eval.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 100 --level 3

# Niveau 5 (difficile)
python eval.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 100 --level 5
```

### Avec output personnalisé

```powershell
python eval.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 100 --output result\evaluations\my_eval.csv
```

### Notes:
- Résultats CSV dans `result\evaluations\`
- Summary .txt généré automatiquement
- Métriques: success rate, steps, rewards breakdown

---

## 🎮 Démonstration avec PyGame

### Démonstration simple

```powershell
# Jouer 5 épisodes avec visualisation
python demo.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 5 --fps 2

# Jouer avec FPS plus rapide
python demo.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 5 --fps 5

# Un seul épisode lent (analyse détaillée)
python demo.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 1 --fps 1
```

### Enregistrement d'épisodes

```powershell
# Enregistrer en images PNG
python demo.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 3 --record --fps 2

# Enregistrer en GIF
python demo.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 3 --record-gif --fps 2

# Enregistrer PNG + GIF
python demo.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 3 --record --record-gif --fps 2
```

### Notes:
- Enregistrements dans `result\episodes\`
- Images: `result\episodes\episode_X\frame_XXXX.png`
- GIFs: `result\episodes\episode_X.gif`
- Fermer la fenêtre pygame pour interrompre

---

## 🔬 Analyses XAI

### Analyse standard (sans Integrated Gradients)

```powershell
# PPO - 10 épisodes
python xai_analysis.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-samples 10 --level 1

# A2C - 10 épisodes
python xai_analysis.py --model result\models\a2c\a2c_level1_YYYYMMDD_HHMMSS_final.zip --algo a2c --n-samples 10 --level 1

# DQN - 10 épisodes (visualise Q-values)
python xai_analysis.py --model result\models\dqn\dqn_level1_YYYYMMDD_HHMMSS_final.zip --algo dqn --n-samples 10 --level 1
```

### Analyse avec Integrated Gradients (plus lent)

```powershell
# PPO avec IG
python xai_analysis.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-samples 5 --use-ig --n-steps-ig 50 --level 1

# Avec plus de steps IG (meilleure qualité, plus lent)
python xai_analysis.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-samples 3 --use-ig --n-steps-ig 100 --level 1
```

### Analyse rapide (debugging)

```powershell
# Seulement 3 épisodes
python xai_analysis.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-samples 3 --level 1
```

### Notes:
- Outputs dans `result\xai\[algo]_level[X]_YYYYMMDD_HHMMSS\`
- Génère:
  - `action_dist_*.png` ou `q_values_*.png` (distributions/Q-values)
  - `trajectory_*.png` (visualisation états)
  - `reward_breakdown.png` (décomposition récompenses)
  - `ig_*.png` (Integrated Gradients, si --use-ig)
  - `rewards_breakdown.csv` (données brutes)

---

## 📈 TensorBoard

### Lancer TensorBoard

```powershell
# Monitoring de tous les algos
tensorboard --logdir result\logs

# Monitoring d'un algo spécifique
tensorboard --logdir result\logs\ppo

# Avec port personnalisé
tensorboard --logdir result\logs --port 6007
```

### Accès

- Ouvrir navigateur: http://localhost:6006
- Métriques disponibles: rewards, episode length, losses, curriculum level

### Notes:
- TensorBoard se met à jour automatiquement pendant l'entraînement
- `Ctrl+C` pour arrêter TensorBoard

---

## 🔧 Workflow Complet (Exemple)

### Pipeline complet PPO 100k steps

```powershell
# 1. Activer l'environnement
conda activate colorball

# 2. Entraîner PPO
python train.py --algo ppo --timesteps 100000 --n-envs 8 --seed 42

# 3. Lancer TensorBoard (dans un autre terminal)
tensorboard --logdir result\logs

# 4. Évaluer le modèle final
python eval.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 100 --level 1

# 5. Démonstration visuelle
python demo.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 5 --record-gif --fps 2

# 6. Analyses XAI
python xai_analysis.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-samples 10 --level 1
```

### Comparaison de plusieurs algorithmes

```powershell
# Entraîner les 3 algorithmes
python train.py --algo ppo --timesteps 100000 --n-envs 8 --seed 42
python train.py --algo a2c --timesteps 100000 --n-envs 8 --seed 42
python train.py --algo dqn --timesteps 100000 --seed 42

# Évaluer tous
python eval.py --model result\models\ppo\ppo_level1_YYYYMMDD_HHMMSS_final.zip --algo ppo --n-episodes 100 --level 1
python eval.py --model result\models\a2c\a2c_level1_YYYYMMDD_HHMMSS_final.zip --algo a2c --n-episodes 100 --level 1
python eval.py --model result\models\dqn\dqn_level1_YYYYMMDD_HHMMSS_final.zip --algo dqn --n-episodes 100 --level 1

# Comparer dans TensorBoard
tensorboard --logdir result\logs
```

---

## 🧹 Nettoyage

### Nettoyer les résultats

```powershell
# Supprimer tous les résultats (ATTENTION!)
Remove-Item -Recurse -Force result\*

# Supprimer seulement les logs
Remove-Item -Recurse -Force result\logs\*

# Supprimer seulement les épisodes enregistrés
Remove-Item -Recurse -Force result\episodes\*
```

---

## 🐛 Debugging

### Test rapide de l'environnement

```powershell
# Tester l'environnement seul
python -c "from envs import BallSortEnv; env = BallSortEnv(); obs, info = env.reset(); print('✓ Env OK'); env.close()"
```

### Test d'un entraînement minimal

```powershell
# 1000 steps seulement (très rapide)
python train.py --algo ppo --timesteps 1000 --n-envs 2 --seed 42
```

### Vérifier les dépendances

```powershell
# Lister les packages installés
pip list

# Vérifier versions spécifiques
python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
```

### Logs verbeux

```powershell
# Tous les scripts supportent les logs Python
$env:PYTHONVERBOSE=1
python train.py --algo ppo --timesteps 10000
```

---

## 📝 Notes Importantes

### Chemins de modèles

Les chemins de modèles suivent ce pattern:
```
result\models\[algo]\[algo]_level[X]_YYYYMMDD_HHMMSS_final.zip
```

Exemple réel:
```
result\models\ppo\ppo_level1_20260110_143527_final.zip
```

### Remplacer les timestamps

Dans toutes les commandes ci-dessus, remplacez:
- `YYYYMMDD_HHMMSS` par le timestamp réel de votre modèle

Ou utilisez le wildcard pour le dernier modèle créé:
```powershell
# PowerShell: récupérer le dernier modèle PPO
$latest_ppo = Get-ChildItem result\models\ppo\*_final.zip | Sort-Object LastWriteTime -Descending | Select-Object -First 1
python eval.py --model $latest_ppo.FullName --algo ppo --n-episodes 100
```

### Performance

- **PPO**: Le plus rapide avec vectorization (n_envs=8)
- **A2C**: Similaire à PPO
- **DQN**: Plus lent (single env), mais plus stable
- **XAI avec IG**: Très lent, limiter n-samples

### Espace disque

Estimations:
- Checkpoint PPO: ~10-20 MB
- Logs TensorBoard: ~50-100 MB / 100k steps
- GIF épisode: ~1-5 MB
- Figures XAI: ~100-500 KB chacune

---

## ✅ Checklist de Validation

```powershell
# 1. Installation
conda activate colorball
pip install -r requirements.txt

# 2. Test environnement
python -c "from envs import BallSortEnv; env = BallSortEnv(); env.reset(); env.close(); print('✓')"

# 3. Training rapide
python train.py --algo ppo --timesteps 1000 --n-envs 2

# 4. Évaluation
python eval.py --model result\models\ppo\*_final.zip --algo ppo --n-episodes 10

# 5. Demo
python demo.py --model result\models\ppo\*_final.zip --algo ppo --n-episodes 1

# 6. XAI
python xai_analysis.py --model result\models\ppo\*_final.zip --algo ppo --n-samples 3

# Si tout fonctionne: ✓ Projet opérationnel!
```

---

**Fin du guide de commandes**

Pour toute question, voir README.md ou les logs d'erreur.
