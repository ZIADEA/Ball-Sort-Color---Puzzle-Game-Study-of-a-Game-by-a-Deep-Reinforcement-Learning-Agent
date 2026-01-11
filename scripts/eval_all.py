"""
Évaluation batch des modèles PPO et sélection du meilleur.

Usage :
    python scripts/eval_all.py --models-glob "result/models/ppo/*.zip" --algo ppo --level 1 --episodes 100

Produit :
    - Un CSV agrégé avec les métriques par modèle (success_rate, reward, steps).
    - Affiche le meilleur modèle (taux de succès max, puis récompense moyenne en cas d’égalité).
"""

import argparse
from pathlib import Path
import pandas as pd

import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.config import ConfigManager
from envs import BallSortEnv
from envs.wrappers import OneHotObservationWrapper
from agents.agent_factory import mask_fn
from sb3_contrib.common.wrappers import ActionMasker
from eval import load_model, evaluate_agent  # réutilise la logique d’éval existante
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Évaluer en batch les modèles et choisir le meilleur")
    parser.add_argument("--models-glob", type=str, default="result/models/ppo/*.zip", help="Glob des modèles à évaluer")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c", "dqn"], help="Algorithme")
    parser.add_argument("--level", type=int, default=1, help="Niveau à évaluer")
    parser.add_argument("--episodes", type=int, default=100, help="Nombre d'épisodes par modèle")
    parser.add_argument("--output", type=str, default=None, help="Chemin du CSV agrégé (par défaut result/evaluations/all_models_levelX.csv)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ConfigManager()

    models = sorted(Path().glob(args.models_glob))
    if not models:
        logger.error(f"Aucun modèle trouvé pour le glob : {args.models_glob}")
        return

    # Agrégation
    rows = []

    for model_path in models:
        logger.info(f"Évaluation de {model_path.name}...")
        try:
            model, use_masking = load_model(str(model_path), args.algo)
        except Exception as e:
            logger.warning(f"Chargement impossible pour {model_path.name} : {e}")
            continue

        env_kwargs = {
            "n_max": config.get("env.n_max", 14),
            "height": config.get("env.height", 4),
            "initial_level": args.level,
            "max_level": args.level,
            "max_steps": config.get("env.max_steps", 500),
            "seed": config.get("training.seed", 42),
        }

        base_env = BallSortEnv(**env_kwargs)
        env = OneHotObservationWrapper(base_env) if config.get("env.use_one_hot", True) else base_env
        if use_masking:
            env = ActionMasker(env, mask_fn)

        try:
            df = evaluate_agent(
                model=model,
                env=env,
                n_episodes=args.episodes,
                deterministic=True,
                render=False,
                use_masking=use_masking,
            )
        except Exception as e:
            logger.warning(f"Échec de l'évaluation pour {model_path.name} : {e}")
            env.close()
            continue

        env.close()

        success_rate = df["success"].mean()
        mean_reward = df["reward"].mean()
        std_reward = df["reward"].std()
        mean_steps = df["steps"].mean()
        std_steps = df["steps"].std()

        rows.append({
            "model": model_path.name,
            "path": str(model_path),
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_steps": mean_steps,
            "std_steps": std_steps,
        })

    if not rows:
        logger.error("Aucun résultat agrégé (modèles corrompus ou erreurs d'évaluation).")
        return

    agg_df = pd.DataFrame(rows)
    output_path = Path(args.output) if args.output else Path("result/evaluations") / f"all_models_level{args.level}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(output_path, index=False)
    logger.info(f"CSV agrégé écrit : {output_path}")

    best = agg_df.sort_values(["success_rate", "mean_reward"], ascending=False).iloc[0]
    logger.info(f"Meilleur modèle : {best['model']} (success={best['success_rate']:.2%}, reward={best['mean_reward']:.2f})")
    print(f"\nMeilleur modèle : {best['model']} (success={best['success_rate']:.2%}, reward={best['mean_reward']:.2f})")


if __name__ == "__main__":
    main()
