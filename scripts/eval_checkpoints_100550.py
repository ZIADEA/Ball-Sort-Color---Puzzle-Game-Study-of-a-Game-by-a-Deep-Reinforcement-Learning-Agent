"""
Évalue tous les checkpoints du run ppo_level1_20260111_100550 (100k -> 900k)
plus le modèle final, sur les niveaux 1..14 (100 épisodes chacun), et produit
un résumé global pour comparer les modèles.
"""

import sys
from pathlib import Path
import pandas as pd

# Rendre le projet importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from eval import load_model, evaluate_agent  # noqa: E402
from envs import BallSortEnv  # noqa: E402
from envs.wrappers import OneHotObservationWrapper  # noqa: E402
from agents.agent_factory import mask_fn  # noqa: E402
from utils.config import ConfigManager  # noqa: E402

try:
    from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
except ImportError:
    ActionMasker = None


def evaluate_model(model_path: Path, config: ConfigManager, levels, episodes: int, out_dir: Path):
    model, use_masking = load_model(str(model_path), "ppo")
    per_level_rows = []

    for lvl in levels:
        env_kwargs = {
            "n_max": config.get("env.n_max", 14),
            "height": config.get("env.height", 4),
            "initial_level": lvl,
            "max_level": lvl,
            "max_steps": config.get("env.max_steps", 500),
            "seed": 42,
            "render_mode": None,
        }
        base_env = BallSortEnv(**env_kwargs)
        env = OneHotObservationWrapper(base_env) if config.get("env.use_one_hot", True) else base_env
        if use_masking and ActionMasker is not None:
            env = ActionMasker(env, mask_fn)

        df = evaluate_agent(
            model=model,
            env=env,
            n_episodes=episodes,
            deterministic=True,
            render=False,
            use_masking=use_masking,
        )
        env.close()

        per_level_rows.append(
            {
                "level": lvl,
                "episodes": len(df),
                "success_rate": df["success"].mean(),
                "mean_steps": df["steps"].mean(),
                "mean_reward": df["reward"].mean(),
                "time_penalty": df["time_penalty"].mean(),
                "purity_reward": df["purity_reward"].mean(),
                "complete_reward": df["complete_reward"].mean(),
                "win_reward": df["win_reward"].mean(),
                "blocked_penalty": df["blocked_penalty"].mean(),
            }
        )

        # Sauvegarde par niveau
        df.to_csv(out_dir / f"eval_level{lvl}.csv", index=False)

    summary = pd.DataFrame(per_level_rows).sort_values("level")
    summary.to_csv(out_dir / "summary_levels_1_14.csv", index=False)

    # Stats globales (moyenne sur les niveaux)
    global_row = {
        "model": model_path.stem,
        "mean_success_rate": summary["success_rate"].mean(),
        "mean_reward": summary["mean_reward"].mean(),
        "mean_steps": summary["mean_steps"].mean(),
    }
    return global_row


def main():
    # Checkpoints à tester (100k -> 900k) + final
    base_dir = project_root / "result" / "models" / "ppo" / "ppo_level1_20260111_100550"
    checkpoints = sorted(base_dir.glob("ppo_*_steps.zip"))
    final_model = project_root / "result" / "models" / "ppo" / "ppo_level1_20260111_100550_final.zip"
    model_paths = checkpoints + [final_model]

    out_root = project_root / "result" / "evaluations" / "checkpoints_100550"
    out_root.mkdir(parents=True, exist_ok=True)

    config = ConfigManager()
    levels = range(1, 15)
    episodes = 100

    overall_rows = []
    for m in model_paths:
        if not m.exists():
            print(f"Skip missing model: {m}")
            continue
        print(f"==> Evaluating {m.name}")
        model_out_dir = out_root / m.stem
        model_out_dir.mkdir(parents=True, exist_ok=True)
        row = evaluate_model(m, config, levels, episodes, model_out_dir)
        overall_rows.append(row)

    if overall_rows:
        overall = pd.DataFrame(overall_rows).sort_values("mean_success_rate", ascending=False)
        overall_path = out_root / "overall_models_summary.csv"
        overall.to_csv(overall_path, index=False)
        print("\nOverall summary:")
        print(overall)
        best = overall.iloc[0]
        print(f"\nBest model: {best['model']} (mean_success_rate={best['mean_success_rate']:.3f})")
    else:
        print("No models evaluated.")


if __name__ == "__main__":
    main()
