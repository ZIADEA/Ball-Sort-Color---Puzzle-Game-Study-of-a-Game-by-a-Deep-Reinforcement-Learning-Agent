"""
Evaluate the good model (ppo_level1_20260111_100550_final.zip) on levels 1..14
and aggregate per-level metrics into a summary CSV.
"""

import sys
from pathlib import Path
import pandas as pd

# Make project root importable
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


def main():
    model_path = project_root / "result" / "models" / "ppo" / "ppo_level1_20260111_100550_final.zip"
    output_dir = project_root / "result" / "evaluations" / "multi_level_100550"
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = 100
    levels = range(1, 15)  # 1..14

    # Load config and model
    config = ConfigManager()
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

        out_csv = output_dir / f"eval_level{lvl}.csv"
        print(f"Evaluating level {lvl} -> {out_csv}")

        df = evaluate_agent(
            model=model,
            env=env,
            n_episodes=episodes,
            deterministic=True,
            render=False,
            use_masking=use_masking,
        )
        df.to_csv(out_csv, index=False)

        # Aggregate for this level
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

        env.close()

    summary = pd.DataFrame(per_level_rows).sort_values("level")
    summary_path = output_dir / "summary_levels_1_14.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSummary:")
    print(summary)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
