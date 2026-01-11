"""
Agrège les métriques par niveau pour un modèle évalué.

Par défaut, lit les CSV par niveau produits dans
result/evaluations/checkpoints_100550/ppo_1000000_steps/ eval_levelX.csv
et écrit un résumé unique (une ligne par niveau).
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Agrège les métriques par niveau pour un modèle évalué"
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(project_root / "result" / "evaluations" / "checkpoints_100550" / "ppo_1000000_steps"),
        help="Dossier contenant les eval_levelX.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Chemin du CSV de sortie (par défaut summary_levels_1_14.csv dans eval-dir)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="1-14",
        help="Plage de niveaux, ex: 1-14 (inclus)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    if "-" in args.levels:
        start, end = args.levels.split("-")
        levels = range(int(start), int(end) + 1)
    else:
        levels = [int(x) for x in args.levels.split(",")]

    rows = []
    for lvl in levels:
        csv_path = eval_dir / f"eval_level{lvl}.csv"
        if not csv_path.exists():
            print(f"Skip missing {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        rows.append(
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

    summary = pd.DataFrame(rows).sort_values("level")
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = eval_dir / "summary_levels_aggregated.csv"
    summary.to_csv(out_path, index=False)
    print(summary)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
