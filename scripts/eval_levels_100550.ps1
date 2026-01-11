# Evaluates the model ppo_level1_20260111_100550_final.zip on levels 1-14,
# then aggregates per-level metrics into a single CSV.

$ErrorActionPreference = "Stop"

# Paths
$Model = "result\models\ppo\ppo_level1_20260111_100550_final.zip"
$EvalDir = "result\evaluations\multi_level_100550"

# Resolve python executable from current session (uses active conda env)
$PythonExe = (Get-Command python).Source
if (-not $PythonExe) { $PythonExe = "python" }
Write-Host "Using python: $PythonExe" -ForegroundColor Cyan

# Ensure output directory exists
if (-not (Test-Path $EvalDir)) {
    New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null
}

Write-Host "==> Evaluating model on levels 1..14 (100 episodes each)..." -ForegroundColor Cyan

# Run eval.py for each level
for ($lvl = 1; $lvl -le 14; $lvl++) {
    $out = Join-Path $EvalDir ("eval_level{0}.csv" -f $lvl)
    Write-Host ("Level {0} -> {1}" -f $lvl, $out)
    & $PythonExe eval.py --model $Model --algo ppo --n-episodes 100 --level $lvl --output $out
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python failed at level $lvl (exit code $LASTEXITCODE). Aborting." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "==> Aggregating results..." -ForegroundColor Cyan

# Aggregate per-level means into one CSV (python -c for PowerShell compatibility)
& $PythonExe -c @"
import pandas as pd, glob, os
eval_dir = r'result/evaluations/multi_level_100550'
rows = []
for path in sorted(glob.glob(os.path.join(eval_dir, 'eval_level*.csv'))):
    df = pd.read_csv(path)
    level = int(os.path.splitext(os.path.basename(path))[0].split('level')[1])
    rows.append({
        'level': level,
        'episodes': len(df),
        'success_rate': df['success'].mean(),
        'mean_steps': df['steps'].mean(),
        'mean_reward': df['reward'].mean(),
        'time_penalty': df['time_penalty'].mean(),
        'purity_reward': df['purity_reward'].mean(),
        'complete_reward': df['complete_reward'].mean(),
        'win_reward': df['win_reward'].mean(),
        'blocked_penalty': df['blocked_penalty'].mean(),
    })
out = pd.DataFrame(rows).sort_values('level')
out_path = os.path.join(eval_dir, 'summary_levels_1_14.csv')
out.to_csv(out_path, index=False)
print(out)
print(f"\nSaved {out_path}")
"@
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python aggregation failed (exit code $LASTEXITCODE). Aborting." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "==> Done. Per-level CSVs in $EvalDir and summary_levels_1_14.csv created." -ForegroundColor Green
