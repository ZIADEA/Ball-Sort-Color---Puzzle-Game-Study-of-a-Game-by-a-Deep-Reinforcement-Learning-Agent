# Evaluer tous les modèles PPO et agréger les résultats dans un CSV unique.
# Usage : powershell -File scripts/eval_all.ps1
# Peut adapter $Algo, $Level, $Episodes si besoin.

$Algo = "ppo"
$Level = 1
$Episodes = 100
$Models = Get-ChildItem -Path "result\models\$Algo\*.zip" -ErrorAction SilentlyContinue

if (-not $Models) {
    Write-Error "Aucun modèle trouvé dans result\models\$Algo"
    exit 1
}

$EvalDir = "result/evaluations"
if (-not (Test-Path $EvalDir)) {
    New-Item -ItemType Directory -Path $EvalDir | Out-Null
}

$AllRows = @()

foreach ($m in $Models) {
    $out = Join-Path $EvalDir ("eval_{0}.csv" -f $m.BaseName)
    Write-Host "Évaluation de $($m.Name)..." -ForegroundColor Cyan
    try {
        python .\eval.py --model $m.FullName --algo $Algo --n-episodes $Episodes --level $Level --output $out
        if (Test-Path $out) {
            $AllRows += Import-Csv $out | Select-Object *, @{Name='model';Expression={$m.BaseName}}
        } else {
            Write-Warning "Fichier d'output manquant pour $($m.Name), évaluation ignorée."
        }
    } catch {
        Write-Warning "Échec sur $($m.Name) : $($_.Exception.Message)"
    }
}

$Aggregated = Join-Path $EvalDir "all_models_level$Level.csv"
if ($AllRows.Count -gt 0) {
    $AllRows | Export-Csv $Aggregated -NoTypeInformation
    Write-Host "Agrégation terminée : $Aggregated" -ForegroundColor Green
    
    # Affiche le meilleur sur le succès moyen
    $best = $AllRows |
        Group-Object model |
        Select-Object @{Name='model';Expression={$_.Name}},
                      @{Name='success_rate';Expression={($_.Group.success -as [double[]]).Average}} |
        Sort-Object success_rate -Descending |
        Select-Object -First 1
    if ($best) {
        Write-Host ("Meilleur modèle (succès moyen) : {0} ({1:P2})" -f $best.model, $best.success_rate) -ForegroundColor Yellow
    }
} else {
    Write-Warning "Aucun résultat agrégé."
}
