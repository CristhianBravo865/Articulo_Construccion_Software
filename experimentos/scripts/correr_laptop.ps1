# Corrida completa de Qwen2.5-Coder-1.5B en la laptop (reanudable: salta casos ya hechos).
# Uso:  powershell -File correr_laptop.ps1
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$py = "$env:USERPROFILE\.venvs\cuantizacion\Scripts\python.exe"
Set-Location $PSScriptRoot

# Orden: E2 (49 casos) antes que E1 (150); Q-8 al final porque es el mas lento.
foreach ($exp in "E2", "E1") {
    foreach ($cfg in "Q-16", "Q-4", "Q-8") {
        & $py 02_generar_parches.py --experimento $exp --config $cfg --hardware laptop 2>&1 |
            Select-String -NotMatch "Warning|warn"
    }
}
& $py 03_evaluar.py (Get-ChildItem ..\resultados\gen_*_laptop.jsonl).FullName 2>&1 |
    Select-String -NotMatch "WARNING"
& $py 04_analizar.py
