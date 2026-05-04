$ErrorActionPreference = "Stop"

$Python = "C:\Users\davazdab\.conda\envs\knime_python\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python "$PSScriptRoot\llm_experiment_app.py"
