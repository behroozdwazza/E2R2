$ErrorActionPreference = "Stop"

$Python = "C:\Users\davazdab\.conda\envs\knime_python\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python -m streamlit run "$PSScriptRoot\streamlit_app.py"
