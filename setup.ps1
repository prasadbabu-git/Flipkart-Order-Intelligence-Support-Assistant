$ErrorActionPreference = "Stop"
Write-Host "=== Flipkart Order Intelligence - Setup ==="

if (!(Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "In VS Code select: .venv\Scripts\python.exe"
