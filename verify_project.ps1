$ErrorActionPreference = "Stop"
if (!(Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Run .\setup.ps1 first."
    exit 1
}
& ".\.venv\Scripts\python.exe" scripts\healthcheck.py
& ".\.venv\Scripts\python.exe" scripts\validate_repo.py
& ".\.venv\Scripts\python.exe" -m pytest -q
