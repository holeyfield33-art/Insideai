# Launch backend + frontend dev servers in two PowerShell windows.
$root = Split-Path -Parent $PSScriptRoot

Start-Process powershell -ArgumentList @(
  "-NoExit", "-Command",
  "& '$root\backend\.venv\Scripts\python.exe' -m uvicorn app.main:app --app-dir '$root\backend' --host 127.0.0.1 --port 8000"
)

Start-Process powershell -ArgumentList @(
  "-NoExit", "-Command",
  "Set-Location '$root\frontend'; npm run dev"
)

Write-Host "Backend  -> http://127.0.0.1:8000  (ws://127.0.0.1:8000/ws)"
Write-Host "Frontend -> http://localhost:3000"
