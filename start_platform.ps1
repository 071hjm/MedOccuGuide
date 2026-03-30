$ErrorActionPreference = "Stop"

$projectDir = "D:\codex"
$pythonExe = "C:\Users\hjm\AppData\Local\Programs\Python\Python311\python.exe"
$appUrl = "http://127.0.0.1:7860"

Set-Location $projectDir

if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$listener = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue

if (-not $listener) {
    Start-Process -FilePath $pythonExe -ArgumentList "D:\codex\app.py" -WorkingDirectory $projectDir -WindowStyle Minimized
    Start-Sleep -Seconds 6
}

Start-Process $appUrl
