# Relance ComfyUI, Chatterbox et le panneau de contrôle webui, chacun dans sa
# propre fenêtre — tue d'abord ce qui tourne déjà sur leur port, pour ne
# jamais laisser une vieille instance (code webui modifié, serveur planté à
# moitié) traîner en silence. Chemins d'installation identiques à RUN.md ; à
# modifier ici si jamais ils changent.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$ComfyDir = "C:\ComfyUI"
$ChatterboxDir = "C:\ChatterboxTTS"

function Test-PortOpen($port) {
    $null -ne (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

function Stop-OnPort($port) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        Write-Host "  arret du process existant sur le port $port (PID $($c.OwningProcess))"
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    if (-not $conns) { return }
    $deadline = (Get-Date).AddSeconds(10)
    while ((Test-PortOpen $port) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 200
    }
}

function Start-InWindow($title, $workDir, $command) {
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "`$host.ui.RawUI.WindowTitle = '$title'; Set-Location '$workDir'; $command"
    ) | Out-Null
}

Write-Host "ComfyUI..."
Stop-OnPort 8188
if (Test-Path "$ComfyDir\venv\Scripts\python.exe") {
    Start-InWindow "ComfyUI" $ComfyDir ".\venv\Scripts\python.exe main.py"
} else {
    Write-Host "  introuvable dans $ComfyDir -- ignore (FLUX.2/SDXL indisponibles)"
}

Write-Host "Chatterbox..."
Stop-OnPort 8189
if (Test-Path "$ChatterboxDir\venv\Scripts\python.exe") {
    Start-InWindow "Chatterbox" $ChatterboxDir ".\venv\Scripts\python.exe server.py"
} else {
    Write-Host "  introuvable dans $ChatterboxDir -- ignore (voix Chatterbox indisponible)"
}

Write-Host "Webui..."
Stop-OnPort 8765
Start-InWindow "Pipeline webui" $root "`$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe webui\server.py"

Write-Host ""
Write-Host "Attente que le webui reponde..."
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline) {
    if (Test-PortOpen 8765) {
        Start-Sleep -Seconds 1
        Start-Process "http://127.0.0.1:8765"
        Write-Host "Pret : http://127.0.0.1:8765"
        exit 0
    }
    Start-Sleep -Milliseconds 500
}
Write-Host "Le webui ne repond pas apres 30s -- regarder sa fenetre pour une erreur."
