<#
.SYNOPSIS
Creates a clean Python 3.11 environment for the project on Windows.

.DESCRIPTION
The script always calls the Python executable stored in .venv. This prevents
packages installed globally by Microsoft Store Python from leaking into the
project. Use -Recreate after a binary NumPy or pandas incompatibility.
#>

[CmdletBinding()]
param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if ($Recreate -and (Test-Path $VenvPath)) {
    Write-Host "Suppression de l'environnement virtuel existant..."
    Remove-Item -Recurse -Force $VenvPath
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Création de l'environnement Python 3.11..."
    $EnvironmentCreated = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv $VenvPath
        $EnvironmentCreated = ($LASTEXITCODE -eq 0)
    }

    if (-not $EnvironmentCreated) {
        & python -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

if (-not (Test-Path $VenvPython)) {
    throw "Impossible de créer .venv. Vérifiez que Python 3.11 est installé."
}

Write-Host "Installation des dépendances isolées..."
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$EditableTarget = "$ProjectRoot[dev]"
& $VenvPython -m pip install -e $EditableTarget
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Fail early if a compiled dependency is incompatible with NumPy.
& $VenvPython -c "import certifi, numpy, pandas; print(f'NumPy {numpy.__version__} | pandas {pandas.__version__}'); print(f'CA bundle: {certifi.where()}')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Environnement prêt. Utilisez .\scripts\tasks.ps1 data"
