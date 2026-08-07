<#
.SYNOPSIS
PowerShell equivalent of the Makefile for Windows users.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("data", "forecast", "plot", "check", "format", "integration")]
    [string]$Task
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Environnement absent. Lancez d'abord .\scripts\bootstrap_windows.ps1"
}

function Invoke-ProjectPython {
    param([string[]]$Arguments)

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Push-Location $ProjectRoot
try {
    switch ($Task) {
        "data" {
            Invoke-ProjectPython @("fetch_open_data.py")
        }
        "forecast" {
            Invoke-ProjectPython @("train_transformer.py")
        }
        "plot" {
            Invoke-ProjectPython @("plot_scenarios.py")
        }
        "check" {
            Invoke-ProjectPython @("-m", "ruff", "check", ".")
            Invoke-ProjectPython @("-m", "ruff", "format", "--check", ".")
            Invoke-ProjectPython @(
                "-m", "mypy", "src", "fetch_open_data.py",
                "plot_scenarios.py", "train_transformer.py"
            )
            Invoke-ProjectPython @(
                "-m", "pytest", "-m", "not integration",
                "--cov=aipoweryou_forecast", "--cov-fail-under=80"
            )
        }
        "format" {
            Invoke-ProjectPython @("-m", "ruff", "check", ".", "--fix")
            Invoke-ProjectPython @("-m", "ruff", "format", ".")
        }
        "integration" {
            $PreviousValue = $env:RUN_INTEGRATION
            $env:RUN_INTEGRATION = "1"
            try {
                Invoke-ProjectPython @("-m", "pytest", "-m", "integration")
            }
            finally {
                $env:RUN_INTEGRATION = $PreviousValue
            }
        }
    }
}
finally {
    Pop-Location
}
