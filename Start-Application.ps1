param(
    [ValidateSet("paper", "backtest", "live")]
    [string]$Mode = "paper",

    [double]$Balance = 100000,

    [int]$DashboardPort = 5000,

    [string]$Currency = "USD",

    [switch]$Visualize,

    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Clear-MissingCertificatePath {
    param([string]$Name)

    $Value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (-not $Value) {
        $Value = [Environment]::GetEnvironmentVariable($Name, "User")
    }
    if (-not $Value) {
        $Value = [Environment]::GetEnvironmentVariable($Name, "Machine")
    }

    if ($Value -and -not (Test-Path -LiteralPath $Value)) {
        Write-Host "Ignoring missing certificate path from $Name=$Value" -ForegroundColor Yellow
        Remove-Item -Path "Env:\$Name" -ErrorAction SilentlyContinue
        [Environment]::SetEnvironmentVariable($Name, $null, "Process")
    }
}

$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $AppRoot

Write-Step "Starting Automated Trading System setup"
Write-Host "Application folder: $AppRoot"

Clear-MissingCertificatePath "PIP_CERT"
Clear-MissingCertificatePath "REQUESTS_CA_BUNDLE"
Clear-MissingCertificatePath "SSL_CERT_FILE"
Clear-MissingCertificatePath "CURL_CA_BUNDLE"

if (-not (Test-Path -LiteralPath "requirements.txt")) {
    throw "requirements.txt was not found. Please run this script from the application folder."
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found on PATH. Install Python 3.10+ and rerun this script."
}

if (-not (Test-Path -LiteralPath "venv")) {
    Write-Step "Creating Python virtual environment"
    python -m venv venv
}

$PythonExe = Join-Path $AppRoot "venv\Scripts\python.exe"
$PipExe = Join-Path $AppRoot "venv\Scripts\pip.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Virtual environment Python was not created correctly: $PythonExe"
}

if (-not $SkipInstall) {
    Write-Step "Installing Python dependencies"
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r requirements.txt
} else {
    Write-Step "Skipping dependency installation"
}

Write-Step "Checking required Python packages"
$DependencyCheck = @'
import importlib.util
import sys

required = ["pandas", "numpy", "flask", "loguru"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(",".join(missing))
    sys.exit(1)
print("OK")
'@

$DependencyCheckFile = Join-Path $env:TEMP "automated_trading_dependency_check.py"
Set-Content -LiteralPath $DependencyCheckFile -Value $DependencyCheck -Encoding UTF8

$CheckResult = & $PythonExe $DependencyCheckFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "Missing required packages: $CheckResult" -ForegroundColor Yellow
    Write-Step "Installing missing dependencies from requirements.txt"
    & $PythonExe -m pip install -r requirements.txt

    $CheckResult = & $PythonExe $DependencyCheckFile
    if ($LASTEXITCODE -ne 0) {
        throw "Dependencies are still missing after install attempt: $CheckResult"
    }
}
Remove-Item -LiteralPath $DependencyCheckFile -ErrorAction SilentlyContinue
Write-Host "Dependency check passed." -ForegroundColor Green

if (-not (Test-Path -LiteralPath ".env") -and (Test-Path -LiteralPath ".env.example")) {
    Write-Step "Creating .env from .env.example"
    Copy-Item -Force ".env.example" ".env"
}

Write-Step "Launching application"

$ArgsList = @(
    "main.py",
    "--mode", $Mode,
    "--balance", $Balance,
    "--currency", $Currency,
    "--dashboard-port", $DashboardPort
)

if ($Mode -eq "backtest" -and $Visualize) {
    $ArgsList += "--visualize"
}

Write-Host "Command: $PythonExe $($ArgsList -join ' ')" -ForegroundColor Yellow

if ($Mode -eq "paper") {
    Write-Host ""
    Write-Host "Dashboard will be available at: http://localhost:$DashboardPort" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop paper trading." -ForegroundColor Green
}

& $PythonExe @ArgsList
