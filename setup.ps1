# Comment-ABSA - PowerShell Setup Script
# This script sets up the virtual environment and installs all dependencies

Write-Host ""
Write-Host "========================================"
Write-Host "  Comment ABSA Analysis Setup"
Write-Host "========================================"
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "Python version: $pythonVersion"
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Create virtual environment if it doesn't exist
if (!(Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Virtual environment created successfully!" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to upgrade pip" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Installing project dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Installing project in development mode..." -ForegroundColor Yellow
pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to install project in development mode" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Creating required directories..." -ForegroundColor Yellow

# Create directory structure
$directories = @(
    "data\raw\restaurants",
    "data\raw\laptops", 
    "data\raw\custom",
    "data\processed\ate",
    "data\processed\asc",
    "data\processed\end2end",
    "data\interim",
    "models\checkpoints\ate",
    "models\checkpoints\asc", 
    "models\checkpoints\end2end",
    "models\final",
    "logs\training\ate",
    "logs\training\asc",
    "logs\training\end2end",
    "reports\figures",
    "visualizations\plots",
    "visualizations\analysis"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "Downloading NLTK data..." -ForegroundColor Yellow
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('vader_lexicon', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); print('NLTK data downloaded successfully!')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to download NLTK data" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================"
Write-Host "  Setup completed successfully!" -ForegroundColor Green
Write-Host "========================================"
Write-Host ""
Write-Host "To activate the environment in the future, run:" -ForegroundColor Cyan
Write-Host "  .\activate.bat" -ForegroundColor White
Write-Host ""
Write-Host "To start training, run:" -ForegroundColor Cyan
Write-Host "  python scripts\train.py --task ate" -ForegroundColor White
Write-Host "  python scripts\train.py --task asc" -ForegroundColor White  
Write-Host "  python scripts\train.py --task end2end" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to continue"
