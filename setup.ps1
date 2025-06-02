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
Write-Host "Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to upgrade pip, setuptools, or wheel" -ForegroundColor Yellow
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
Write-Host "Installing Jupyter kernel for this environment..." -ForegroundColor Yellow
python -m ipykernel install --user --name=comment-absa-env --display-name="Python (Comment ABSA)"
Write-Host "Jupyter kernel installed" -ForegroundColor Green

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
    "visualizations\analysis",
    "data",
    "models",
    "logs",
    "outputs",
    "reports",
    "notebooks",
    "visualizations",
    "cache"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Cyan
    }
}

# Add .pth file for PYTHONPATH
$sitePackagesDir = Join-Path -Path ".\venv\Lib\site-packages" -ChildPath "comment_absa.pth"
$projectPath = (Get-Item -Path ".").FullName
$projectPath | Out-File -FilePath $sitePackagesDir -Encoding ascii
Write-Host "Created .pth file for automatic Python path configuration" -ForegroundColor Green

Write-Host ""
Write-Host "Downloading NLTK data..." -ForegroundColor Yellow
$nltkDownloads = @("punkt", "stopwords", "vader_lexicon", "wordnet", "averaged_perceptron_tagger")
foreach ($corpus in $nltkDownloads) {
    python -c "import nltk; nltk.download('$corpus', quiet=True)"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Downloaded NLTK corpus: $corpus" -ForegroundColor Green
    } else {
        Write-Host "Warning: Failed to download NLTK corpus: $corpus" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Downloading pre-trained DeBERTa model..." -ForegroundColor Yellow
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('microsoft/deberta-v3-base'); AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Downloaded DeBERTa model" -ForegroundColor Green
} else {
    Write-Host "Warning: Failed to download DeBERTa model" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================"
Write-Host "  Setup completed successfully!" -ForegroundColor Green
Write-Host "========================================"
Write-Host "To activate the environment in the future, run: .\activate.bat" -ForegroundColor Cyan
Write-Host "For Jupyter, run: jupyter notebook" -ForegroundColor Cyan
Write-Host "If you encounter import errors, ensure the venv is activated and PYTHONPATH is set." -ForegroundColor Yellow
