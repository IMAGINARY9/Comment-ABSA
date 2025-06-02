@echo off
REM Comment ABSA Analysis - Windows Batch Setup Script
REM This script sets up the virtual environment and installs all dependencies

echo.
echo ========================================
echo  Comment ABSA Analysis Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip
)

REM Upgrade pip, setuptools, and wheel
python -m pip install --upgrade pip setuptools wheel

echo.
echo Installing project dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Install dependencies
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo Warning: requirements.txt not found
)

echo.
echo Installing project in development mode...
pip install -e .
if errorlevel 1 (
    echo WARNING: Failed to install project in development mode
)

echo.
echo Creating required directories...
if not exist "data\raw\restaurants" mkdir data\raw\restaurants
if not exist "data\raw\laptops" mkdir data\raw\laptops
if not exist "data\raw\custom" mkdir data\raw\custom
if not exist "data\processed\ate" mkdir data\processed\ate
if not exist "data\processed\asc" mkdir data\processed\asc
if not exist "data\processed\end2end" mkdir data\processed\end2end
if not exist "data\interim" mkdir data\interim
if not exist "models\checkpoints\ate" mkdir models\checkpoints\ate
if not exist "models\checkpoints\asc" mkdir models\checkpoints\asc
if not exist "models\checkpoints\end2end" mkdir models\checkpoints\end2end
if not exist "models\final" mkdir models\final
if not exist "logs\training\ate" mkdir logs\training\ate
if not exist "logs\training\asc" mkdir logs\training\asc
if not exist "logs\training\end2end" mkdir logs\training\end2end
if not exist "reports\figures" mkdir reports\figures
if not exist "visualizations\plots" mkdir visualizations\plots
if not exist "visualizations\analysis" mkdir visualizations\analysis

REM Install Jupyter kernel for this environment
python -m ipykernel install --user --name=comment-absa-env --display-name="Python (Comment ABSA)"

REM Create necessary directories (expanded)
for %%d in (data models logs outputs reports notebooks visualizations cache) do (
    if not exist %%d (
        mkdir %%d
        echo Created directory: %%d
    )
)

REM Add .pth file for PYTHONPATH
if exist venv\Lib\site-packages (
    for /f %%i in ('cd') do echo %%i > venv\Lib\site-packages\comment_absa.pth
    echo Created .pth file for automatic Python path configuration
)

echo.
echo Downloading NLTK data...
python -c "import nltk; [nltk.download(x, quiet=True) for x in ['punkt','stopwords','vader_lexicon','wordnet','averaged_perceptron_tagger']]"
if errorlevel 1 (
    echo WARNING: Failed to download NLTK data
)

echo.
echo Downloading pre-trained transformer model (DeBERTa)...
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('microsoft/deberta-v3-base'); AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')"
if errorlevel 1 (
    echo WARNING: Failed to download pre-trained transformer model
)

echo.
echo ========================================
echo  Setup completed successfully!
echo ========================================
echo.
echo To activate the environment in the future, run:
echo   activate.bat
echo.
echo To start training, run:
echo   python scripts\train.py --task ate
echo   python scripts\train.py --task asc
echo   python scripts\train.py --task end2end
echo.
pause
