@echo off
REM Comment ABSA Analysis - Environment Activation Script
REM This script activates the virtual environment for development

echo.
echo ========================================
echo  Comment ABSA Analysis
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to create the environment.
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Environment activated successfully!
echo.
echo Available commands:
echo   python scripts\train.py --task ate      - Train Aspect Term Extraction
echo   python scripts\train.py --task asc      - Train Aspect Sentiment Classification
echo   python scripts\train.py --task end2end  - Train End-to-End ABSA
echo   python -m pytest tests\                 - Run tests
echo   python scripts\evaluate.py              - Evaluate models
echo   python scripts\predict.py               - Make predictions
echo.
echo Available tasks:
echo   ate     - Aspect Term Extraction
echo   asc     - Aspect Sentiment Classification
echo   end2end - End-to-End ABSA
echo.
echo To deactivate, type: deactivate
echo.

REM Keep the command prompt open
cmd /k
