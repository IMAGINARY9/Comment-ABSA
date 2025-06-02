"""
Setup script for ABSA project.

This script installs dependencies, downloads models, and prepares the environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description, use_venv=True):
    print(f"\n{description}...")
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    if use_venv and os.path.exists(venv_python):
        if command.startswith("python "):
            command = command.replace("python", f'"{venv_python}"', 1)
        elif command.startswith("pip "):
            command = command.replace("pip", f'"{venv_python}" -m pip', 1)
    try:
        result = subprocess.run(command, check=True, shell=True, capture_output=True, text=True)
        print(f"\u2713 {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\u2717 {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def ensure_venv():
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        result = subprocess.run("python -m venv venv", shell=True)
        if result.returncode != 0:
            print("\u2717 Failed to create virtual environment")
            sys.exit(1)
        print("\u2713 Virtual environment created.")
    else:
        print("\u2713 Virtual environment already exists.")

def install_dependencies():
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    run_command(f'"{venv_python}" -m pip install --upgrade pip setuptools wheel', "Upgrading pip and build tools", use_venv=False)
    return run_command(f'"{venv_python}" -m pip install -r requirements.txt', "Installing dependencies", use_venv=False)

def download_models():
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    commands = [
        f'"{venv_python}" -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained(\'microsoft/deberta-v3-base\'); AutoTokenizer.from_pretrained(\'microsoft/deberta-v3-base\')"',
        f'"{venv_python}" -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained(\'bert-base-uncased\'); AutoTokenizer.from_pretrained(\'bert-base-uncased\')"',
    ]
    success = True
    for cmd in commands:
        success &= run_command(cmd, "Downloading pre-trained models", use_venv=False)
    return success

def setup_directories():
    directories = [
        "data/semeval_datasets", "data/custom_datasets",
        "models/ate", "models/asc", "models/end_to_end",
        "logs/ate", "logs/asc", "logs/end_to_end",
        "outputs/ate", "outputs/asc", "outputs/end_to_end",
        "reports", "notebooks", "visualizations", "cache"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"\u2713 Created directory: {directory}")
    return True

def download_nltk_data():
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    commands = [
        f'"{venv_python}" -c "import nltk; nltk.download(\'punkt\', quiet=True)"',
        f'"{venv_python}" -c "import nltk; nltk.download(\'stopwords\', quiet=True)"',
        f'"{venv_python}" -c "import nltk; nltk.download(\'wordnet\', quiet=True)"',
        f'"{venv_python}" -c "import nltk; nltk.download(\'averaged_perceptron_tagger\', quiet=True)"',
    ]
    success = True
    for cmd in commands:
        success &= run_command(cmd, "Downloading NLTK data", use_venv=False)
    return success

def install_spacy_model():
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    return run_command(f'"{venv_python}" -m spacy download en_core_web_sm', "Installing spaCy English model", use_venv=False)

def check_datasets():
    print("\nChecking for ABSA datasets...")
    data_dir = Path("data/semeval_datasets")
    if not any(data_dir.glob("*.xml")):
        print("\u26a0 No SemEval XML datasets found.")
        print("Please download SemEval ABSA datasets:")
        print("- SemEval 2014 Task 4: http://alt.qcri.org/semeval2014/task4/")
        print("- SemEval 2015 Task 12: http://alt.qcri.org/semeval2015/task12/")
        print("- SemEval 2016 Task 5: http://alt.qcri.org/semeval2016/task5/")
        print("Place XML files in: ./data/semeval_datasets/")
        return False
    else:
        xml_files = list(data_dir.glob("*.xml"))
        print(f"\u2713 Found {len(xml_files)} XML dataset files")
        return True

def main():
    print("Setting up Aspect-Based Sentiment Analysis (ABSA) Project")
    print("=" * 60)
    if sys.version_info < (3, 8):
        print("\u2717 Python 3.8 or higher is required")
        sys.exit(1)
    print(f"\u2713 Using Python {sys.version}")
    ensure_venv()
    steps = [
        ("Setting up directories", setup_directories),
        ("Installing dependencies", install_dependencies),
        ("Downloading NLTK data", download_nltk_data),
        ("Installing spaCy model", install_spacy_model),
        ("Downloading pre-trained models", download_models),
        ("Checking datasets", check_datasets),
    ]
    success_count = 0
    for description, func in steps:
        if func():
            success_count += 1
        else:
            print(f"\u26a0 Warning: {description} failed, but continuing...")
    print(f"\nSetup completed: {success_count}/{len(steps)} steps successful")
    if success_count >= len(steps) - 1:
        print("\n\U0001F389 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add SemEval datasets to data/semeval_datasets/ (if not already done)")
        print("2. Prepare ABSA data: python scripts/prepare_absa_data.py")
        print("3. Train ATE model: python scripts/train_absa.py --task ate --config configs/deberta_ate.yaml")
        print("4. Train ASC model: python scripts/train_absa.py --task asc --config configs/deberta_asc.yaml")
        print("5. Evaluate pipeline: python scripts/evaluate_absa.py")
    else:
        print("\n\u26a0 Setup completed with some warnings. Check the output above.")

if __name__ == "__main__":
    main()
