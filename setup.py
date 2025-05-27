"""
Setup script for ABSA project.

This script installs dependencies, downloads models, and prepares the environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, check=True, shell=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_dependencies():
    """Install Python dependencies."""
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies"
    )

def download_models():
    """Download pre-trained models for ABSA."""
    commands = [
        "python -c \"from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('microsoft/deberta-v3-base'); AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')\"",
        "python -c \"from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('bert-base-uncased'); AutoTokenizer.from_pretrained('bert-base-uncased')\"",
    ]
    
    success = True
    for cmd in commands:
        success &= run_command(cmd, "Downloading pre-trained models")
    
    return success

def setup_directories():
    """Create necessary directories."""
    directories = [
        "data/semeval_datasets", "data/custom_datasets",
        "models/ate", "models/asc", "models/end_to_end",
        "logs/ate", "logs/asc", "logs/end_to_end",
        "outputs/ate", "outputs/asc", "outputs/end_to_end",
        "reports", "notebooks", "visualizations", "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def download_nltk_data():
    """Download required NLTK data."""
    commands = [
        "python -c \"import nltk; nltk.download('punkt', quiet=True)\"",
        "python -c \"import nltk; nltk.download('stopwords', quiet=True)\"",
        "python -c \"import nltk; nltk.download('wordnet', quiet=True)\"",
        "python -c \"import nltk; nltk.download('averaged_perceptron_tagger', quiet=True)\"",
    ]
    
    success = True
    for cmd in commands:
        success &= run_command(cmd, "Downloading NLTK data")
    
    return success

def install_spacy_model():
    """Install spaCy English model."""
    return run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Installing spaCy English model"
    )

def check_datasets():
    """Check for ABSA datasets."""
    print("\nChecking for ABSA datasets...")
    data_dir = Path("data/semeval_datasets")
    
    if not any(data_dir.glob("*.xml")):
        print("⚠ No SemEval XML datasets found.")
        print("Please download SemEval ABSA datasets:")
        print("- SemEval 2014 Task 4: http://alt.qcri.org/semeval2014/task4/")
        print("- SemEval 2015 Task 12: http://alt.qcri.org/semeval2015/task12/")
        print("- SemEval 2016 Task 5: http://alt.qcri.org/semeval2016/task5/")
        print("Place XML files in: ./data/semeval_datasets/")
        return False
    else:
        xml_files = list(data_dir.glob("*.xml"))
        print(f"✓ Found {len(xml_files)} XML dataset files")
        return True

def main():
    print("Setting up Aspect-Based Sentiment Analysis (ABSA) Project")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Using Python {sys.version}")
    
    # Setup steps
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
            print(f"⚠ Warning: {description} failed, but continuing...")
    
    print(f"\nSetup completed: {success_count}/{len(steps)} steps successful")
    
    if success_count >= len(steps) - 1:  # Allow datasets to be optional for initial setup
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add SemEval datasets to data/semeval_datasets/ (if not already done)")
        print("2. Prepare ABSA data: python scripts/prepare_absa_data.py")
        print("3. Train ATE model: python scripts/train_absa.py --task ate --config configs/deberta_ate.yaml")
        print("4. Train ASC model: python scripts/train_absa.py --task asc --config configs/deberta_asc.yaml")
        print("5. Evaluate pipeline: python scripts/evaluate_absa.py")
    else:
        print("\n⚠ Setup completed with some warnings. Check the output above.")

if __name__ == "__main__":
    main()
