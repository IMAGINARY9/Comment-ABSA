# Aspect-Based Sentiment Analysis (ABSA) for Comments

This project implements aspect-based sentiment analysis for comments, extracting aspects and their associated sentiments.

## Overview

- **Task**: Extract aspect terms and classify sentiment for each aspect in comments
- **Approach**: Pipeline (ATE + ASC) and end-to-end models using BERT/DeBERTa
- **Challenges**: Multiple aspects per comment, implicit aspects, aspect-sentiment pairs

## Project Structure

```
comment-absa/
├── data/                    # ABSA datasets (SemEval, domain-specific)
├── src/                     # Source code
├── models/                  # Trained models and checkpoints
├── notebooks/               # Jupyter notebooks for experiments
├── configs/                 # Configuration files
├── scripts/                 # Training and evaluation scripts
├── references/              # Reference notebooks and papers
├── logs/                    # Training logs
├── reports/                 # Analysis reports and results
├── visualizations/          # ABSA visualizations
└── tests/                   # Unit tests
```

## Key Features

1. **Aspect Term Extraction (ATE)**: Token classification for aspect identification
2. **Aspect Sentiment Classification (ASC)**: Sentiment classification per aspect
3. **End-to-End Models**: Joint aspect and sentiment prediction
4. **Advanced Models**: DeBERTa with LoRA, BERT-based pipelines
5. **Evaluation**: Comprehensive ABSA metrics

## Setup

### Automated Setup (Recommended)

Use the provided setup scripts to create a virtual environment and install all dependencies:

**Windows:**
```bash
# Run the setup script
.\setup.bat

# Activate the environment
.\activate.bat
```

**PowerShell:**
```powershell
# Run the setup script
.\setup.ps1

# Activate the environment
.\activate.bat
```

**Unix/Linux/macOS:**
```bash
# Make script executable and run
chmod +x setup.sh
./setup.sh

# Activate the environment
source venv/bin/activate
```

### Manual Setup

If you prefer manual setup:
1. Create virtual environment: `python -m venv venv`
2. Activate environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix/Linux)
3. Install dependencies: `pip install -r requirements.txt`

## Quick Start

1. **Setup environment**: Use `setup.bat` (Windows) or `setup.sh` (Unix/Linux)
2. **Activate environment**: Run `activate.bat` or `source venv/bin/activate`
3. **Prepare data**: `python scripts/prepare_absa_data.py`
4. **Train ATE model**: `python scripts/train_ate.py --config configs/deberta_ate.yaml`
5. **Train ASC model**: `python scripts/train_asc.py --config configs/deberta_asc.yaml`
6. **Evaluate pipeline**: `python scripts/evaluate_absa.py --ate-model ./models/ate_best.pt --asc-model ./models/asc_best.pt`

## Datasets

- SemEval 2014 Task 4 (Laptops, Restaurants)
- SemEval 2015 Task 12 (Hotels, Restaurants)
- SemEval 2016 Task 5 (Laptops, Restaurants)
- Custom domain-specific ABSA datasets

## Models

- **DeBERTa with LoRA**: Parameter-efficient fine-tuning
- **BERT Pipeline**: Separate models for ATE and ASC
- **End-to-End Models**: Joint aspect-sentiment prediction
- **Traditional Models**: CRF, BiLSTM-CRF for comparison

## Evaluation Metrics

- **ATE**: Precision, Recall, F1 for aspect extraction
- **ASC**: Accuracy, F1 for sentiment classification
- **End-to-End**: Exact match, aspect-sentiment F1
