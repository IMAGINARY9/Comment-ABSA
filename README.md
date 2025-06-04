# Aspect-Based Sentiment Analysis (ABSA) for Comments

A modular, robust, and extensible pipeline for Aspect-Based Sentiment Analysis (ABSA) on comment data. Supports Aspect Term Extraction (ATE), Aspect Sentiment Classification (ASC), and joint end-to-end (multitask) models. Built for research and practical deployment, with strong configuration, error handling, and reporting.

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Supported Tasks & Model Types](#supported-tasks--model-types)
- [Data Handling & Preprocessing](#data-handling--preprocessing)
- [Setup](#setup)
- [Workflow](#workflow)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)
    - [End-to-End Model](#end-to-end-model)
    - [Pipeline (ATE+ASC)](#pipeline-ateasc)
- [Evaluation & Reporting](#evaluation--reporting)
- [Extensibility & Advanced Features](#extensibility--advanced-features)
- [References & Credits](#references--credits)
- [Contact](#contact)

---

## Features
- **End-to-End ABSA**: Joint aspect extraction and sentiment classification in a single model, or as a pipeline.
- **Modular Architecture**: Swap models, datasets, and configs easily. Supports BERT, DeBERTa, LoRA, and more.
- **Config-Driven Workflow**: All training, evaluation, and prediction are controlled by YAML config files.
- **Per-Domain Data Splits**: Robust handling of domain-specific datasets (e.g., laptops, restaurants, tweets).
- **Advanced Logging & Reporting**: Model-type-specific logs, detailed evaluation reports, and visualizations.
- **Error-Resistant**: Defensive coding for missing labels, batch alignment, and checkpoint loading.

---

## Project Structure
```
Comment-ABSA/
├── configs/           # YAML config files for all model types and domains
├── data/              # Raw, preprocessed, and split datasets (per domain)
├── src/               # Source code (models, training, utils, preprocessing)
├── scripts/           # Entry-point scripts for training, evaluation, prediction
├── models/            # Saved checkpoints and best models (per run/domain)
├── logs/              # Training logs (per model type and run)
├── reports/           # Evaluation reports and plots
├── outputs/           # Output configs and training histories
├── notebooks/         # Data cleaning, exploration, and reference notebooks
├── tests/             # Unit tests
└── visualizations/    # Analysis and plots
```

---

## Supported Tasks & Model Types

### Tasks
- **Aspect Term Extraction (ATE)**: Sequence labeling to extract aspect terms.
- **Aspect Sentiment Classification (ASC)**: Classify sentiment for each aspect.
- **End-to-End ABSA**: Jointly extract aspects and classify their sentiment (multitask learning).

### Model Types
- **BERT/DeBERTa-based**: For ATE, ASC, and end-to-end multitask.
- **LoRA**: Parameter-efficient fine-tuning (optional).
- **Traditional**: BiLSTM-CRF, CRF (for comparison).

---

## Data Handling & Preprocessing
- **Per-domain splits**: Data in `data/splits/{domain}/[ate|asc|end2end]/`.
- **Preprocessing**: Modular, robust, and extensible (see `src/preprocessing.py`).
- **Collate Functions**: Custom batching for multitask and pipeline models.
- **Data cleaning and splitting**: Use provided notebooks/scripts for preparing and splitting data.

---

## Setup

### Automated Setup
- **Windows (PowerShell):**
  ```powershell
  ./setup.ps1
  ./activate.bat
  ```
- **Windows (CMD):**
  ```bat
  setup.bat
  activate.bat
  ```
- **Unix/Linux/macOS:**
  ```bash
  chmod +x setup.sh
  ./setup.sh
  source venv/bin/activate
  ```

### Manual Setup
1. Create venv: `python -m venv venv`
2. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix/Linux)
3. Install: `pip install -r requirements.txt`

---

## Workflow

### Data Preparation
- Place raw data in `data/raw/` or use provided scripts/notebooks for cleaning.
- Preprocess and split: *(see notebooks or custom scripts for data preparation)*

### Training
- **ATE**:
  ```powershell
  python scripts/train.py --config configs/deberta_ate.yaml --domain tweets
  ```
- **ASC**:
  ```powershell
  python scripts/train.py --config configs/deberta_asc.yaml --domain tweets
  ```
- **End-to-End**:
  ```powershell
  python scripts/train.py --config configs/bert_end_to_end.yaml --domain restaurants
  ```

### Evaluation
- Evaluate a trained model:
  ```powershell
  python scripts/evaluate.py --config configs/bert_end_to_end.yaml --domain restaurants --model_path models/end2end_restaurants_YYYYMMDD_HHMMSS/best_model.pt
  ```

### Prediction
#### End-to-End Model
- Use a single multitask model to extract aspects and classify their sentiment in one step.
  ```powershell
  python scripts/predict.py --config configs/bert_end_to_end.yaml --model_path models/end2end_restaurants_YYYYMMDD_HHMMSS/best_model.pt --text "The food was great but the service was slow."
  ```
- For batch prediction:
  ```powershell
  python scripts/predict.py --config configs/bert_end_to_end.yaml --model_path models/end2end_restaurants_YYYYMMDD_HHMMSS/best_model.pt --input_file data/some_texts.csv --output_file outputs/predictions.json
  ```

#### Pipeline (ATE+ASC)
- Use two models: first an ATE model to extract aspect terms, then an ASC model to classify sentiment for each aspect.
  ```powershell
  python scripts/predict.py --model_path models/asc_tweets_YYYYMMDD_HHMMSS/best_model.pt --config configs/deberta_asc.yaml --ate_model_path models/ate_tweets_YYYYMMDD_HHMMSS/best_model.pt --ate_config configs/deberta_ate.yaml --text "why the hell do i follow donald trump on twitter?"
  ```
- For batch prediction:
  ```powershell
  python scripts/predict.py --model_path models/asc_tweets_YYYYMMDD_HHMMSS/best_model.pt --config configs/deberta_asc.yaml --ate_model_path models/ate_tweets_YYYYMMDD_HHMMSS/best_model.pt --ate_config configs/deberta_ate.yaml --input_file data/some_texts.csv --output_file outputs/predictions.json
  ```

---

## Evaluation & Reporting
- **ATE**: Precision, Recall, F1 (aspect extraction)
- **ASC**: Accuracy, F1 (sentiment classification)
- **End-to-End**: Exact match, aspect-sentiment F1
- **Reports**: See `reports/evaluation/` for classification reports, confusion matrices, and prediction distributions.

---

## Extensibility & Advanced Features
- **Add new models**: Implement in `src/models.py` and update config.
- **Custom data**: Add to `data/raw/` and update splits.
- **Advanced evaluation**: Extend `src/evaluation.py` or add new scripts.
- **Visualization**: Use `visualizations/` for custom plots.

---

## References & Credits
- SemEval ABSA datasets (2014-2016)
- BERT, DeBERTa, LoRA papers
- See `references/` for notebooks and key papers

---

## Contact
For questions, issues, or contributions, please open an issue or contact the maintainer.
