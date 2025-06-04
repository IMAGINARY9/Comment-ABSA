# Aspect-Based Sentiment Analysis (ABSA) for Comments

A modular, robust, and extensible pipeline for Aspect-Based Sentiment Analysis (ABSA) on comment data, supporting Aspect Term Extraction (ATE), Aspect Sentiment Classification (ASC), and joint end-to-end (multitask) models. Built for research and practical deployment, with strong configuration, error handling, and reporting.

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
├── activate.bat
├── README.md
├── requirements.txt
├── setup.bat / setup.ps1 / setup.py
├── configs/           # YAML config files for all model types and domains
│   ├── bert_end_to_end.yaml
│   ├── deberta_asc.yaml
│   ├── deberta_ate.yaml
│   └── ...
├── data/              # Raw, preprocessed, and split datasets (per domain)
│   ├── raw/
│   ├── preprocessed/
│   ├── processed/
│   ├── splits/
│   └── ...
├── src/               # Source code (models, training, utils, preprocessing)
│   ├── absa_trainer.py
│   ├── data_utils.py
│   ├── evaluation.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── training.py
│   └── ...
├── scripts/           # Entry-point scripts for training, evaluation, prediction
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/            # Saved checkpoints and best models (per run/domain)
│   ├── asc_tweets_*/
│   ├── ate_tweets_*/
│   ├── end2end_restaurants_*/
│   ├── checkpoints/
│   └── final/
├── logs/              # Training logs (per model type and run)
│   ├── asc/
│   ├── ate/
│   ├── end2end/
│   └── ...
├── reports/           # Evaluation reports and plots
│   └── evaluation/
├── outputs/           # Output configs and training histories
├── notebooks/         # Data cleaning, exploration, and reference notebooks
├── tests/             # Unit tests
└── visualizations/    # Analysis and plots
```

---

## Architecture & Workflow

### Supported Tasks
- **Aspect Term Extraction (ATE)**: Sequence labeling to extract aspect terms.
- **Aspect Sentiment Classification (ASC)**: Classify sentiment for each aspect.
- **End-to-End ABSA**: Jointly extract aspects and classify their sentiment (multitask learning).

### Model Types
- **BERT/DeBERTa-based**: For ATE, ASC, and end-to-end multitask.
- **LoRA**: Parameter-efficient fine-tuning (optional).
- **Traditional**: BiLSTM-CRF, CRF (for comparison).

### Data Handling
- **Per-domain splits**: Data in `data/splits/{domain}/[ate|asc|end2end]/`.
- **Preprocessing**: Modular, robust, and extensible (see `src/preprocessing.py`).
- **Collate Functions**: Custom batching for multitask and pipeline models.

### Training & Evaluation
- **Config-driven**: All scripts accept a `--config` argument (YAML).
- **Domain-specific**: Use `--domain` to specify the dataset domain (e.g., `laptops`, `restaurants`, `tweets`).
- **Output directory**: Use `--output_dir` to specify where to save model checkpoints.
- **Logging**: Logs saved in `logs/{ate,asc,end2end}/`.
- **Checkpoints**: Saved in `models/{ate,asc,end2end}/`.
- **Evaluation**: Reports and plots in `reports/evaluation/`.

### Prediction
- **Script**: `scripts/predict.py` for all model types.
- **Post-processing**: Robust mapping from model outputs to aspect/sentiment labels.

---

## Setup

### Automated Setup

**Windows (PowerShell):**
```powershell
./setup.ps1
./activate.bat
```
**Windows (CMD):**
```bat
setup.bat
activate.bat
```
**Unix/Linux/macOS:**
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

## Quick Start

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
  python scripts/evaluate.py --config configs/bert_end_to_end.yaml --domain restaurants --model_dir models/end2end_restaurants_YYYYMMDD_HHMMSS
  ```

### Prediction
- Predict on new data:
  ```powershell
  python scripts/predict.py --config configs/bert_end_to_end.yaml --domain restaurants --model_dir models/end2end_restaurants_YYYYMMDD_HHMMSS --input <input_file> --output <output_file>
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
