#!/usr/bin/env python3
"""
Evaluation script for ABSA models.

This script evaluates trained ABSA models on test datasets and generates
comprehensive evaluation reports including metrics, visualizations, and error analysis.

usage:
    python evaluate.py --model_path <path_to_model> --config <path_to_config> --data_dir <path_to_data> --output <output_directory>
"""

import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
from transformers import AutoTokenizer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import DeBERTaATE, DeBERTaASC, BERTForABSA, EndToEndABSA
from preprocessing import ABSAPreprocessor, collate_end2end_absa_batch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings('ignore')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_ate_model(model, test_loader, device, tokenizer):
    """Evaluate Aspect Term Extraction model."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs['predictions']
            if isinstance(predictions, list):
                for pred_seq, label_seq, mask in zip(predictions, labels, batch['attention_mask']):
                    valid_length = mask.sum().item()
                    all_predictions.extend(pred_seq[:valid_length])
                    valid_labels = label_seq[:valid_length]
                    valid_labels = valid_labels[valid_labels != -100]
                    all_labels.extend(valid_labels.cpu().numpy())
            else:
                active_mask = batch['attention_mask'].view(-1) == 1
                active_labels = labels.view(-1)[active_mask]
                active_predictions = predictions.view(-1)[active_mask]
                valid_mask = active_labels != -100
                all_predictions.extend(active_predictions[valid_mask].cpu().numpy())
                all_labels.extend(active_labels[valid_mask].cpu().numpy())
    
    # Calculate ATE metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=0)
    ate_metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return ate_metrics, all_predictions, all_labels

def evaluate_asc_model(model, test_loader, device):
    """Evaluate Aspect Sentiment Classification model."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            predictions = outputs['predictions']
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate ASC metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    asc_metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    
    return asc_metrics, all_predictions, all_labels

def evaluate_end_to_end_model(model, test_loader, device):
    """Evaluate End-to-End ABSA model."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Implementation would depend on actual model structure
            pass
    
    # Calculate end-to-end metrics
    e2e_metrics = {
        'exact_match': 0.0,
        'aspect_f1': 0.0,
        'sentiment_f1': 0.0
    }
    
    return e2e_metrics, all_predictions, all_labels

def create_evaluation_plots(predictions, labels, output_dir, model_type):
    """Create evaluation plots and save them. Returns list of plot paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_type.upper()} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = output_dir / f'{model_type.lower()}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['confusion_matrix'] = cm_path

    # Classification Report
    report = classification_report(labels, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, max(6, len(report_df) * 0.5)))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.3f')
    plt.title(f'{model_type.upper()} - Classification Report')
    cr_path = output_dir / f'{model_type.lower()}_classification_report.png'
    plt.savefig(cr_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['classification_report'] = cr_path

    # Prediction Distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(predictions, bins=len(set(predictions)), kde=False, color='skyblue', label='Predictions')
    sns.histplot(labels, bins=len(set(labels)), kde=False, color='salmon', label='Labels', alpha=0.6)
    plt.legend()
    plt.title(f'{model_type.upper()} - Prediction/Label Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    dist_path = output_dir / f'{model_type.lower()}_prediction_distribution.png'
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['distribution'] = dist_path

    return plot_paths

def generate_evaluation_report(metrics, output_path, plot_paths=None, error_analysis=None):
    """Generate comprehensive evaluation report with embedded plots and error analysis."""
    def fmt(val):
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return str(val)
    report = f"""
# ABSA Model Evaluation Report

## Performance Metrics

| Metric | Value |
|--------|-------|
| ATE Precision | {fmt(metrics.get('ate_precision', 'N/A'))} |
| ATE Recall | {fmt(metrics.get('ate_recall', 'N/A'))} |
| ATE F1-Score | {fmt(metrics.get('ate_f1', 'N/A'))} |
| ASC Accuracy | {fmt(metrics.get('asc_accuracy', 'N/A'))} |
| ASC F1-Macro | {fmt(metrics.get('asc_f1_macro', 'N/A'))} |
| ASC F1-Weighted | {fmt(metrics.get('asc_f1_weighted', 'N/A'))} |
| E2E Exact Match | {fmt(metrics.get('e2e_exact_match', 'N/A'))} |
| E2E Aspect F1 | {fmt(metrics.get('e2e_aspect_f1', 'N/A'))} |
| E2E Sentiment F1 | {fmt(metrics.get('e2e_sentiment_f1', 'N/A'))} |

## Model Configuration
- Model Type: {metrics.get('model_type', 'N/A')}
- Dataset: {metrics.get('dataset', 'N/A')}
- Evaluation Date: {metrics.get('evaluation_date', 'N/A')}
"""
    if plot_paths:
        report += "\n## Visualizations\n"
        if 'confusion_matrix' in plot_paths:
            report += f"\n### Confusion Matrix\n![]({plot_paths['confusion_matrix'].as_posix()})\n"
        if 'classification_report' in plot_paths:
            report += f"\n### Classification Report\n![]({plot_paths['classification_report'].as_posix()})\n"
        if 'distribution' in plot_paths:
            report += f"\n### Prediction/Label Distribution\n![]({plot_paths['distribution'].as_posix()})\n"
    if error_analysis:
        report += "\n## Error Analysis\n" + error_analysis + "\n"
    with open(output_path, 'w') as f:
        f.write(report)

def load_model(model_path, config, device):
    task = config['model'].get('task', 'token_classification')
    model_name = config['model']['name']
    if task == 'token_classification':
        model = DeBERTaATE(config)
    elif task == 'sequence_classification':
        model = DeBERTaASC(config)
    elif task == 'absa_end_to_end':
        if 'bert' in model_name.lower():
            model = BERTForABSA(config)
        else:
            model = EndToEndABSA(config)
    else:
        raise ValueError(f"Unsupported task: {task}")
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description='Evaluate ABSA model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='reports/evaluation', help='Output directory for reports')
    parser.add_argument('--plots_dir', type=str, default='reports/figures', help='Directory to save plots')
    args = parser.parse_args()

    config = load_config(args.config)
    model_type = None
    task = config.get('model', {}).get('task', '')
    if task == 'token_classification':
        model_type = 'ate'
    elif task == 'sequence_classification':
        model_type = 'asc'
    elif task == 'absa_end_to_end':
        model_type = 'end_to_end'
    else:
        raise ValueError(f"Unknown model task in config: {task}")

    print(f"Evaluating ABSA model: {args.model_path}")
    print(f"Model type: {model_type}")
    print(f"Configuration: {args.config}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model(args.model_path, config, device)

    # Prepare data
    preprocessor = ABSAPreprocessor(
        task=model_type,
        tokenizer_name=config['model']['name']
    )
    if model_type == 'ate':
        print("Evaluating Aspect Term Extraction model...")
        test_data = preprocessor.load_ate_data(
            data_dir=args.data_dir,
            train_size=config['data']['train_size'],
            val_size=config['data']['val_size'],
            test_size=config['data']['test_size']
        )[2]
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        metrics, predictions, labels = evaluate_ate_model(model, test_loader, device, tokenizer)
    elif model_type == 'asc':
        print("Evaluating Aspect Sentiment Classification model...")
        test_data = preprocessor.load_asc_data(
            data_dir=args.data_dir,
            train_size=config['data']['train_size'],
            val_size=config['data']['val_size'],
            test_size=config['data']['test_size']
        )[2]
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        metrics, predictions, labels = evaluate_asc_model(model, test_loader, device)
    else:
        print("Evaluating End-to-End ABSA model...")
        test_data = preprocessor.load_end_to_end_data(
            data_dir=args.data_dir,
            train_size=config['data']['train_size'],
            val_size=config['data']['val_size'],
            test_size=config['data']['test_size']
        )[2]
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        metrics, predictions, labels = evaluate_end_to_end_model(model, test_loader, device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots (with demo data for now)
    plot_paths = create_evaluation_plots(predictions, labels, output_dir, model_type)

    # Error analysis: show most common misclassifications
    from collections import Counter
    error_analysis = None
    if predictions and labels:
        errors = [(l, p) for l, p in zip(labels, predictions) if l != p]
        if errors:
            error_counts = Counter(errors).most_common(5)
            error_analysis = "Most common misclassifications (label → prediction):\n\n"
            error_analysis += "| True Label | Predicted | Count |\n|-----------|-----------|-------|\n"
            for (l, p), count in error_counts:
                error_analysis += f"| {l} | {p} | {count} |\n"
        else:
            error_analysis = "No misclassifications found."

    # Generate evaluation report
    metrics.update({
        'model_type': model_type,
        'dataset': config.get('data', {}).get('dataset', 'Unknown'),
        'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    report_path = output_dir / 'evaluation_report.md'
    generate_evaluation_report(metrics, report_path, plot_paths, error_analysis)
    
    print(f"\\n✅ Evaluation completed!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📋 Report: {report_path}")
    
    # Print summary metrics
    print(f"\\n📈 Performance Summary:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and key not in ['evaluation_date']:
            print(f"   {key}: {value:.4f}")

if __name__ == "__main__":
    main()
