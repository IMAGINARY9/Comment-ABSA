"""
ABSA Evaluation Module

This module provides comprehensive evaluation utilities for Aspect-Based Sentiment Analysis:
- Metrics for ATE (Aspect Term Extraction)
- Metrics for ASC (Aspect Sentiment Classification)
- End-to-end ABSA evaluation
- Visualization and analysis tools
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, f1_score
)
from collections import Counter, defaultdict
import re
from tqdm import tqdm

from .models import DeBERTaATE, DeBERTaASC, ABSAPredictor


class ABSAEvaluator:
    """
    Comprehensive evaluator for ABSA models.
    
    Provides evaluation metrics and analysis for aspect extraction
    and sentiment classification tasks.
    """
    
    def __init__(self, task: str = "ate", label_names: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            task: Task type ('ate', 'asc', 'end_to_end')
            label_names: Names for labels
        """
        self.task = task
        self.logger = logging.getLogger(__name__)
        
        # Set label names based on task
        if label_names:
            self.label_names = label_names
        elif task == "ate":
            self.label_names = ['B-ASP', 'I-ASP', 'B-OP', 'I-OP', 'O']
        elif task == "asc":
            self.label_names = ['negative', 'neutral', 'positive']
        else:
            self.label_names = []
        
        # Aspect extraction patterns
        self.aspect_patterns = [
            r'\b(food|meal|dish|cuisine|menu)\b',
            r'\b(service|staff|waiter|waitress|server)\b',
            r'\b(price|cost|expensive|cheap|affordable)\b',
            r'\b(ambiance|atmosphere|environment|decor)\b',
            r'\b(location|place|area|neighborhood)\b'
        ]
    
    def evaluate_ate(self, model: DeBERTaATE, dataloader, tokenizer) -> Dict:
        """
        Evaluate Aspect Term Extraction model.
        
        Args:
            model: Trained ATE model
            dataloader: Test data loader
            tokenizer: Tokenizer used for the model
            
        Returns:
            Evaluation metrics and analysis
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        extracted_aspects = []
        true_aspects = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating ATE"):
                input_ids = batch['input_ids'].to(model.device if hasattr(model, 'device') else 'cpu')
                attention_mask = batch['attention_mask'].to(model.device if hasattr(model, 'device') else 'cpu')
                labels = batch['labels']
                
                # Get model outputs
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = outputs['predictions']
                
                # Extract aspects from predictions
                batch_aspects = model.extract_aspects(input_ids, attention_mask, tokenizer)
                extracted_aspects.extend(batch_aspects)
                
                # Extract true aspects from labels (this would need the original texts and aspects)
                # For now, we'll collect token-level predictions and labels
                if isinstance(predictions, list):
                    # CRF predictions
                    for pred_seq, label_seq, mask in zip(predictions, labels, attention_mask):
                        valid_length = mask.sum().item()
                        all_predictions.extend(pred_seq[:valid_length])
                        valid_labels = label_seq[:valid_length]
                        valid_labels = valid_labels[valid_labels != -100]
                        all_labels.extend(valid_labels.numpy())
                else:
                    # Standard predictions
                    active_mask = attention_mask.view(-1) == 1
                    active_labels = labels.view(-1)[active_mask]
                    active_predictions = predictions.view(-1)[active_mask]
                    
                    # Filter out ignore_index
                    valid_mask = active_labels != -100
                    all_predictions.extend(active_predictions[valid_mask].cpu().numpy())
                    all_labels.extend(active_labels[valid_mask].cpu().numpy())
        
        # Calculate token-level metrics
        token_accuracy = accuracy_score(all_labels, all_predictions)
        token_precision, token_recall, token_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=self.label_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Aspect-level evaluation (simplified)
        aspect_metrics = self._evaluate_aspect_extraction(extracted_aspects, true_aspects)
        
        results = {
            'token_level': {
                'accuracy': token_accuracy,
                'precision': token_precision,
                'recall': token_recall,
                'f1': token_f1,
                'classification_report': class_report,
                'confusion_matrix': cm.tolist()
            },
            'aspect_level': aspect_metrics,
            'extracted_aspects': extracted_aspects
        }
        
        return results
    
    def evaluate_asc(self, model: DeBERTaASC, dataloader) -> Dict:
        """
        Evaluate Aspect Sentiment Classification model.
        
        Args:
            model: Trained ASC model
            dataloader: Test data loader
            
        Returns:
            Evaluation metrics and analysis
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        prediction_details = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating ASC"):
                input_ids = batch['input_ids'].to(model.device if hasattr(model, 'device') else 'cpu')
                attention_mask = batch['attention_mask'].to(model.device if hasattr(model, 'device') else 'cpu')
                labels = batch['labels']
                
                # Get model outputs
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs['logits']
                predictions = outputs['predictions']
                
                # Calculate probabilities
                probabilities = torch.softmax(logits, dim=-1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Store prediction details for analysis
                for i, (pred, true, prob) in enumerate(zip(predictions, labels, probabilities)):
                    prediction_details.append({
                        'predicted': pred.item(),
                        'true': true.item(),
                        'confidence': prob.max().item(),
                        'probabilities': prob.cpu().numpy()
                    })
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Per-class metrics
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=self.label_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Confidence analysis
        confidence_analysis = self._analyze_confidence(all_labels, all_predictions, all_probabilities)
        
        # Error analysis
        error_analysis = self._analyze_classification_errors(all_labels, all_predictions, prediction_details)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'confidence_analysis': confidence_analysis,
            'error_analysis': error_analysis,
            'prediction_details': prediction_details
        }
        
        return results
    
    def evaluate_end_to_end(self, predictor: ABSAPredictor, test_texts: List[str], 
                           true_aspects: List[List[str]], true_sentiments: List[List[str]]) -> Dict:
        """
        Evaluate end-to-end ABSA performance.
        
        Args:
            predictor: ABSA predictor combining ATE and ASC
            test_texts: Test texts
            true_aspects: True aspect terms for each text
            true_sentiments: True sentiments for each aspect
            
        Returns:
            End-to-end evaluation metrics
        """
        predictions = predictor.predict(test_texts)
        
        # Extract predicted aspects and sentiments
        pred_aspects = []
        pred_sentiments = []
        
        for pred in predictions:
            text_aspects = [asp['aspect'] for asp in pred['aspects']]
            text_sentiments = [asp['sentiment'] for asp in pred['aspects']]
            pred_aspects.append(text_aspects)
            pred_sentiments.append(text_sentiments)
        
        # Calculate aspect extraction metrics
        aspect_metrics = self._evaluate_aspect_extraction(pred_aspects, true_aspects)
        
        # Calculate sentiment classification metrics (for correctly extracted aspects)
        sentiment_metrics = self._evaluate_aspect_sentiment_pairs(
            pred_aspects, pred_sentiments, true_aspects, true_sentiments
        )
        
        # Calculate end-to-end metrics (exact match)
        exact_match_scores = []
        for pred_asp, pred_sent, true_asp, true_sent in zip(
            pred_aspects, pred_sentiments, true_aspects, true_sentiments
        ):
            # Create aspect-sentiment pairs
            pred_pairs = set(zip(pred_asp, pred_sent))
            true_pairs = set(zip(true_asp, true_sent))
            
            if len(true_pairs) == 0:
                exact_match = 1.0 if len(pred_pairs) == 0 else 0.0
            else:
                exact_match = 1.0 if pred_pairs == true_pairs else 0.0
            
            exact_match_scores.append(exact_match)
        
        exact_match_accuracy = np.mean(exact_match_scores)
        
        results = {
            'aspect_extraction': aspect_metrics,
            'sentiment_classification': sentiment_metrics,
            'exact_match_accuracy': exact_match_accuracy,
            'predictions': predictions
        }
        
        return results
    
    def _evaluate_aspect_extraction(self, pred_aspects: List[List[str]], 
                                   true_aspects: List[List[str]]) -> Dict:
        """
        Evaluate aspect extraction at aspect level.
        
        Args:
            pred_aspects: Predicted aspects for each text
            true_aspects: True aspects for each text
            
        Returns:
            Aspect extraction metrics
        """
        if not true_aspects or all(not aspects for aspects in true_aspects):
            # If no true aspects provided, return basic statistics
            return {
                'avg_aspects_per_text': np.mean([len(aspects) for aspects in pred_aspects]),
                'total_predicted_aspects': sum(len(aspects) for aspects in pred_aspects),
                'unique_aspects': len(set(asp for aspects in pred_aspects for asp in aspects))
            }
        
        # Calculate precision, recall, F1 at aspect level
        total_predicted = 0
        total_true = 0
        total_correct = 0
        
        for pred, true in zip(pred_aspects, true_aspects):
            pred_set = set(pred)
            true_set = set(true)
            
            total_predicted += len(pred_set)
            total_true += len(true_set)
            total_correct += len(pred_set & true_set)
        
        precision = total_correct / total_predicted if total_predicted > 0 else 0
        recall = total_correct / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_predicted': total_predicted,
            'total_true': total_true,
            'total_correct': total_correct
        }
    
    def _evaluate_aspect_sentiment_pairs(self, pred_aspects: List[List[str]], 
                                        pred_sentiments: List[List[str]],
                                        true_aspects: List[List[str]], 
                                        true_sentiments: List[List[str]]) -> Dict:
        """
        Evaluate sentiment classification for aspect-sentiment pairs.
        
        Args:
            pred_aspects: Predicted aspects
            pred_sentiments: Predicted sentiments
            true_aspects: True aspects
            true_sentiments: True sentiments
            
        Returns:
            Sentiment classification metrics
        """
        # Collect sentiment predictions for correctly extracted aspects
        correct_sentiments = []
        predicted_sentiments = []
        
        for pred_asp, pred_sent, true_asp, true_sent in zip(
            pred_aspects, pred_sentiments, true_aspects, true_sentiments
        ):
            for i, aspect in enumerate(pred_asp):
                if i < len(pred_sent) and aspect in true_asp:
                    # Find the sentiment for this aspect in true data
                    try:
                        true_idx = true_asp.index(aspect)
                        if true_idx < len(true_sent):
                            correct_sentiments.append(true_sent[true_idx])
                            predicted_sentiments.append(pred_sent[i])
                    except ValueError:
                        continue
        
        if not correct_sentiments:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Map sentiments to numerical labels
        sentiment_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        correct_labels = [sentiment_to_id.get(sent, 1) for sent in correct_sentiments]
        predicted_labels = [sentiment_to_id.get(sent, 1) for sent in predicted_sentiments]
        
        accuracy = accuracy_score(correct_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            correct_labels, predicted_labels, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(correct_labels)
        }
    
    def _analyze_confidence(self, true_labels: List[int], predictions: List[int], 
                           probabilities: List[np.ndarray]) -> Dict:
        """
        Analyze prediction confidence patterns.
        
        Args:
            true_labels: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
            
        Returns:
            Confidence analysis results
        """
        confidences = [prob.max() for prob in probabilities]
        correct_mask = np.array(true_labels) == np.array(predictions)
        
        # Overall confidence statistics
        avg_confidence = np.mean(confidences)
        correct_confidence = np.mean([conf for conf, correct in zip(confidences, correct_mask) if correct])
        incorrect_confidence = np.mean([conf for conf, correct in zip(confidences, correct_mask) if not correct])
        
        # Confidence bins
        confidence_bins = [0.0, 0.5, 0.7, 0.9, 1.0]
        bin_accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            min_conf, max_conf = confidence_bins[i], confidence_bins[i + 1]
            mask = (np.array(confidences) >= min_conf) & (np.array(confidences) < max_conf)
            if mask.sum() > 0:
                bin_accuracy = correct_mask[mask].mean()
                bin_accuracies.append({
                    'range': f"{min_conf:.1f}-{max_conf:.1f}",
                    'accuracy': bin_accuracy,
                    'count': mask.sum()
                })
        
        return {
            'avg_confidence': avg_confidence,
            'correct_confidence': correct_confidence,
            'incorrect_confidence': incorrect_confidence,
            'confidence_bins': bin_accuracies
        }
    
    def _analyze_classification_errors(self, true_labels: List[int], predictions: List[int],
                                     prediction_details: List[Dict]) -> Dict:
        """
        Analyze classification errors and patterns.
        
        Args:
            true_labels: True labels
            predictions: Predicted labels
            prediction_details: Detailed prediction information
            
        Returns:
            Error analysis results
        """
        errors = []
        error_patterns = defaultdict(int)
        
        for true_label, pred_label, details in zip(true_labels, predictions, prediction_details):
            if true_label != pred_label:
                error_type = f"{self.label_names[true_label]} -> {self.label_names[pred_label]}"
                error_patterns[error_type] += 1
                
                errors.append({
                    'true_label': self.label_names[true_label],
                    'predicted_label': self.label_names[pred_label],
                    'confidence': details['confidence'],
                    'probabilities': details['probabilities'].tolist()
                })
        
        # Most common error patterns
        common_errors = dict(Counter(error_patterns).most_common(5))
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(true_labels),
            'common_patterns': common_errors,
            'detailed_errors': errors[:10]  # Top 10 errors for inspection
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_confidence_distribution(self, confidences: List[float], 
                                   correct_mask: List[bool],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confidence score distribution.
        
        Args:
            confidences: Prediction confidences
            correct_mask: Whether predictions were correct
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        
        correct_conf = [conf for conf, correct in zip(confidences, correct_mask) if correct]
        incorrect_conf = [conf for conf, correct in zip(confidences, correct_mask) if not correct]
        
        plt.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green', density=True)
        plt.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red', density=True)
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Density')
        plt.title('Confidence Score Distribution')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def generate_evaluation_report(self, results: Dict, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("ABSA EVALUATION REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Overall metrics
        if 'accuracy' in results:
            report_lines.append(f"Overall Accuracy: {results['accuracy']:.4f}")
            report_lines.append(f"Precision: {results['precision']:.4f}")
            report_lines.append(f"Recall: {results['recall']:.4f}")
            report_lines.append(f"F1-Score: {results['f1']:.4f}")
            report_lines.append("")
        
        # Task-specific results
        if 'aspect_extraction' in results:
            asp_results = results['aspect_extraction']
            report_lines.append("ASPECT EXTRACTION:")
            report_lines.append(f"  Precision: {asp_results.get('precision', 0):.4f}")
            report_lines.append(f"  Recall: {asp_results.get('recall', 0):.4f}")
            report_lines.append(f"  F1-Score: {asp_results.get('f1', 0):.4f}")
            report_lines.append("")
        
        if 'sentiment_classification' in results:
            sent_results = results['sentiment_classification']
            report_lines.append("SENTIMENT CLASSIFICATION:")
            report_lines.append(f"  Accuracy: {sent_results.get('accuracy', 0):.4f}")
            report_lines.append(f"  F1-Score: {sent_results.get('f1', 0):.4f}")
            report_lines.append("")
        
        # Confidence analysis
        if 'confidence_analysis' in results:
            conf_results = results['confidence_analysis']
            report_lines.append("CONFIDENCE ANALYSIS:")
            report_lines.append(f"  Average Confidence: {conf_results['avg_confidence']:.4f}")
            report_lines.append(f"  Correct Predictions Confidence: {conf_results['correct_confidence']:.4f}")
            report_lines.append(f"  Incorrect Predictions Confidence: {conf_results['incorrect_confidence']:.4f}")
            report_lines.append("")
        
        # Error analysis
        if 'error_analysis' in results:
            error_results = results['error_analysis']
            report_lines.append("ERROR ANALYSIS:")
            report_lines.append(f"  Total Errors: {error_results['total_errors']}")
            report_lines.append(f"  Error Rate: {error_results['error_rate']:.4f}")
            report_lines.append("  Common Error Patterns:")
            for pattern, count in error_results['common_patterns'].items():
                report_lines.append(f"    {pattern}: {count}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


class ABSABenchmark:
    """
    Benchmark suite for comparing different ABSA models and approaches.
    """
    
    def __init__(self, evaluator: ABSAEvaluator):
        self.evaluator = evaluator
        self.results = {}
        self.logger = logging.getLogger(__name__)
    
    def add_model_results(self, model_name: str, results: Dict):
        """
        Add results for a model to the benchmark.
        
        Args:
            model_name: Name of the model
            results: Evaluation results
        """
        self.results[model_name] = results
        self.logger.info(f"Added results for {model_name}")
    
    def compare_models(self, metric: str = 'f1') -> pd.DataFrame:
        """
        Compare models based on a specific metric.
        
        Args:
            metric: Metric to compare on
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {'model': model_name}
            
            # Extract metric from results
            if metric in results:
                row[metric] = results[metric]
            elif 'aspect_extraction' in results and metric in results['aspect_extraction']:
                row[f'aspect_{metric}'] = results['aspect_extraction'][metric]
            elif 'sentiment_classification' in results and metric in results['sentiment_classification']:
                row[f'sentiment_{metric}'] = results['sentiment_classification'][metric]
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_model_comparison(self, metrics: List[str] = ['f1', 'precision', 'recall'],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot model comparison chart.
        
        Args:
            metrics: Metrics to include in comparison
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        comparison_df = self.compare_models()
        
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(comparison_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                plt.bar(x + i * width, comparison_df[metric], width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('ABSA Model Comparison')
        plt.xticks(x + width, comparison_df['model'], rotation=45)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
