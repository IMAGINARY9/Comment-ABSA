"""
ABSA Training Module

This module provides training utilities for Aspect-Based Sentiment Analysis:
- Trainers for ATE (Aspect Term Extraction)
- Trainers for ASC (Aspect Sentiment Classification) 
- End-to-end training pipeline
- Advanced training techniques (LoRA, class weighting, etc.)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

from .models import DeBERTaATE, DeBERTaASC, BERTForABSA, EndToEndABSA
from .preprocessing import ABSADataset, ABSADataLoader


class ABSATrainer:
    """
    Base trainer class for ABSA models.
    
    Provides common training infrastructure and utilities.
    """
    
    def __init__(self, model: nn.Module, config: Dict, use_wandb: bool = False):
        """
        Initialize trainer.
        
        Args:
            model: ABSA model to train
            config: Training configuration
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.config = config
        self.use_wandb = use_wandb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Initialize optimizer and scheduler (will be set in train())
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.early_stopping_counter = 0
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        # Initialize wandb if specified
        if self.use_wandb:
            self.init_wandb()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/training_{int(time.time())}.log'),
                logging.StreamHandler()
            ]
        )
    
    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.get('project_name', 'absa-training'),
            config=self.config,
            name=f"{self.config['model']['name']}_{self.config['model']['task']}"
        )
        wandb.watch(self.model)
    
    def setup_optimizer_scheduler(self, train_dataloader: DataLoader):
        """
        Setup optimizer and learning rate scheduler.
        
        Args:
            train_dataloader: Training data loader
        """
        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['training']['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config['training']['learning_rate'],
            eps=1e-8
        )
        
        # Scheduler
        num_training_steps = len(train_dataloader) * self.config['training']['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )
    
    def save_checkpoint(self, epoch: int, metric: float, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metric: Current metric value
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metric': metric,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config['paths']['model_dir']) / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config['paths']['model_dir']) / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with metric: {metric:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['metric']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def calculate_loss(self, outputs, labels):
        """Calculate loss based on model outputs and labels."""
        if 'loss' in outputs and outputs['loss'] is not None:
            return outputs['loss']
        else:
            # Fallback loss calculation
            logits = outputs['logits']
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    def log_metrics(self, metrics: Dict, step: int, prefix: str = "train"):
        """
        Log metrics to console and wandb.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step
            prefix: Metric prefix (train/val)
        """
        # Log to console
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {prefix} | {metric_str}")
        
        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
            wandb_metrics['step'] = step
            wandb_metrics['epoch'] = self.current_epoch
            wandb.log(wandb_metrics)
    
    def should_stop_early(self, metric: float, patience: int = 5) -> bool:
        """
        Check if training should stop early.
        
        Args:
            metric: Current validation metric
            patience: Patience for early stopping
            
        Returns:
            True if training should stop
        """
        if metric > self.best_metric:
            self.best_metric = metric
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= patience


class ATETrainer(ABSATrainer):
    """
    Trainer for Aspect Term Extraction (ATE) models.
    
    Handles token classification training with BIO tagging.
    """
    
    def __init__(self, model: DeBERTaATE, config: Dict, use_wandb: bool = False):
        super().__init__(model, config, use_wandb)
        
        # ATE-specific configuration
        self.label_names = ['B-ASP', 'I-ASP', 'B-OP', 'I-OP', 'O']
        self.ignore_index = -100
    
    def train_epoch(self, train_dataloader: DataLoader) -> Dict:
        """
        Train for one epoch.
        
        Args:
            train_dataloader: Training data loader
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            predictions = outputs['predictions']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            if (batch_idx + 1) % self.config['training'].get('gradient_accumulation_steps', 1) == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Collect metrics
            total_loss += loss.item()
            
            # Flatten predictions and labels for metric calculation
            if isinstance(predictions, list):
                # CRF predictions
                for pred_seq, label_seq, mask in zip(predictions, labels, attention_mask):
                    valid_length = mask.sum().item()
                    all_predictions.extend(pred_seq[:valid_length])
                    valid_labels = label_seq[:valid_length]
                    valid_labels = valid_labels[valid_labels != self.ignore_index]
                    all_labels.extend(valid_labels.cpu().numpy())
            else:
                # Standard predictions
                active_mask = attention_mask.view(-1) == 1
                active_labels = labels.view(-1)[active_mask]
                active_predictions = predictions.view(-1)[active_mask]
                
                # Filter out ignore_index
                valid_mask = active_labels != self.ignore_index
                all_predictions.extend(active_predictions[valid_mask].cpu().numpy())
                all_labels.extend(active_labels[valid_mask].cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log training step
            if self.global_step % self.config['training']['logging_steps'] == 0:
                step_metrics = {'loss': loss.item()}
                self.log_metrics(step_metrics, self.global_step, 'train_step')
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict:
        """
        Evaluate model on validation/test set.
        
        Args:
            eval_dataloader: Evaluation data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                predictions = outputs['predictions']
                
                total_loss += loss.item()
                
                # Collect predictions and labels
                if isinstance(predictions, list):
                    # CRF predictions
                    for pred_seq, label_seq, mask in zip(predictions, labels, attention_mask):
                        valid_length = mask.sum().item()
                        all_predictions.extend(pred_seq[:valid_length])
                        valid_labels = label_seq[:valid_length]
                        valid_labels = valid_labels[valid_labels != self.ignore_index]
                        all_labels.extend(valid_labels.cpu().numpy())
                else:
                    # Standard predictions
                    active_mask = attention_mask.view(-1) == 1
                    active_labels = labels.view(-1)[active_mask]
                    active_predictions = predictions.view(-1)[active_mask]
                    
                    # Filter out ignore_index
                    valid_mask = active_labels != self.ignore_index
                    all_predictions.extend(active_predictions[valid_mask].cpu().numpy())
                    all_labels.extend(active_labels[valid_mask].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(eval_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=self.label_names,
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': class_report
        }
        
        return metrics
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None):
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
        """
        self.logger.info("Starting ATE training...")
        
        # Setup optimizer and scheduler
        self.setup_optimizer_scheduler(train_dataloader)
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader)
            self.log_metrics(train_metrics, self.global_step, 'train')
            
            # Validation
            if val_dataloader:
                val_metrics = self.evaluate(val_dataloader)
                self.log_metrics(val_metrics, self.global_step, 'val')
                
                # Save checkpoint
                val_f1 = val_metrics['f1']
                is_best = val_f1 > self.best_metric
                self.save_checkpoint(epoch, val_f1, is_best)
                
                # Early stopping
                if self.should_stop_early(val_f1):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                # Save checkpoint based on training metrics
                train_f1 = train_metrics['f1']
                is_best = train_f1 > self.best_metric
                self.save_checkpoint(epoch, train_f1, is_best)
        
        self.logger.info("Training completed!")
        
        if self.use_wandb:
            wandb.finish()


class ASCTrainer(ABSATrainer):
    """
    Trainer for Aspect Sentiment Classification (ASC) models.
    
    Handles sequence classification for aspect-sentiment pairs.
    """
    
    def __init__(self, model: DeBERTaASC, config: Dict, use_wandb: bool = False):
        super().__init__(model, config, use_wandb)
        
        # ASC-specific configuration
        self.sentiment_names = ['negative', 'neutral', 'positive']
    
    def train_epoch(self, train_dataloader: DataLoader) -> Dict:
        """
        Train for one epoch.
        
        Args:
            train_dataloader: Training data loader
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            predictions = outputs['predictions']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            if (batch_idx + 1) % self.config['training'].get('gradient_accumulation_steps', 1) == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Collect metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log training step
            if self.global_step % self.config['training']['logging_steps'] == 0:
                step_metrics = {'loss': loss.item()}
                self.log_metrics(step_metrics, self.global_step, 'train_step')
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict:
        """
        Evaluate model on validation/test set.
        
        Args:
            eval_dataloader: Evaluation data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                predictions = outputs['predictions']
                
                total_loss += loss.item()
                
                # Collect predictions and labels
                probabilities = torch.softmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(eval_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Per-class metrics
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=self.sentiment_names,
            output_dict=True
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': class_report,
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }
        
        return metrics
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None):
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
        """
        self.logger.info("Starting ASC training...")
        
        # Setup optimizer and scheduler
        self.setup_optimizer_scheduler(train_dataloader)
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader)
            self.log_metrics(train_metrics, self.global_step, 'train')
            
            # Validation
            if val_dataloader:
                val_metrics = self.evaluate(val_dataloader)
                self.log_metrics(val_metrics, self.global_step, 'val')
                
                # Save checkpoint
                val_f1 = val_metrics['f1']
                is_best = val_f1 > self.best_metric
                self.save_checkpoint(epoch, val_f1, is_best)
                
                # Early stopping
                if self.should_stop_early(val_f1):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                # Save checkpoint based on training metrics
                train_f1 = train_metrics['f1']
                is_best = train_f1 > self.best_metric
                self.save_checkpoint(epoch, train_f1, is_best)
        
        self.logger.info("Training completed!")
        
        if self.use_wandb:
            wandb.finish()


class MultiTaskABSATrainer(ABSATrainer):
    """
    Multi-task trainer for joint ATE and ASC training.
    
    Trains a model to simultaneously extract aspects and classify sentiments.
    """
    
    def __init__(self, model: BERTForABSA, config: Dict, use_wandb: bool = False):
        super().__init__(model, config, use_wandb)
        
        # Multi-task configuration
        self.task_weights = config.get('task_weights', {'ate': 1.0, 'asc': 1.0})
    
    def train_epoch(self, train_dataloader: DataLoader) -> Dict:
        """
        Train for one epoch with multi-task learning.
        
        Args:
            train_dataloader: Training data loader
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        total_loss = 0
        aspect_predictions = []
        aspect_labels = []
        sentiment_predictions = []
        sentiment_labels = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            aspect_labels_batch = batch['aspect_labels'].to(self.device)
            sentiment_labels_batch = batch['sentiment_labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                aspect_labels=aspect_labels_batch,
                sentiment_labels=sentiment_labels_batch
            )
            
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            if (batch_idx + 1) % self.config['training'].get('gradient_accumulation_steps', 1) == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Collect metrics
            total_loss += loss.item()
            
            # Collect predictions for metrics calculation
            aspect_preds = outputs['aspect_predictions']
            sentiment_preds = outputs['sentiment_predictions']
            
            aspect_predictions.extend(aspect_preds.cpu().numpy().flatten())
            aspect_labels.extend(aspect_labels_batch.cpu().numpy().flatten())
            
            # Only collect sentiment metrics for aspects that are present
            for i in range(aspect_labels_batch.size(0)):
                for j in range(aspect_labels_batch.size(1)):
                    if aspect_labels_batch[i, j] == 1:  # Aspect is present
                        sentiment_predictions.append(sentiment_preds[i, j].item())
                        sentiment_labels.append(sentiment_labels_batch[i, j].item())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_dataloader)
        
        # Aspect metrics
        aspect_accuracy = accuracy_score(aspect_labels, aspect_predictions)
        aspect_precision, aspect_recall, aspect_f1, _ = precision_recall_fscore_support(
            aspect_labels, aspect_predictions, average='weighted'
        )
        
        # Sentiment metrics (only for present aspects)
        if sentiment_predictions:
            sentiment_accuracy = accuracy_score(sentiment_labels, sentiment_predictions)
            sentiment_precision, sentiment_recall, sentiment_f1, _ = precision_recall_fscore_support(
                sentiment_labels, sentiment_predictions, average='weighted'
            )
        else:
            sentiment_accuracy = sentiment_precision = sentiment_recall = sentiment_f1 = 0.0
        
        metrics = {
            'loss': avg_loss,
            'aspect_accuracy': aspect_accuracy,
            'aspect_precision': aspect_precision,
            'aspect_recall': aspect_recall,
            'aspect_f1': aspect_f1,
            'sentiment_accuracy': sentiment_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_recall': sentiment_recall,
            'sentiment_f1': sentiment_f1,
            'overall_f1': (aspect_f1 + sentiment_f1) / 2
        }
        
        return metrics
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None):
        """
        Main training loop for multi-task learning.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
        """
        self.logger.info("Starting multi-task ABSA training...")
        
        # Setup optimizer and scheduler
        self.setup_optimizer_scheduler(train_dataloader)
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader)
            self.log_metrics(train_metrics, self.global_step, 'train')
            
            # Validation
            if val_dataloader:
                val_metrics = self.evaluate(val_dataloader)
                self.log_metrics(val_metrics, self.global_step, 'val')
                
                # Save checkpoint based on overall F1
                val_f1 = val_metrics['overall_f1']
                is_best = val_f1 > self.best_metric
                self.save_checkpoint(epoch, val_f1, is_best)
                
                # Early stopping
                if self.should_stop_early(val_f1):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.logger.info("Training completed!")
        
        if self.use_wandb:
            wandb.finish()
