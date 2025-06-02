"""
ABSA-specific training utilities and dataset classes.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
import json

class ABSADataset(Dataset):
    """Dataset class for ABSA tasks."""
    
    def __init__(self, texts: List[str], labels: List, tokenizer, task: str = "ate", max_length: int = 512):
        """
        Initialize ABSA dataset.
        
        Args:
            texts: List of input texts
            labels: List of labels (BIO tags for ATE, sentiments for ASC)
            tokenizer: Hugging Face tokenizer
            task: Task type ("ate", "asc", "end_to_end")
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length
        
        if task == "asc":
            self.label_map = {"positive": 0, "negative": 1, "neutral": 2}
        elif task == "ate":
            self.label_map = {"O": 0, "B-ASPECT": 1, "I-ASPECT": 2}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        if self.task == "ate":
            # For ATE, labels are BIO tags
            encoding = self.tokenizer(
                text.split(),  # Already tokenized
                is_split_into_words=True,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Align labels with tokenized input
            aligned_labels = self._align_labels_with_tokens(text.split(), labels, encoding)
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(aligned_labels, dtype=torch.long)
            }
        
        elif self.task == "asc":
            # For ASC, text might be "text [SEP] aspect"
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            label_idx = self.label_map.get(labels, 2)  # Default to neutral
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(label_idx, dtype=torch.long)
            }
    
    def _align_labels_with_tokens(self, tokens: List[str], labels: List[str], encoding) -> List[int]:
        """Align BIO labels with tokenizer output."""
        word_ids = encoding.word_ids()
        aligned_labels = []
        
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                # First token of a word
                if word_idx < len(labels):
                    aligned_labels.append(self.label_map.get(labels[word_idx], 0))
                else:
                    aligned_labels.append(0)  # O tag
            else:
                # Continuation of a word
                if word_idx < len(labels) and labels[word_idx].startswith('B-'):
                    # Convert B- to I- for continuation tokens
                    aligned_labels.append(self.label_map.get('I-ASPECT', 2))
                elif word_idx < len(labels):
                    aligned_labels.append(self.label_map.get(labels[word_idx], 0))
                else:
                    aligned_labels.append(0)
            
            previous_word_idx = word_idx
        
        return aligned_labels

class ABSATrainerUtils:
    """Utility functions for ABSA training."""
    
    @staticmethod
    def create_dataloaders(train_texts: List[str], train_labels: List,
                          val_texts: List[str], val_labels: List,
                          tokenizer, task: str = "ate", batch_size: int = 16,
                          max_length: int = 512) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        train_dataset = ABSADataset(train_texts, train_labels, tokenizer, task, max_length)
        val_dataset = ABSADataset(val_texts, val_labels, tokenizer, task, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    @staticmethod
    def compute_ate_metrics(predictions: List[List[str]], labels: List[List[str]]) -> Dict[str, float]:
        """Compute metrics for Aspect Term Extraction."""
        total_pred = sum(len(pred) for pred in predictions)
        total_true = sum(len(label) for label in labels)
        total_correct = 0
        
        for pred, true in zip(predictions, labels):
            pred_set = set(pred)
            true_set = set(true)
            total_correct += len(pred_set & true_set)
        
        precision = total_correct / total_pred if total_pred > 0 else 0
        recall = total_correct / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def compute_asc_metrics(predictions: List[str], labels: List[str]) -> Dict[str, float]:
        """Compute metrics for Aspect Sentiment Classification."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def save_predictions(predictions: List, labels: List, texts: List[str], 
                        output_path: str, task: str = "ate"):
        """Save predictions to file."""
        results = []
        
        for i, (pred, label, text) in enumerate(zip(predictions, labels, texts)):
            result = {
                'id': i,
                'text': text,
                'prediction': pred,
                'ground_truth': label
            }
            results.append(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

def collate_absa_batch(batch):
    """Custom collate function for ABSA batches."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def prepare_absa_training_data(texts: List[str], aspect_data: List[List[Dict]], 
                              task: str = "ate") -> Tuple[List[str], List]:
    """
    Prepare training data for specific ABSA task.
    
    Args:
        texts: List of input texts
        aspect_data: List of aspect information
        task: Task type ("ate" or "asc")
    
    Returns:
        Tuple of prepared texts and labels
    """
    if task == "ate":
        from data_utils import prepare_ate_data
        return prepare_ate_data(texts, aspect_data)
    elif task == "asc":
        from data_utils import prepare_asc_data
        return prepare_asc_data(texts, aspect_data)
    else:
        raise ValueError(f"Unknown task: {task}")
