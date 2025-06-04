"""
ABSA Models

This module contains various models for Aspect-Based Sentiment Analysis:
- DeBERTa for Aspect Term Extraction (ATE)
- DeBERTa for Aspect Sentiment Classification (ASC)
- BERT-based end-to-end models
- Traditional BiLSTM-CRF models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    DebertaV2Model, DebertaV2Config,
    BertModel, BertConfig
)
from TorchCRF import CRF
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from peft import LoraConfig, get_peft_model, TaskType


class DeBERTaATE(nn.Module):
    """
    DeBERTa model for Aspect Term Extraction (ATE).
    
    Uses token classification approach with BIO tagging scheme.
    B-ASP, I-ASP: Beginning/Inside aspect terms
    B-OP, I-OP: Beginning/Inside opinion terms  
    O: Outside any term
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_labels = config['model']['num_labels']
        
        # Load DeBERTa model
        model_name = config['model']['name']
        if config.get('lora', {}).get('use_lora', False):
            from transformers import AutoModelForTokenClassification
            self.deberta_config = AutoConfig.from_pretrained(model_name, num_labels=self.num_labels)
            self.deberta = AutoModelForTokenClassification.from_pretrained(model_name, config=self.deberta_config)
            self._apply_lora(config['lora'])
            self.classifier = None  # Use built-in classifier
        else:
            self.deberta_config = AutoConfig.from_pretrained(model_name)
            self.deberta = AutoModel.from_pretrained(model_name, config=self.deberta_config)
            self.classifier = nn.Linear(self.deberta.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(config['model']['dropout'])
        
        # Optional CRF layer for sequence labeling
        self.use_crf = config.get('use_crf', False)
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)
        
        # Class weights for loss
        class_weights = config.get('class_weights', {})
        if class_weights:
            weights = torch.ones(self.num_labels)
            label_to_id = config['labels']
            for label, weight in class_weights.items():
                if label in label_to_id:
                    weights[label_to_id[label]] = weight
            self.register_buffer('class_weights', weights)
        else:
            self.class_weights = None
    
    def _apply_lora(self, lora_config: Dict):
        """Apply LoRA to the model for efficient fine-tuning."""
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            target_modules=lora_config['target_modules']
        )
        self.deberta = get_peft_model(self.deberta, peft_config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        if self.classifier is None:
            # LoRA/PEFT: use built-in classifier and pass labels
            outputs = self.deberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits = outputs.logits
            loss = outputs.loss if labels is not None else None
        else:
            outputs = self.deberta(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            sequence_output = outputs.last_hidden_state
            sequence_output = self.dropout(sequence_output)
            logits = self.classifier(sequence_output)
            loss = None
            if labels is not None:
                if self.use_crf:
                    loss = -self.crf(logits, labels, mask=attention_mask.bool())
                else:
                    loss_fct = nn.CrossEntropyLoss(
                        weight=self.class_weights,
                        ignore_index=-100
                    )
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
        return {
            'loss': loss,
            'logits': logits,
            'predictions': self.decode(logits, attention_mask)
        }
    
    def decode(self, logits, attention_mask):
        """Decode predictions from logits."""
        if self.use_crf:
            predictions = self.crf.decode(logits, mask=attention_mask.bool())
            return [pred for pred in predictions]  # List of sequences
        else:
            predictions = torch.argmax(logits, dim=-1)
            return predictions
    
    def extract_aspects(self, input_ids, attention_mask, tokenizer):
        """Extract aspect terms from input."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            predictions = outputs['predictions']
            
        aspects = []
        for i, (pred_seq, input_seq, mask) in enumerate(zip(predictions, input_ids, attention_mask)):
            tokens = tokenizer.convert_ids_to_tokens(input_seq)
            seq_len = mask.sum().item()
            
            current_aspect = []
            aspect_terms = []
            
            for j in range(seq_len):
                if isinstance(predictions, list):
                    pred_label = pred_seq[j] if j < len(pred_seq) else 4  # O label
                else:
                    pred_label = pred_seq[j].item()
                
                token = tokens[j]
                
                if pred_label == 0:  # B-ASP
                    if current_aspect:  # Save previous aspect
                        aspect_terms.append(tokenizer.convert_tokens_to_string(current_aspect))
                    current_aspect = [token]
                elif pred_label == 1 and current_aspect:  # I-ASP
                    current_aspect.append(token)
                else:  # O, B-OP, I-OP
                    if current_aspect:
                        aspect_terms.append(tokenizer.convert_tokens_to_string(current_aspect))
                        current_aspect = []
            
            # Don't forget the last aspect
            if current_aspect:
                aspect_terms.append(tokenizer.convert_tokens_to_string(current_aspect))
            
            aspects.append(aspect_terms)
        
        return aspects


class DeBERTaASC(nn.Module):
    """
    DeBERTa model for Aspect Sentiment Classification (ASC).
    
    Takes input in format: "Aspect: {aspect}. Sentence: {sentence}"
    Classifies sentiment as positive, negative, or neutral.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_labels = config['model']['num_labels']  # 3 for pos/neg/neu
        
        # Load DeBERTa model
        model_name = config['model']['name']
        self.deberta_config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name, config=self.deberta_config)
        
        # Apply LoRA if specified
        if config.get('lora', {}).get('use_lora', False):
            self._apply_lora(config['lora'])
        
        # Classification head
        self.dropout = nn.Dropout(config['model']['dropout'])
        self.classifier = nn.Linear(self.deberta.config.hidden_size, self.num_labels)
        
        # Class weights for loss
        class_weights = config.get('class_weights', {})
        if class_weights:
            weights = torch.ones(self.num_labels)
            sentiment_to_id = config['sentiments']
            for sentiment, weight in class_weights.items():
                if sentiment in sentiment_to_id:
                    weights[sentiment_to_id[sentiment]] = weight
            self.register_buffer('class_weights', weights)
        else:
            self.class_weights = None
    
    def _apply_lora(self, lora_config: Dict):
        """Apply LoRA to the model for efficient fine-tuning."""
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            target_modules=lora_config['target_modules']
        )
        self.deberta = get_peft_model(self.deberta, peft_config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)
        return {
            'loss': loss,
            'logits': logits,
            'predictions': torch.argmax(logits, dim=-1)
        }


class BERTForABSA(nn.Module):
    """
    BERT-based model for end-to-end ABSA.
    
    Joint model that can predict multiple aspect-sentiment pairs
    for a given input text.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Load BERT model
        model_name = config['model']['name']
        self.bert_config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.bert_config)
        
        # Task-specific parameters
        self.num_aspect_categories = config['model'].get('num_aspect_labels', 8)
        self.num_sentiment_classes = config['model'].get('num_sentiment_labels', 3)
        
        # Multi-head approach for multiple aspects
        self.dropout = nn.Dropout(config['model']['dropout'])
        
        # Single multi-label aspect head
        self.aspect_classifier = nn.Linear(self.bert.config.hidden_size, self.num_aspect_categories)
        
        # Sentiment heads: one per aspect category
        self.sentiment_classifiers = nn.ModuleList([
            nn.Linear(self.bert.config.hidden_size, self.num_sentiment_classes)
            for _ in range(self.num_aspect_categories)
        ])
        
        # Attention mechanism for aspect-specific representations
        self.aspect_attention = nn.MultiheadAttention(
            self.bert.config.hidden_size, 
            num_heads=8, 
            dropout=config['model']['dropout']
        )
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, aspect_labels=None, sentiment_labels=None, **kwargs):
        """Forward pass for joint aspect-sentiment prediction."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token, [batch, hidden]
        batch_size = sequence_output.size(0)
        # Multihead attention
        query = pooled_output.unsqueeze(0)  # [1, batch, hidden]
        key = value = sequence_output.transpose(0, 1)  # [seq_len, batch, hidden]
        attended_output, _ = self.aspect_attention(query, key, value)
        attended_output = attended_output.squeeze(0)  # [batch, hidden]
        attended_output = self.dropout(attended_output)
        # Aspect prediction
        aspect_logits = self.aspect_classifier(attended_output)  # [batch, num_aspect_categories]
        # Sentiment prediction for each aspect category
        sentiment_logits = []
        for i in range(self.num_aspect_categories):
            sentiment_logit = self.sentiment_classifiers[i](attended_output)  # [batch, num_sentiment_classes]
            sentiment_logits.append(sentiment_logit)
        sentiment_logits = torch.stack(sentiment_logits, dim=1)  # [batch, num_aspect_categories, num_sentiment_classes]
        output = {
            'aspect_logits': aspect_logits,
            'sentiment_logits': sentiment_logits,
            'aspect_predictions': (torch.sigmoid(aspect_logits) > 0.5).long(),
            'sentiment_predictions': torch.argmax(sentiment_logits, dim=-1)
        }
        # Optionally compute loss if labels are provided
        loss = None
        if labels is not None:
            if isinstance(labels, dict):
                aspect_labels = labels.get('aspect_labels', None)
                sentiment_labels = labels.get('sentiment_labels', None)
            else:
                aspect_labels = aspect_labels
                sentiment_labels = sentiment_labels
        if aspect_labels is not None and sentiment_labels is not None:
            loss_fct_aspect = nn.BCEWithLogitsLoss()
            # Add label smoothing to sentiment loss
            loss_fct_sentiment = nn.CrossEntropyLoss(label_smoothing=0.1)
            # aspect_labels: [batch, num_aspect_categories] (float, 0/1)
            aspect_loss = loss_fct_aspect(aspect_logits, aspect_labels.float())
            # sentiment_labels: [batch, num_aspect_categories] (int, 0/1/2 or -100 for not present)
            # Only compute sentiment loss for present aspects
            active_sentiment_loss = []
            for i in range(self.num_aspect_categories):
                mask = (aspect_labels[:, i] == 1)
                if mask.sum() == 0:
                    continue
                logits_i = sentiment_logits[mask, i, :]
                labels_i = sentiment_labels[mask, i]
                active_sentiment_loss.append(loss_fct_sentiment(logits_i, labels_i))
            if active_sentiment_loss:
                sentiment_loss = torch.stack(active_sentiment_loss).mean()
            else:
                sentiment_loss = torch.tensor(0.0, device=aspect_logits.device)
            loss = aspect_loss + sentiment_loss
        output['loss'] = loss
        return output


class EndToEndABSA(nn.Module):
    """
    End-to-end ABSA model using sequence-to-sequence approach.
    
    Generates structured output in format:
    "{aspect1}:{sentiment1}, {aspect2}:{sentiment2}, ..."
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Use T5 or BART for sequence-to-sequence
        from transformers import T5ForConditionalGeneration, T5Config
        
        model_name = config['model'].get('name', 't5-base')
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Special tokens for ABSA format
        self.aspect_sep_token = "<asp>"
        self.sentiment_sep_token = "<sent>"
        self.pair_sep_token = "<pair>"
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for generation."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }
    
    def generate(self, input_ids, attention_mask=None, max_length=100):
        """Generate aspect-sentiment pairs."""
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        return outputs


class BiLSTMCRF(nn.Module):
    """
    Traditional BiLSTM-CRF model for aspect term extraction.
    
    Uses word embeddings + BiLSTM + CRF for sequence labeling.
    Useful as baseline comparison to transformer models.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Model parameters
        vocab_size = config['vocab_size']
        embedding_dim = config['embedding_dim']
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        self.num_labels = config['num_labels']
        dropout = config.get('dropout', 0.1)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Pre-trained embeddings if provided
        if 'pretrained_embeddings' in config:
            self.embedding.weight.data.copy_(torch.tensor(config['pretrained_embeddings']))
            if config.get('freeze_embeddings', False):
                self.embedding.weight.requires_grad = False
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout and projection layer
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, self.num_labels)
        
        # CRF layer
        self.crf = CRF(self.num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        # Embedding
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        
        # Projection to tag space
        emissions = self.hidden2tag(lstm_out)
        
        # Calculate loss and predictions
        if labels is not None:
            # CRF loss
            mask = attention_mask.bool() if attention_mask is not None else None
            loss = -self.crf(emissions, labels, mask=mask)
            predictions = self.crf.decode(emissions, mask=mask)
        else:
            loss = None
            mask = attention_mask.bool() if attention_mask is not None else None
            predictions = self.crf.decode(emissions, mask=mask)
        
        return {
            'loss': loss,
            'logits': emissions,
            'predictions': predictions
        }


class ABSAPredictor:
    """
    High-level predictor that combines ATE and ASC models for full ABSA pipeline.
    """
    
    def __init__(self, ate_model, asc_model, tokenizer, config: Dict):
        self.ate_model = ate_model
        self.asc_model = asc_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.ate_model.to(self.device)
        self.asc_model.to(self.device)
        
        # Set to eval mode
        self.ate_model.eval()
        self.asc_model.eval()
        
        # Sentiment mapping
        self.id_to_sentiment = {v: k for k, v in config.get('sentiments', {}).items()}
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict aspects and sentiments for input texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of dictionaries with aspects and sentiments
        """
        results = []
        
        for text in texts:
            # Step 1: Extract aspects using ATE model
            ate_inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config['model']['max_length']
            ).to(self.device)
            
            aspects = self.ate_model.extract_aspects(
                ate_inputs['input_ids'],
                ate_inputs['attention_mask'],
                self.tokenizer
            )[0]  # Get first (and only) result
            
            # Step 2: Classify sentiment for each aspect
            aspect_sentiments = []
            for aspect in aspects:
                # Format input for ASC
                asc_input = f"Aspect: {aspect}. Sentence: {text}"
                asc_inputs = self.tokenizer(
                    asc_input,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config['model']['max_length']
                ).to(self.device)
                
                with torch.no_grad():
                    asc_outputs = self.asc_model(**asc_inputs)
                    sentiment_id = asc_outputs['predictions'].item()
                    sentiment = self.id_to_sentiment.get(sentiment_id, 'neutral')
                
                aspect_sentiments.append({
                    'aspect': aspect,
                    'sentiment': sentiment,
                    'confidence': torch.softmax(asc_outputs['logits'], dim=-1).max().item()
                })
            
            results.append({
                'text': text,
                'aspects': aspect_sentiments,
                'num_aspects': len(aspect_sentiments)
            })
        
        return results
    
    def predict_single(self, text: str) -> Dict:
        """Predict aspects and sentiments for a single text."""
        return self.predict([text])[0]
