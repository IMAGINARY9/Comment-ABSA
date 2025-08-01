#!/usr/bin/env python3
"""
Prediction script for ABSA models.

This script performs inference using trained ABSA models on new text data,
supporting both single text prediction and batch processing.

Usage:
    python predict.py --model_path <path_to_model> --config <path_to_config> [--text <single_text>] [--input_file <file_with_texts>] [--output_file <output_file>] [--format <json|csv>]
"""

import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import DeBERTaATE, DeBERTaASC, BERTForABSA, EndToEndABSA
from transformers import AutoTokenizer
import torch.nn as nn
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(model_path, config, device):
    """Load trained ABSA model and tokenizer."""
    # Determine model class from config
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
    # Load checkpoint (support both .pt and .bin)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def preprocess_text(text, config):
    """Preprocess input text for ABSA prediction."""
    # Basic preprocessing (would be expanded based on config)
    text = text.strip()
    text = text.lower() if config.get('preprocessing', {}).get('lowercase', False) else text
    return text

def predict_single_text(model, tokenizer, text, config, ate_model=None, ate_tokenizer=None, ate_config=None):
    """Predict aspects and sentiments for a single text."""
    task = config['model'].get('task', 'token_classification')
    from src.preprocessing import ABSAPreprocessor
    
    # Get NER configuration if available
    model_config = config.get('model', {})
    ner_model_path = model_config.get('ner_model_path') if model_config.get('use_ner_features', False) else None
    ner_word_tokenizer_path = model_config.get('ner_word_tokenizer_path') if model_config.get('use_ner_features', False) else None
    ner_tag_vocab_path = model_config.get('ner_tag_vocab_path') if model_config.get('use_ner_features', False) else None
    ner_max_seq_length = model_config.get('ner_max_seq_length', 128)
    
    preproc = ABSAPreprocessor(
        tokenizer_name=config['model']['name'],
        ner_model_path=ner_model_path,
        ner_word_tokenizer_path=ner_word_tokenizer_path,
        ner_tag_vocab_path=ner_tag_vocab_path,
        ner_max_seq_length=ner_max_seq_length
    )
    cleaned_text = preproc.clean_text(text)
    device = next(model.parameters()).device
    with torch.no_grad():
        if task == 'token_classification' and ate_model is None:
            # ATE: Token classification (BIO tagging)
            max_length = config['model'].get('max_length', 128)
            # Tokenize as in training (split into words, then encode)
            tokens = nltk.word_tokenize(cleaned_text)
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            if 'token_type_ids' in encoding:
                encoding.pop('token_type_ids')
            encoding = {k: v.to(device) for k, v in encoding.items()}
            outputs = model(**encoding)
            # Get predicted label ids
            pred_ids = outputs['logits'].argmax(-1).squeeze(0).tolist()
            word_ids = encoding['input_ids'].shape[1]
            # Map predictions back to tokens
            word_tokens = tokens
            bio_tags = []
            for i, token in enumerate(word_tokens):
                if i < len(pred_ids):
                    tag_id = pred_ids[i]
                    tag = {0: 'B-ASP', 1: 'I-ASP', 2: 'B-OP', 3: 'I-OP', 4: 'O'}.get(tag_id, 'O')
                    bio_tags.append(tag)
                else:
                    bio_tags.append('O')
            # Extract aspect terms from BIO tags
            aspects = []
            current = []
            for token, tag in zip(word_tokens, bio_tags):
                if tag == 'B-ASP':
                    if current:
                        aspects.append(' '.join(current))
                        current = []
                    current = [token]
                elif tag == 'I-ASP' and current:
                    current.append(token)
                else:
                    if current:
                        aspects.append(' '.join(current))
                        current = []
            if current:
                aspects.append(' '.join(current))
            filtered_aspects = [(a, None) for a in aspects if a and len(a.strip()) > 1]
            result = {
                'text': text,
                'processed_text': cleaned_text,
                'aspects': filtered_aspects,
                'confidence': 1.0,
                'prediction_time': datetime.now().isoformat()
            }
        elif task == 'sequence_classification' and ate_model is not None and ate_tokenizer is not None and ate_config is not None:
            # Use ATE model to extract aspects from cleaned text
            ate_result = predict_single_text(ate_model, ate_tokenizer, text, ate_config)
            aspects = [a[0] for a in ate_result['aspects'] if a[0] and len(a[0].strip()) > 1]
            aspect_sentiments = []
            max_length = config['model'].get('max_length', 128)
            for aspect in aspects:
                input_text = f"Aspect: {aspect}. Sentence: {cleaned_text}"
                encoding = tokenizer(
                    input_text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                if 'token_type_ids' in encoding:
                    encoding.pop('token_type_ids')
                encoding = {k: v.to(device) for k, v in encoding.items()}
                outputs = model(**encoding)
                pred = outputs['predictions'].item() if hasattr(outputs['predictions'], 'item') else int(outputs['predictions'][0])
                sentiment_map = config.get('labels', ['negative', 'neutral', 'positive'])
                sentiment = sentiment_map[pred] if pred < len(sentiment_map) else str(pred)
                aspect_sentiments.append((aspect, sentiment))
            result = {
                'text': text,
                'processed_text': cleaned_text,
                'aspects': aspect_sentiments,
                'confidence': float(np.mean([outputs['logits'].softmax(-1).max().item() for _ in aspect_sentiments])) if aspect_sentiments else 1.0,
                'prediction_time': datetime.now().isoformat()
            }
        elif task == 'absa_end_to_end':
            # End-to-end: extract aspects and classify sentiment
            max_length = config['model'].get('max_length', 128)
            encoding = tokenizer(
                cleaned_text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            if 'token_type_ids' in encoding:
                encoding.pop('token_type_ids')
            encoding = {k: v.to(device) for k, v in encoding.items()}
            outputs = model(**encoding)
            # Robustly handle model outputs
            aspect_logits = outputs.get('aspect_logits', None)
            sentiment_logits = outputs.get('sentiment_logits', None)
            if aspect_logits is None or sentiment_logits is None:
                raise ValueError("Model output missing 'aspect_logits' or 'sentiment_logits'. Check model and config.")
            aspect_probs = torch.sigmoid(aspect_logits)
            aspect_threshold = config.get('aspect_threshold', 0.5)
            # Get indices of aspects above threshold (batch dim = 1)
            aspect_indices = (aspect_probs[0] > aspect_threshold).nonzero(as_tuple=True)[0].tolist()
            aspects = []
            aspect_map = config.get('aspect_categories', {})
            if isinstance(aspect_map, dict):
                aspect_names = [k for k, v in sorted(aspect_map.items(), key=lambda x: x[1])]
            else:
                aspect_names = list(aspect_map) if aspect_map else [str(i) for i in range(aspect_probs.shape[1])]
            sentiment_map = config.get('sentiments', ['negative', 'neutral', 'positive'])
            for idx in aspect_indices:
                if idx < len(aspect_names):
                    aspect_name = aspect_names[idx]
                else:
                    aspect_name = str(idx)
                # Sentiment prediction for this aspect
                if sentiment_logits.shape[1] > idx:
                    sentiment_idx = torch.argmax(sentiment_logits[0, idx]).item()
                    sentiment = sentiment_map[sentiment_idx] if sentiment_idx < len(sentiment_map) else str(sentiment_idx)
                else:
                    sentiment = 'unknown'
                aspects.append((aspect_name, sentiment))
            result = {
                'text': text,
                'processed_text': cleaned_text,
                'aspects': aspects,
                'confidence': float(aspect_probs.max().item()),
                'prediction_time': datetime.now().isoformat()
            }
        else:
            raise ValueError(f"Unsupported task: {task}")
    return result

def predict_batch(model, tokenizer, texts, config):
    """Predict aspects and sentiments for a batch of texts."""
    predictions = []
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"Processing {i+1}/{len(texts)} texts...")
        prediction = predict_single_text(model, tokenizer, text, config)
        predictions.append(prediction)
    return predictions

def format_prediction_output(prediction):
    """Format prediction output for display."""
    text = prediction['text']
    aspects = prediction['aspects']
    confidence = prediction['confidence']
    
    output = f"Text: \"{text}\"\n"
    output += f"Confidence: {confidence:.3f}\n"
    output += "Aspects and Sentiments:\n"
    
    if aspects:
        for aspect, sentiment in aspects:
            output += f"  - {aspect}: {sentiment}\n"
    else:
        output += "  No aspects detected\n"
    
    return output

def save_batch_predictions(predictions, output_path):
    """Save batch predictions to file."""
    output_path = Path(output_path)
    
    if output_path.suffix.lower() == '.csv':
        # Save as CSV
        rows = []
        for pred in predictions:
            for aspect, sentiment in pred['aspects']:
                rows.append({
                    'text': pred['text'],
                    'aspect': aspect,
                    'sentiment': sentiment,
                    'confidence': pred['confidence'],
                    'prediction_time': pred['prediction_time']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
    else:
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"Predictions saved to: {output_path}")

def load_input_file(file_path):
    """Load input texts from file."""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
        # Assume the text column is named 'text' or use the first column
        text_column = 'text' if 'text' in df.columns else df.columns[0]
        return df[text_column].tolist()
    
    elif file_path.suffix.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    elif file_path.suffix.lower() == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'texts' in data:
                return data['texts']
            else:
                raise ValueError("JSON file must contain a list of texts or a dict with 'texts' key")
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def main():
    parser = argparse.ArgumentParser(description='Predict aspects and sentiments using trained ABSA model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ate_model_path', type=str, help='Path to trained ATE model (for ATE+ASC pipeline)')
    parser.add_argument('--ate_config', type=str, help='Path to ATE config (for ATE+ASC pipeline)')
    parser.add_argument('--text', type=str, help='Single text to analyze')
    parser.add_argument('--input_file', type=str, help='File containing texts to analyze')
    parser.add_argument('--output_file', type=str, help='Output file for batch predictions')
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='json', 
                       help='Output format for batch predictions')
    args = parser.parse_args()

    # Validate arguments
    if not args.text and not args.input_file:
        parser.error("Either --text or --input_file must be provided")
    if args.input_file and not args.output_file:
        parser.error("--output_file must be provided when using --input_file")

    # Load configuration
    config = load_config(args.config)
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path, config, device)

    # Optionally load ATE model for ATE+ASC pipeline
    ate_model = None
    ate_tokenizer = None
    ate_config = None
    if config['model'].get('task', 'token_classification') == 'sequence_classification':
        if args.ate_model_path and args.ate_config:
            ate_config = load_config(args.ate_config)
            ate_model, ate_tokenizer = load_model(args.ate_model_path, ate_config, device)
        else:
            print("[WARNING] For ASC models, provide --ate_model_path and --ate_config for ATE+ASC pipeline.")

    if args.text:
        print("\nAnalyzing single text...")
        prediction = predict_single_text(model, tokenizer, args.text, config, ate_model=ate_model, ate_tokenizer=ate_tokenizer, ate_config=ate_config)
        print("\nPrediction Results:")
        print("=" * 50)
        print(format_prediction_output(prediction))
    else:
        print(f"\nLoading texts from: {args.input_file}")
        texts = load_input_file(args.input_file)
        print(f"Loaded {len(texts)} texts for analysis")
        print("\nStarting batch prediction...")
        predictions = [predict_single_text(model, tokenizer, text, config, ate_model=ate_model, ate_tokenizer=ate_tokenizer, ate_config=ate_config) for text in texts]
        # Determine output file extension
        if args.output_file:
            output_path = args.output_file
        else:
            output_path = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}"
        save_batch_predictions(predictions, output_path)
        total_aspects = sum(len(pred['aspects']) for pred in predictions)
        avg_confidence = np.mean([pred['confidence'] for pred in predictions])
        print(f"\nBatch prediction completed!")
        print(f"Processed: {len(predictions)} texts")
        print(f"Total aspects detected: {total_aspects}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
