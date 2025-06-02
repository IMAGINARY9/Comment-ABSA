#!/usr/bin/env python3
"""
Prediction script for ABSA models.

This script performs inference using trained ABSA models on new text data,
supporting both single text prediction and batch processing.
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

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(model_path, config, device):
    """Load trained ABSA model."""
    print(f"Loading model from: {model_path}")
    
    # Implementation would depend on actual model structure
    # model = torch.load(model_path, map_location=device)
    # model.eval()
    
    # For now, return a placeholder
    class DummyModel:
        def predict(self, text):
            # Dummy prediction for demonstration
            return {
                'aspects': [('food', 'positive'), ('service', 'negative')],
                'confidence': 0.85
            }
    
    return DummyModel()

def preprocess_text(text, config):
    """Preprocess input text for ABSA prediction."""
    # Basic preprocessing (would be expanded based on config)
    text = text.strip()
    text = text.lower() if config.get('preprocessing', {}).get('lowercase', False) else text
    return text

def predict_single_text(model, text, config):
    """Predict aspects and sentiments for a single text."""
    # Preprocess text
    processed_text = preprocess_text(text, config)
    
    # Make prediction
    prediction = model.predict(processed_text)
    
    return {
        'text': text,
        'processed_text': processed_text,
        'aspects': prediction.get('aspects', []),
        'confidence': prediction.get('confidence', 0.0),
        'prediction_time': datetime.now().isoformat()
    }

def predict_batch(model, texts, config):
    """Predict aspects and sentiments for a batch of texts."""
    predictions = []
    
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"Processing {i+1}/{len(texts)} texts...")
        
        prediction = predict_single_text(model, text, config)
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

    # Load model
    model = load_model(args.model_path, config, device)
    
    if args.text:
        # Single text prediction
        print("\\n🔍 Analyzing single text...")
        prediction = predict_single_text(model, args.text, config)
        
        print("\\n📊 Prediction Results:")
        print("=" * 50)
        print(format_prediction_output(prediction))
    
    else:
        # Batch prediction
        print(f"\\n🔍 Loading texts from: {args.input_file}")
        texts = load_input_file(args.input_file)
        print(f"Loaded {len(texts)} texts for analysis")
        
        print("\\n📊 Starting batch prediction...")
        predictions = predict_batch(model, texts, config)
        
        # Determine output file extension
        if args.output_file:
            output_path = args.output_file
        else:
            output_path = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}"
        
        # Save results
        save_batch_predictions(predictions, output_path)
        
        # Print summary
        total_aspects = sum(len(pred['aspects']) for pred in predictions)
        avg_confidence = np.mean([pred['confidence'] for pred in predictions])
        
        print(f"\\n✅ Batch prediction completed!")
        print(f"📝 Processed: {len(predictions)} texts")
        print(f"🎯 Total aspects detected: {total_aspects}")
        print(f"📈 Average confidence: {avg_confidence:.3f}")
        print(f"💾 Results saved to: {output_path}")

if __name__ == "__main__":
    main()
