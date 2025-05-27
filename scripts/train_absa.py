"""
Training script for ABSA models (ATE and ASC).

Usage:
    python train_ate.py --config configs/deberta_ate.yaml
    python train_asc.py --config configs/deberta_asc.yaml
    python train_end_to_end.py --config configs/bert_end_to_end.yaml
"""

import argparse
import yaml
import torch
import wandb
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing import ABSAPreprocessor
from models import DeBERTaATE, DeBERTaASC, EndToEndABSA
from training import ATETrainer, ASCTrainer, ABSATrainer
from evaluation import ABSAEvaluator

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_ate(config, args):
    """Train Aspect Term Extraction model."""
    print("Training ATE model...")
    
    # Initialize components
    preprocessor = ABSAPreprocessor(task="ate")
    model = DeBERTaATE(config)
    trainer = ATETrainer(model, config, use_wandb=args.wandb)
    evaluator = ABSAEvaluator(task="ate")
    
    # Train and evaluate
    trainer.train()
    results = evaluator.evaluate(model, test_data)
    
    # Save model
    model_path = Path(config['paths']['model_dir']) / "ate_best.pt"
    model.save_pretrained(model_path)
    print(f"ATE model saved to {model_path}")
    
    return results

def train_asc(config, args):
    """Train Aspect Sentiment Classification model."""
    print("Training ASC model...")
    
    # Initialize components
    preprocessor = ABSAPreprocessor(task="asc")
    model = DeBERTaASC(config)
    trainer = ASCTrainer(model, config, use_wandb=args.wandb)
    evaluator = ABSAEvaluator(task="asc")
    
    # Train and evaluate
    trainer.train()
    results = evaluator.evaluate(model, test_data)
    
    # Save model
    model_path = Path(config['paths']['model_dir']) / "asc_best.pt"
    model.save_pretrained(model_path)
    print(f"ASC model saved to {model_path}")
    
    return results

def train_end_to_end(config, args):
    """Train end-to-end ABSA model."""
    print("Training end-to-end ABSA model...")
    
    # Initialize components
    preprocessor = ABSAPreprocessor(task="end_to_end")
    model = EndToEndABSA(config)
    trainer = ABSATrainer(model, config, use_wandb=args.wandb)
    evaluator = ABSAEvaluator(task="end_to_end")
    
    # Train and evaluate
    trainer.train()
    results = evaluator.evaluate(model, test_data)
    
    # Save model
    model_path = Path(config['paths']['model_dir']) / "end_to_end_best.pt"
    model.save_pretrained(model_path)
    print(f"End-to-end model saved to {model_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train ABSA models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--task', type=str, choices=['ate', 'asc', 'end_to_end'], 
                       required=True, help='ABSA task to train')
    parser.add_argument('--data', type=str, default=None, help='Data directory override')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.data:
        config['paths']['data_dir'] = args.data
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project="comment-absa",
            config=config,
            name=f"absa_{args.task}_{config['model']['name'].replace('/', '_')}"
        )
    
    print(f"Training ABSA model with config: {args.config}")
    print(f"Task: {args.task}")
    print(f"Model: {config['model']['name']}")
    
    # Train based on task
    if args.task == 'ate':
        results = train_ate(config, args)
    elif args.task == 'asc':
        results = train_asc(config, args)
    elif args.task == 'end_to_end':
        results = train_end_to_end(config, args)
    
    print(f"Training complete. Results: {results}")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
