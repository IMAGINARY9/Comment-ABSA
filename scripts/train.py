#!/usr/bin/env python3
"""
Comprehensive training script for Aspect-Based Sentiment Analysis (ABSA).
Supports ATE (Aspect Term Extraction), ASC (Aspect Sentiment Classification),
and end-to-end ABSA training.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import wandb
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed

# Add src to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))

from models import (
    DeBERTaATE, DeBERTaASC, BERTForABSA, EndToEndABSA, 
    BiLSTMCRF, ABSAPredictor
)
from preprocessing import ABSAPreprocessor
from training import ATETrainer, ASCTrainer, MultiTaskABSATrainer
from evaluation import ABSAEvaluator


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log")
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_device() -> torch.device:
    """Setup and return the appropriate device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    return device


def prepare_data(config: Dict[str, Any], data_dir: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare data loaders for training, validation, and testing."""
    logging.info("Preparing data...")
    
    preprocessor = ABSAPreprocessor(
        model_name=config['model']['name'],
        max_length=config['model']['max_length'],
        task=config['model']['task']
    )
    
    # Load and preprocess data based on task
    task = config['model']['task']
    
    if task == "token_classification":
        # ATE task - load BIO tagged data
        train_data, val_data, test_data = preprocessor.load_ate_data(
            data_dir=data_dir,
            train_size=config['data']['train_size'],
            val_size=config['data']['val_size'],
            test_size=config['data']['test_size']
        )
    elif task == "sequence_classification":
        # ASC task - load aspect-sentiment pairs
        train_data, val_data, test_data = preprocessor.load_asc_data(
            data_dir=data_dir,
            train_size=config['data']['train_size'],
            val_size=config['data']['val_size'],
            test_size=config['data']['test_size']
        )
    elif task == "absa_end_to_end":
        # End-to-end ABSA
        train_data, val_data, test_data = preprocessor.load_end_to_end_data(
            data_dir=data_dir,
            train_size=config['data']['train_size'],
            val_size=config['data']['val_size'],
            test_size=config['data']['test_size']
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logging.info(f"Data loaded - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_loader, val_loader, test_loader


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and initialize the model based on configuration."""
    model_name = config['model']['name']
    task = config['model']['task']
    
    logging.info(f"Creating {task} model with {model_name}")
    
    if task == "token_classification":
        # ATE model
        model = DeBERTaATE(
            model_name=model_name,
            num_labels=config['model']['num_labels'],
            dropout=config['model'].get('dropout', 0.1),
            use_lora=config.get('lora', {}).get('use_lora', False),
            lora_config=config.get('lora', {})
        )
    elif task == "sequence_classification":
        # ASC model
        model = DeBERTaASC(
            model_name=model_name,
            num_labels=config['model']['num_labels'],
            dropout=config['model'].get('dropout', 0.1),
            use_lora=config.get('lora', {}).get('use_lora', False),
            lora_config=config.get('lora', {})
        )
    elif task == "absa_end_to_end":
        if "bert" in model_name.lower():
            model = BERTForABSA(
                model_name=model_name,
                num_aspect_labels=config['model']['num_aspect_labels'],
                num_sentiment_labels=config['model']['num_sentiment_labels'],
                dropout=config['model'].get('dropout', 0.1)
            )
        else:
            model = EndToEndABSA(
                model_name=model_name,
                num_aspect_labels=config['model']['num_aspect_labels'],
                num_sentiment_labels=config['model']['num_sentiment_labels'],
                dropout=config['model'].get('dropout', 0.1)
            )
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    model = model.to(device)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model created - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    return model


def create_trainer(config: Dict[str, Any], model: nn.Module, device: torch.device):
    """Create the appropriate trainer based on task."""
    task = config['model']['task']
    
    if task == "token_classification":
        trainer = ATETrainer(
            model=model,
            device=device,
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01),
            warmup_steps=config['training'].get('warmup_steps', 0),
            class_weights=config.get('class_weights', {}),
            gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1)
        )
    elif task == "sequence_classification":
        trainer = ASCTrainer(
            model=model,
            device=device,
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01),
            warmup_steps=config['training'].get('warmup_steps', 0),
            class_weights=config.get('class_weights', {}),
            gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1)
        )
    elif task == "absa_end_to_end":
        trainer = MultiTaskABSATrainer(
            model=model,
            device=device,
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01),
            warmup_steps=config['training'].get('warmup_steps', 0),
            aspect_weight=config.get('multitask', {}).get('aspect_weight', 0.5),
            sentiment_weight=config.get('multitask', {}).get('sentiment_weight', 0.5),
            gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1)
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return trainer


def train_model(
    trainer, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    config: Dict[str, Any],
    output_dir: str,
    use_wandb: bool = True
) -> Dict[str, Any]:
    """Train the model and return training history."""
    logging.info("Starting training...")
    
    # Training parameters
    num_epochs = config['training']['num_epochs']
    save_steps = config['training'].get('save_steps', 500)
    eval_steps = config['training'].get('eval_steps', 250)
    logging_steps = config['training'].get('logging_steps', 50)
    
    # Train the model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_dir=output_dir,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        use_wandb=use_wandb
    )
    
    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logging.info(f"Training completed. Best validation score: {history.get('best_val_score', 'N/A')}")
    
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    config: Dict[str, Any],
    output_dir: str,
    tokenizer: Optional[Any] = None
) -> Dict[str, Any]:
    """Evaluate the trained model."""
    logging.info("Starting evaluation...")
    
    evaluator = ABSAEvaluator(
        task=config['model']['task'],
        label_names=list(config.get('labels', {}).keys()) if 'labels' in config else None,
        output_dir=output_dir
    )
    
    # Evaluate the model
    results = evaluator.evaluate_model(
        model=model,
        test_loader=test_loader,
        tokenizer=tokenizer
    )
    
    # Generate comprehensive evaluation report
    evaluation_results = evaluator.generate_evaluation_report(
        results=results,
        save_plots=True,
        save_errors=True
    )
    
    # Log key metrics
    logging.info("Evaluation Results:")
    for metric, value in evaluation_results.get('metrics', {}).items():
        logging.info(f"  {metric}: {value:.4f}")
    
    return evaluation_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ABSA models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--wandb_project", type=str, default="absa-training", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, help="Wandb run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Debug mode enabled")
    
    # Load configuration
    config = load_config(args.config)
    logging.info(f"Loaded config from {args.config}")
    
    # Setup device
    device = setup_device()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config to output directory
    config_save_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=config,
                dir=args.output_dir
            )
            logging.info("Wandb initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            use_wandb = False
    
    try:
        # Prepare data
        train_loader, val_loader, test_loader = prepare_data(config, args.data_dir)
        
        # Create model
        model = create_model(config, device)
        
        # Create trainer
        trainer = create_trainer(config, model, device)
        
        # Train model
        training_history = train_model(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=args.output_dir,
            use_wandb=use_wandb
        )
        
        # Load best model for evaluation
        best_model_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            logging.info("Loading best model for evaluation...")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        # Create tokenizer for evaluation
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        
        # Evaluate model
        evaluation_results = evaluate_model(
            model=model,
            test_loader=test_loader,
            config=config,
            output_dir=args.output_dir,
            tokenizer=tokenizer
        )
        
        # Log final results
        if use_wandb:
            wandb.log({
                "final_test_score": evaluation_results.get('metrics', {}).get('f1_score', 0),
                "final_test_accuracy": evaluation_results.get('metrics', {}).get('accuracy', 0)
            })
        
        # Save final results summary
        final_results = {
            "config": config,
            "training_history": training_history,
            "evaluation_results": evaluation_results,
            "model_info": {
                "total_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        }
        
        results_path = os.path.join(args.output_dir, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logging.info(f"Training completed successfully! Results saved to {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise
    
    finally:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
