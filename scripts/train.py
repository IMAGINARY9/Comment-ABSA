#!/usr/bin/env python3
"""
Comprehensive training script for Aspect-Based Sentiment Analysis (ABSA).
Supports ATE (Aspect Term Extraction), ASC (Aspect Sentiment Classification),
and end-to-end ABSA training.

usage:
    python train.py --config config.yaml --data_dir data/absa --output_dir outputs/absa
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import datetime

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
from preprocessing import ABSAPreprocessor, collate_end2end_absa_batch
from training import ATETrainer, ASCTrainer, MultiTaskABSATrainer
from evaluation import ABSAEvaluator
from data_utils import load_data_from_config
from absa_trainer import ABSATrainerUtils


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
    
    # Map config['model']['task'] to ABSAPreprocessor's 'task' argument
    task_map = {
        "token_classification": "ate",
        "sequence_classification": "asc",
        "absa_end_to_end": "end_to_end"
    }
    preprocessor = ABSAPreprocessor(
        task=task_map.get(config['model']['task'], "ate"),
        tokenizer_name=config['model']['name']
    )
    
    # Load and preprocess data based on task
    task = config['model']['task']
    domain = None
    # Patch: infer domain for end-to-end multitask
    if task == "absa_end_to_end":
        # Try config['data']['dataset'] or infer from data_dir
        if 'dataset' in config.get('data', {}):
            domain = config['data']['dataset']
        else:
            # Use last part of data_dir if it matches a known domain
            domain_candidate = os.path.basename(os.path.normpath(data_dir))
            if domain_candidate in ["laptops", "restaurants", "tweets"]:
                domain = domain_candidate
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
        # End-to-end ABSA (now with domain support)
        train_data, val_data, test_data = preprocessor.load_end_to_end_data(
            data_dir=data_dir,
            train_size=config['data']['train_size'],
            val_size=config['data']['val_size'],
            test_size=config['data']['test_size'],
            domain=domain
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
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_end2end_absa_batch if task == "absa_end_to_end" else None
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_end2end_absa_batch if task == "absa_end_to_end" else None
    )
    
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_end2end_absa_batch if task == "absa_end_to_end" else None
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
        model = DeBERTaATE(config)
    elif task == "sequence_classification":
        # ASC model
        model = DeBERTaASC(config)
    elif task == "absa_end_to_end":
        if "bert" in model_name.lower():
            model = BERTForABSA(config)
        else:
            model = EndToEndABSA(config)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    model = model.to(device)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model created - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    return model


def create_trainer(config: Dict[str, Any], model: nn.Module, use_wandb: bool = False):
    """Create the appropriate trainer based on task."""
    task = config['model']['task']
    if task == "token_classification":
        trainer = ATETrainer(model, config, use_wandb=use_wandb)
    elif task == "sequence_classification":
        trainer = ASCTrainer(model, config, use_wandb=use_wandb)
    elif task == "absa_end_to_end":
        trainer = MultiTaskABSATrainer(model, config, use_wandb=use_wandb)
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
        train_dataloader=train_loader,
        val_dataloader=val_loader
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

    task_map = {
        "token_classification": "ate",
        "sequence_classification": "asc",
        "absa_end_to_end": "end_to_end"
    }
    evaluator_task_key = config['model']['task']
    evaluator_task = task_map.get(evaluator_task_key, evaluator_task_key)

    evaluator = ABSAEvaluator(
        task=evaluator_task,  # Use the mapped task
        label_names=list(config.get('labels', {}).keys()) if 'labels' in config and config.get('labels') else None
    )
    
    # Evaluate the model
    task = config['model']['task']
    if task == "token_classification":
        results = evaluator.evaluate_ate(
            model=model,
            dataloader=test_loader,
            tokenizer=tokenizer
        )
    elif task == "sequence_classification":
        results = evaluator.evaluate_asc(
            model=model,
            dataloader=test_loader
        )
    elif task == "absa_end_to_end":
        results = evaluator.evaluate_end_to_end(
            predictor=model,  # or ABSAPredictor if needed
            test_texts=None,  # fill as needed
            true_aspects=None,
            true_sentiments=None
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Generate comprehensive evaluation report
    evaluation_report = evaluator.generate_evaluation_report(
        results=results
    )
    logging.info("Evaluation Report:\n" + evaluation_report)
    # Optionally, parse metrics from results for further use
    evaluation_metrics = results.get('metrics', {}) if isinstance(results, dict) else {}
    return {'report': evaluation_report, 'metrics': evaluation_metrics}


def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ABSA models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--wandb_project", type=str, default="absa-training", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, help="Wandb run name")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging (default: off)")
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
    
    # Add timestamp to output_dir if not already present
    timestamp = get_timestamp()
    # Compose a descriptive model_dir name: task_dataset_timestamp
    task = None
    dataset = None
    if '--config' in sys.argv:
        # Try to extract dataset/task from config if possible
        # (config is loaded below, so we will update model_dir after loading config)
        pass
    if not args.output_dir.endswith(timestamp):
        output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load configuration
    config = load_config(args.config)
    logging.info(f"Loaded config from {args.config}")

    # Setup device
    device = setup_device()

    # Compose model_dir with task, dataset, and timestamp
    task = config['model'].get('task', 'absa')
    dataset = config.get('data', {}).get('dataset', os.path.basename(os.path.normpath(args.data_dir)))
    # Shorten task for folder name
    task_short = {'token_classification': 'ate', 'sequence_classification': 'asc', 'absa_end_to_end': 'end2end'}.get(task, task)
    model_dir_name = f"{task_short}_{dataset}_{timestamp}"
    model_dir = os.path.join('models', model_dir_name)
    os.makedirs(model_dir, exist_ok=True)

    # Update model_dir in config to use unique, descriptive path
    if 'paths' not in config:
        config['paths'] = {}
    config['paths']['model_dir'] = model_dir

    # Save config to output directory
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize wandb
    use_wandb = args.wandb
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
        trainer = create_trainer(config, model, use_wandb)
        
        # Train model
        training_history = train_model(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=output_dir,
            use_wandb=use_wandb
        )
        
        # Load best model for evaluation
        best_model_path = os.path.join(config['paths']['model_dir'], "best_model.pt")
        if os.path.exists(best_model_path):
            logging.info("Loading best model for evaluation...")
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
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
        
        results_path = os.path.join(output_dir, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logging.info(f"Training completed successfully! Results saved to {output_dir}")
        
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
