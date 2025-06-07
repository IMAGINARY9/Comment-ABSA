#!/usr/bin/env python3
"""
Test script for NER integration in Comment-ABSA project.
This script tests the end-to-end pipeline with NER features.
"""

import sys
import os
from pathlib import Path
import logging
import yaml
import torch
from torch.utils.data import DataLoader

# Add src to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "src"))

# Import required modules
sys.path.append('src')
from preprocessing import ABSAPreprocessor
from models import DeBERTaATE
from training import ABSADataset, filter_model_inputs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ner_integration():
    """Test the NER integration pipeline."""
    logger.info("Starting NER integration test...")
    
    # Load NER-enabled configuration
    config_path = "configs/deberta_ate_with_ner.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found!")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config: {config['model']['name']}")
      # Test 1: Initialize NER-enhanced preprocessor
    logger.info("Test 1: Initializing NER-enhanced preprocessor...")
    try:
        model_config = config.get('model', {})
        preprocessor = ABSAPreprocessor(
            task='token_classification',
            tokenizer_name=config['model']['name'],
            ner_model_path=model_config.get('ner_model_path'),
            ner_word_tokenizer_path=model_config.get('ner_word_tokenizer_path'), 
            ner_tag_vocab_path=model_config.get('ner_tag_vocab_path'),
            ner_max_seq_length=model_config.get('ner_max_seq_length', 128)
        )
        logger.info("✓ NER-enhanced preprocessor initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize preprocessor: {e}")
        return False
    
    # Test 2: Load and prepare data with NER features
    logger.info("Test 2: Loading and preparing data with NER features...")
    try:
        data_dir = "data/splits/laptops"
        if not os.path.exists(data_dir):
            logger.error(f"Data directory {data_dir} not found!")
            return False
            
        # Load a small subset for testing
        train_data, val_data, test_data = preprocessor.load_ate_data(
            data_dir=data_dir,
            train_size=10,  # Small subset for testing
            val_size=5,
            test_size=5
        )
        
        logger.info(f"✓ Data loaded: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        # Check if NER tags are present in the data
        if len(train_data) > 0:
            sample = train_data[0]
            if 'ner_tags' in sample:
                logger.info(f"✓ NER tags found in data: {len(sample['ner_tags'])} tags")
            else:
                logger.warning("⚠ NER tags not found in sample data")
                
    except Exception as e:
        logger.error(f"✗ Failed to load data: {e}")
        return False
    
    # Test 3: Initialize NER-enhanced model
    logger.info("Test 3: Initializing NER-enhanced model...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        logger.info(f"Using device: {device}")
        
        model = DeBERTaATE(config).to(device)
        
        logger.info(f"✓ NER-enhanced model initialized successfully")
        logger.info(f"  - Use NER features: {config['model'].get('use_ner_features', False)}")
        logger.info(f"  - NER embedding dim: {config['model'].get('ner_embedding_dim', 64)}")
        logger.info(f"  - Combine method: {config['model'].get('combine_ner_method', 'concatenate')}")
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize model: {e}")
        return False
      # Test 4: Create dataset and test forward pass
    logger.info("Test 4: Testing forward pass with NER features...")
    try:
        # Convert loaded data to the expected format for ABSADataset
        from torch.nn.utils.rnn import pad_sequence
        
        # Extract individual components from the loaded data
        input_ids = []
        attention_masks = []
        labels = []
        ner_tags = []
        
        for item in train_data:
            input_ids.append(torch.tensor(item['input_ids'], dtype=torch.long))
            attention_masks.append(torch.tensor(item['attention_mask'], dtype=torch.long))
            labels.append(torch.tensor(item['labels'], dtype=torch.long))
            if 'ner_tags' in item:
                ner_tags.append(torch.tensor(item['ner_tags'], dtype=torch.long))
        
        # Pad sequences to the same length
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        if ner_tags:
            ner_tags = pad_sequence(ner_tags, batch_first=True, padding_value=0)
          # Create the data dictionary expected by ABSADataset
        prepared_data = {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }
        
        if len(ner_tags) > 0:
            prepared_data['ner_tags'] = ner_tags
        
        # Create dataset
        dataset = ABSADataset(prepared_data, task="ate")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Filter batch inputs to match model requirements
                batch = filter_model_inputs(batch)
                
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Debug: Print batch information
                logger.info(f"  - Batch keys: {list(batch.keys())}")
                logger.info(f"  - Input shape: {batch['input_ids'].shape}")
                if 'ner_tags' in batch:
                    logger.info(f"  - NER tags shape: {batch['ner_tags'].shape}")
                    logger.info(f"  - NER tags present: True")
                    # Debug: Check NER tag values
                    unique_tags = torch.unique(batch['ner_tags'])
                    logger.info(f"  - Unique NER tag values: {unique_tags}")
                    max_tag = torch.max(batch['ner_tags'])
                    min_tag = torch.min(batch['ner_tags'])
                    logger.info(f"  - NER tag range: [{min_tag}, {max_tag}]")
                else:
                    logger.info(f"  - NER tags present: False")
                  # Forward pass
                outputs = model(**batch)
                
                logger.info(f"✓ Forward pass successful")
                logger.info(f"  - Batch keys: {list(batch.keys())}")
                logger.info(f"  - Output keys: {list(outputs.keys())}")
                if 'logits' in outputs:
                    logger.info(f"  - Output shape: {outputs['logits'].shape}")
                else:
                    logger.info(f"  - Available output keys: {list(outputs.keys())}")
                
                break  # Test with just one batch
                
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    logger.info("🎉 All NER integration tests passed successfully!")
    return True

def main():
    """Main test function."""
    success = test_ner_integration()
    if success:
        logger.info("✅ NER integration test completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ NER integration test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
