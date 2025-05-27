"""
Script to prepare and organize ABSA data.

This script processes SemEval and other ABSA datasets and prepares them for training.
"""

import pandas as pd
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def parse_semeval_xml(xml_path):
    """Parse SemEval XML format to extract aspects and sentiments."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    data = []
    for sentence in root.findall('.//sentence'):
        text = sentence.find('text').text if sentence.find('text') is not None else ""
        
        # Extract aspects and opinions
        aspects = []
        sentiments = []
        
        # Look for aspectTerms
        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is not None:
            for aspect_term in aspect_terms.findall('aspectTerm'):
                term = aspect_term.get('term', '')
                polarity = aspect_term.get('polarity', 'neutral')
                aspects.append(term)
                sentiments.append(polarity)
        
        # Look for aspectCategories
        aspect_categories = sentence.find('aspectCategories')
        if aspect_categories is not None:
            for aspect_cat in aspect_categories.findall('aspectCategory'):
                category = aspect_cat.get('category', '')
                polarity = aspect_cat.get('polarity', 'neutral')
                aspects.append(category)
                sentiments.append(polarity)
        
        data.append({
            'text': text,
            'aspects': aspects,
            'sentiments': sentiments,
            'num_aspects': len(aspects)
        })
    
    return data

def process_semeval_datasets():
    """Process SemEval datasets from XML to CSV format."""
    data_dir = Path("./data/semeval_datasets")
    
    if not data_dir.exists():
        print("SemEval datasets not found. Please ensure data is in ./data/semeval_datasets/")
        return
    
    # Process all XML files
    for xml_file in data_dir.glob("**/*.xml"):
        print(f"Processing {xml_file}")
        
        try:
            data = parse_semeval_xml(xml_file)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Save as CSV
            csv_path = xml_file.parent / f"{xml_file.stem}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved to {csv_path}: {len(df)} sentences")
            
        except Exception as e:
            print(f"  Error processing {xml_file}: {e}")

def create_bio_format_data():
    """Convert aspect data to BIO format for token classification."""
    data_dir = Path("./data/semeval_datasets")
    
    for csv_file in data_dir.glob("**/*.csv"):
        if "bio_format" in csv_file.name:
            continue
            
        print(f"Converting {csv_file} to BIO format")
        df = pd.read_csv(csv_file)
        
        bio_data = []
        for _, row in df.iterrows():
            text = row['text']
            aspects = eval(row['aspects']) if isinstance(row['aspects'], str) else row['aspects']
            
            if not aspects:
                continue
            
            # Simple tokenization (word-level)
            tokens = text.split()
            labels = ['O'] * len(tokens)
            
            # Mark aspect terms with BIO tags
            for aspect in aspects:
                aspect_tokens = aspect.split()
                for i in range(len(tokens) - len(aspect_tokens) + 1):
                    if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                        labels[i] = 'B-ASP'
                        for j in range(1, len(aspect_tokens)):
                            if i + j < len(labels):
                                labels[i + j] = 'I-ASP'
                        break
            
            bio_data.append({
                'tokens': tokens,
                'labels': labels,
                'text': text
            })
        
        # Save BIO format data
        bio_df = pd.DataFrame(bio_data)
        bio_path = csv_file.parent / f"bio_format_{csv_file.name}"
        bio_df.to_csv(bio_path, index=False)
        print(f"  Saved BIO format to {bio_path}")

def create_aspect_sentiment_pairs():
    """Create aspect-sentiment pairs for ASC training."""
    data_dir = Path("./data/semeval_datasets")
    
    for csv_file in data_dir.glob("**/*.csv"):
        if "bio_format" in csv_file.name or "asc_pairs" in csv_file.name:
            continue
            
        print(f"Creating aspect-sentiment pairs from {csv_file}")
        df = pd.read_csv(csv_file)
        
        pair_data = []
        for _, row in df.iterrows():
            text = row['text']
            aspects = eval(row['aspects']) if isinstance(row['aspects'], str) else row['aspects']
            sentiments = eval(row['sentiments']) if isinstance(row['sentiments'], str) else row['sentiments']
            
            if not aspects or not sentiments:
                continue
            
            # Create pairs
            for aspect, sentiment in zip(aspects, sentiments):
                # Format: "Aspect: {aspect}. Sentence: {text}"
                formatted_input = f"Aspect: {aspect}. Sentence: {text}"
                pair_data.append({
                    'input_text': formatted_input,
                    'aspect': aspect,
                    'text': text,
                    'sentiment': sentiment
                })
        
        # Save aspect-sentiment pairs
        pairs_df = pd.DataFrame(pair_data)
        pairs_path = csv_file.parent / f"asc_pairs_{csv_file.name}"
        pairs_df.to_csv(pairs_path, index=False)
        print(f"  Saved aspect-sentiment pairs to {pairs_path}: {len(pairs_df)} pairs")

def create_train_val_test_splits():
    """Create train/validation/test splits for all processed datasets."""
    data_dir = Path("./data")
    
    for dataset_dir in data_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        print(f"Creating splits for {dataset_dir.name}")
        
        for csv_file in dataset_dir.glob("*.csv"):
            if any(split in csv_file.name for split in ['train', 'val', 'test']):
                continue
                
            df = pd.read_csv(csv_file)
            
            # Shuffle data
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Split data
            train_size = int(0.8 * len(df))
            val_size = int(0.1 * len(df))
            
            train_df = df[:train_size]
            val_df = df[train_size:train_size + val_size]
            test_df = df[train_size + val_size:]
            
            # Save splits
            base_name = csv_file.stem
            train_df.to_csv(dataset_dir / f"{base_name}_train.csv", index=False)
            val_df.to_csv(dataset_dir / f"{base_name}_val.csv", index=False)
            test_df.to_csv(dataset_dir / f"{base_name}_test.csv", index=False)
            
            print(f"  Split {csv_file.name}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

if __name__ == "__main__":
    print("Preparing ABSA data...")
    
    # Create output directories
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./models/ate", exist_ok=True)
    os.makedirs("./models/asc", exist_ok=True)
    os.makedirs("./models/end_to_end", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Process SemEval datasets
    print("Processing SemEval XML datasets...")
    process_semeval_datasets()
    
    # Create BIO format data for ATE
    print("Creating BIO format data for ATE...")
    create_bio_format_data()
    
    # Create aspect-sentiment pairs for ASC
    print("Creating aspect-sentiment pairs for ASC...")
    create_aspect_sentiment_pairs()
    
    # Create splits
    print("Creating train/val/test splits...")
    create_train_val_test_splits()
    
    print("ABSA data preparation complete!")
