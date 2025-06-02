"""
Data loading utilities for ABSA project.
"""
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def load_semeval_data(data_path: str, dataset_type: str = "train") -> Tuple[List[str], List[List[Dict]]]:
    """
    Load SemEval ABSA data from XML files.
    
    Args:
        data_path: Path to the SemEval data directory
        dataset_type: Type of dataset ("train", "test", "dev")
    
    Returns:
        Tuple of (texts, aspect_data) where aspect_data contains aspect terms and sentiments
    """
    xml_file = Path(data_path) / f"{dataset_type}.xml"
    
    if not xml_file.exists():
        raise FileNotFoundError(f"Data file not found: {xml_file}")
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    texts = []
    aspect_data = []
    
    for sentence in root.findall('.//sentence'):
        text = sentence.find('text').text if sentence.find('text') is not None else ""
        texts.append(text)
        
        aspects = []
        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is not None:
            for aspect_term in aspect_terms.findall('aspectTerm'):
                aspect_info = {
                    'term': aspect_term.get('term', ''),
                    'polarity': aspect_term.get('polarity', 'neutral'),
                    'from': int(aspect_term.get('from', 0)),
                    'to': int(aspect_term.get('to', 0))
                }
                aspects.append(aspect_info)
        
        aspect_data.append(aspects)
    
    return texts, aspect_data

def load_data_from_config(config: Dict, dataset_override: Optional[str] = None, 
                         data_dir_override: Optional[str] = None) -> Tuple[List[str], List[List[Dict]]]:
    """
    Load ABSA data based on configuration.
    
    Args:
        config: Configuration dictionary
        dataset_override: Override for dataset selection
        data_dir_override: Override for data directory
    
    Returns:
        Tuple of (texts, aspect_data)
    """
    data_dir = Path(data_dir_override or config['data'].get('data_dir', 'data/semeval_datasets'))
    dataset = dataset_override or config['data'].get('dataset', 'semeval2014')
    
    # Map dataset names to directories
    dataset_map = {
        'semeval2014': 'SemEval-2014-Task4',
        'semeval2015': 'SemEval-2015-Task12',
        'semeval2016': 'SemEval-2016-Task5'
    }
    
    dataset_dir = data_dir / dataset_map.get(dataset, dataset)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Check for preprocessed data first
    preprocessed_dir = Path('data/preprocessed')
    preprocessed_file = preprocessed_dir / f"{dataset}_processed.csv"
    
    if preprocessed_file.exists():
        print(f"Using preprocessed data: {preprocessed_file}")
        return load_preprocessed_data(preprocessed_file)
    else:
        print(f"Loading raw data from: {dataset_dir}")
        return load_semeval_data(dataset_dir, "train")

def load_preprocessed_data(file_path: Path) -> Tuple[List[str], List[List[Dict]]]:
    """
    Load preprocessed ABSA data from CSV.
    
    Args:
        file_path: Path to preprocessed CSV file
    
    Returns:
        Tuple of (texts, aspect_data)
    """
    df = pd.read_csv(file_path)
    
    texts = df['text'].astype(str).tolist()
    
    # Parse aspect data from JSON strings
    aspect_data = []
    for aspects_str in df['aspects']:
        if pd.isna(aspects_str) or aspects_str == '[]':
            aspect_data.append([])
        else:
            try:
                import json
                aspects = json.loads(aspects_str)
                aspect_data.append(aspects)
            except:
                aspect_data.append([])
    
    return texts, aspect_data

def prepare_ate_data(texts: List[str], aspect_data: List[List[Dict]]) -> Tuple[List[str], List[List[str]]]:
    """
    Prepare data for Aspect Term Extraction (ATE) task.
    
    Args:
        texts: List of input texts
        aspect_data: List of aspect information per text
    
    Returns:
        Tuple of (texts, bio_tags) for sequence labeling
    """
    bio_texts = []
    bio_tags = []
    
    for text, aspects in zip(texts, aspect_data):
        tokens = text.split()  # Simple tokenization
        tags = ['O'] * len(tokens)
        
        # Convert aspect spans to BIO tags
        for aspect in aspects:
            term = aspect['term']
            start_pos = aspect.get('from', 0)
            end_pos = aspect.get('to', 0)
            
            # Find token positions (simplified approach)
            term_tokens = term.split()
            if term_tokens:
                # Mark first token as B- and rest as I-
                for i, token in enumerate(tokens):
                    if token in term_tokens:
                        if i == 0 or tags[i-1] == 'O':
                            tags[i] = 'B-ASPECT'
                        else:
                            tags[i] = 'I-ASPECT'
        
        bio_texts.append(tokens)
        bio_tags.append(tags)
    
    return bio_texts, bio_tags

def prepare_asc_data(texts: List[str], aspect_data: List[List[Dict]]) -> Tuple[List[str], List[str]]:
    """
    Prepare data for Aspect Sentiment Classification (ASC) task.
    
    Args:
        texts: List of input texts
        aspect_data: List of aspect information per text
    
    Returns:
        Tuple of (aspect_texts, sentiments) for classification
    """
    aspect_texts = []
    sentiments = []
    
    for text, aspects in zip(texts, aspect_data):
        for aspect in aspects:
            # Create text with aspect term highlighted
            aspect_text = f"{text} [SEP] {aspect['term']}"
            aspect_texts.append(aspect_text)
            sentiments.append(aspect['polarity'])
    
    return aspect_texts, sentiments
