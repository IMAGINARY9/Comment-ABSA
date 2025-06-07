"""
ABSA Preprocessing Module

This module provides preprocessing utilities for Aspect-Based Sentiment Analysis:
- Text cleaning and normalization
- BIO tagging for aspect term extraction
- Data formatting for ASC (Aspect Sentiment Classification)
- Dataset preparation and augmentation
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import torch
import nltk
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Import the NER utility
from ner_utils import CommentABSANERExtractor  # Assuming ner_utils.py is in the same directory

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ABSAPreprocessor:
    """
    Comprehensive preprocessor for ABSA tasks.
    
    Handles text cleaning, tokenization, and format preparation
    for both ATE (token classification) and ASC (sequence classification).
    """
    
    def __init__(self, task: str = "ate", tokenizer_name: str = "microsoft/deberta-v3-base", ner_model_path: Optional[str] = None, ner_word_tokenizer_path: Optional[str] = None, ner_tag_vocab_path: Optional[str] = None, ner_max_seq_length: int = 128):
        """
        Initialize preprocessor.
        
        Args:
            task: Task type (\'ate\', \'asc\', \'end_to_end\')
            tokenizer_name: Name of the tokenizer to use
            ner_model_path: Path to the trained NER Keras model.
            ner_word_tokenizer_path: Path to the Keras Tokenizer JSON for NER.
            ner_tag_vocab_path: Path to the JSON file for NER tag vocabulary.
            ner_max_seq_length: Max sequence length for the NER model.
        """
        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.logger = logging.getLogger(__name__)
          # Initialize NER Extractor if paths are provided
        self.ner_extractor = None
        self.ner_tag_vocab = None
        if ner_model_path and ner_word_tokenizer_path and ner_tag_vocab_path:
            try:
                self.ner_extractor = CommentABSANERExtractor(
                    model_path=ner_model_path,
                    word_tokenizer_path=ner_word_tokenizer_path,
                    tag_vocab_path=ner_tag_vocab_path,
                    ner_max_seq_length=ner_max_seq_length
                )
                # Load NER tag vocabulary for ID mapping
                import json
                with open(ner_tag_vocab_path, 'r', encoding='utf-8') as f:
                    self.ner_tag_vocab = json.load(f)
                self.logger.info("NER Extractor initialized successfully.")
            except Exception as e:
                self.logger.error(f"Failed to initialize NER Extractor: {e}")
        else:
            self.logger.info("NER Extractor paths not provided, NER features will not be used.")

        # Load spaCy model for advanced processing
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            self.logger.warning(f"spaCy model not loaded: {e}")
            self.nlp = None
        
        # Stop words
        self.stop_words = set(stopwords.words('english'))
        
        # Common aspect domains for validation
        self.aspect_domains = {
            'restaurant': ['food', 'service', 'ambiance', 'price', 'location', 'staff'],
            'laptop': ['battery', 'screen', 'keyboard', 'performance', 'price', 'design'],
            'hotel': ['room', 'service', 'location', 'price', 'staff', 'amenities']
        }
        
        # Sentiment mappings
        self.sentiment_mappings = {
            'positive': ['positive', 'pos', '1', 1, 2],
            'negative': ['negative', 'neg', '0', 0, 0],
            'neutral': ['neutral', 'neu', 'conflict', '2', 2, 1]
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for ABSA.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text or (hasattr(pd, 'isna') and pd.isna(text)):
            return ""
        text = str(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\'\"]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text)
        return text.strip()
    
    def extract_aspects_from_text(self, text: str, domain: str = None) -> List[str]:
        """
        Extract potential aspects from text using linguistic patterns.
        
        Args:
            text: Input text
            domain: Domain hint (restaurant, laptop, hotel)
            
        Returns:
            List of potential aspect terms
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        aspects = []
        
        # Rule-based aspect extraction patterns
        for token in doc:
            # Nouns that are not stop words
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                aspects.append(token.text)
        
        # Domain-specific filtering
        if domain and domain in self.aspect_domains:
            aspects = [a for a in aspects if a.lower() in self.aspect_domains[domain]]
        
        return list(set(aspects))
    
    def create_bio_tags(self, text: str, aspects: List[str]) -> List[str]:
        """
        Create BIO tags for aspect terms in text.
        
        Args:
            text: Input text
            aspects: List of aspect terms
            
        Returns:
            List of BIO tags
        """
        tokens = word_tokenize(text.lower())
        tags = ['O'] * len(tokens)
        
        for aspect in aspects:
            aspect_tokens = word_tokenize(aspect.lower())
            
            # Find aspect in tokens
            for i in range(len(tokens) - len(aspect_tokens) + 1):
                if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                    # Mark B-ASP for first token, I-ASP for rest
                    tags[i] = 'B-ASP'
                    for j in range(1, len(aspect_tokens)):
                        tags[i+j] = 'I-ASP'
                    break
        
        return tags
    
    def prepare_ate_data(self, texts: List[str], aspects_list: List[List[str]], max_length: int = 128) -> Dict:
        """
        Prepare data for Aspect Term Extraction (token classification).
        
        Args:
            texts: List of input texts
            aspects_list: List of aspect terms for each text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        import torch
        tokenized_inputs = []
        labels = []
        all_ner_tags_for_batch = [] # To store NER tags for the batch

        for text, aspects in zip(texts, aspects_list):
            cleaned = self.clean_text(text)
            
            # Get NER tags if extractor is available
            ner_tokens = []
            ner_tags = []
            if self.ner_extractor:
                try:
                    ner_tokens, ner_tags = self.ner_extractor.predict_ner_tags(cleaned)
                    # Ensure ner_tags align with word_tokenize(cleaned.lower())
                    # This might require careful alignment if tokenization differs significantly
                except Exception as e:
                    self.logger.warning(f"NER prediction failed for text: \'{cleaned[:50]}...\'. Error: {e}")
                    ner_tags = [] # Fallback to no NER tags

            bio_tags = self.create_bio_tags(cleaned, aspects) # These are based on word_tokenize(cleaned.lower())
            tokens = word_tokenize(cleaned.lower()) # ABSA tokenizer's tokens            # Align NER tags with ABSA tokens (this is a crucial and potentially complex step)
            # For simplicity, we'll assume a direct mapping if token counts are similar,
            # otherwise, we might need a more sophisticated alignment or just use 'O' for NER.
            aligned_ner_tags = []
            if ner_tags and len(ner_tokens) == len(tokens): # Simple check
                aligned_ner_tags = ner_tags
            elif ner_tags and len(ner_tokens) > 0: # Attempt simple alignment
                # Try to align by finding the best matching tokens
                aligned_ner_tags = self._align_ner_tags_with_absa_tokens(ner_tokens, ner_tags, tokens)
            else: # Fallback if tokenization mismatch or no NER tags
                aligned_ner_tags = ['O'] * len(tokens)
            
            all_ner_tags_for_batch.append(aligned_ner_tags)

            encoding = self.tokenizer(tokens, is_split_into_words=True, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
            label_ids = []
            # We need a vocabulary for NER tags if we want to convert them to IDs
            # For now, let's assume we'll handle NER tags as a separate input or find a way to map them.
            # This part needs further refinement based on how NER tags will be consumed by the model.
            # Placeholder: ner_tag_ids = [self.ner_tag_to_id.get(tag, 0) for tag in aligned_ner_tags_for_word]

            word_ids = encoding.word_ids(batch_index=0)
            current_ner_tag_ids_for_encoding = [] # For this specific encoding

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    current_ner_tag_ids_for_encoding.append(-100) # Corresponding NER tag padding
                elif word_idx < len(bio_tags): # Ensure word_idx is within bounds for bio_tags
                    tag = bio_tags[word_idx]
                    label_ids.append({'B-ASP': 0, 'I-ASP': 1, 'O': 4}.get(tag, 4)) # Make sure your ABSA model uses these IDs                    # NER tag for the current word_idx
                    # This assumes aligned_ner_tags has one tag per token from word_tokenize(cleaned.lower())
                    if word_idx < len(aligned_ner_tags):
                        ner_tag_for_word = aligned_ner_tags[word_idx]
                        # Convert ner_tag_for_word to an ID using the NER tag vocabulary
                        if self.ner_tag_vocab:
                            current_ner_tag_ids_for_encoding.append(self.ner_tag_vocab.get(ner_tag_for_word, self.ner_tag_vocab.get('O', 2)))
                        else:
                            current_ner_tag_ids_for_encoding.append(2)  # Default to 'O' tag ID
                    else:
                        current_ner_tag_ids_for_encoding.append(-100) # Pad if out of bounds
                else: # word_idx is out of bounds for bio_tags (e.g. special tokens not aligned to words)
                    label_ids.append(-100)
                    current_ner_tag_ids_for_encoding.append(-100)
            
            encoding['labels'] = torch.tensor(label_ids)
            encoding['ner_tags'] = torch.tensor(current_ner_tag_ids_for_encoding) # Add NER tags to the encoding
            tokenized_inputs.append(encoding)
            # labels.append(label_ids) # This 'labels' list seems redundant if using encoding['labels']

        input_ids = torch.cat([x['input_ids'] for x in tokenized_inputs], dim=0)
        attention_mask = torch.cat([x['attention_mask'] for x in tokenized_inputs], dim=0)
        labels_tensor = torch.stack([x['labels'] for x in tokenized_inputs])
        ner_tags_tensor = torch.stack([x['ner_tags'] for x in tokenized_inputs]) # Stack NER tags

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_tensor,
            'ner_tags': ner_tags_tensor # Return NER tags as part of the prepared data
        }
    
    def prepare_asc_data(self, texts: List[str], aspects: List[str], 
                        sentiments: List[str], max_length: int = 256) -> Dict:
        """
        Prepare data for Aspect Sentiment Classification.
        
        Args:
            texts: List of input texts
            aspects: List of aspect terms
            sentiments: List of sentiment labels
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        formatted_inputs = []
        sentiment_labels = []
        
        for text, aspect, sentiment in zip(texts, aspects, sentiments):
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Format input: "Aspect: {aspect}. Sentence: {text}"
            formatted_input = f"Aspect: {aspect}. Sentence: {cleaned_text}"
            formatted_inputs.append(formatted_input)
            
            # Normalize sentiment label
            normalized_sentiment = self.normalize_sentiment(sentiment)
            sentiment_labels.append(normalized_sentiment)
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_inputs,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(sentiment_labels, dtype=torch.long)
        }
    
    def normalize_sentiment(self, sentiment: Union[str, int]) -> int:
        """
        Normalize sentiment label to standard format.
        
        Args:
            sentiment: Sentiment label (various formats)
            
        Returns:
            Normalized sentiment (0: negative, 1: neutral, 2: positive)
        """
        sentiment = str(sentiment).lower().strip()
        
        for normalized, variants in self.sentiment_mappings.items():
            if sentiment in [str(v).lower() for v in variants]:
                if normalized == 'negative':
                    return 0
                elif normalized == 'neutral':
                    return 1
                elif normalized == 'positive':
                    return 2
        
        # Default to neutral if unknown
        return 1
    
    def augment_data(self, texts: List[str], aspects_list: List[List[str]], 
                    sentiments_list: List[List[str]]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """
        Data augmentation for ABSA training.
        
        Args:
            texts: List of input texts
            aspects_list: List of aspects for each text
            sentiments_list: List of sentiments for each text
            
        Returns:
            Augmented texts, aspects, and sentiments
        """
        augmented_texts = texts.copy()
        augmented_aspects = aspects_list.copy()
        augmented_sentiments = sentiments_list.copy()
        
        for text, aspects, sentiments in zip(texts, aspects_list, sentiments_list):
            # Synonym replacement for aspects
            if self.nlp:
                doc = self.nlp(text)
                augmented_text = text
                
                for aspect in aspects:
                    # Simple synonym replacement (can be enhanced)
                    synonyms = {
                        'food': 'meal',
                        'service': 'staff',
                        'price': 'cost',
                        'ambiance': 'atmosphere'
                    }
                    
                    if aspect.lower() in synonyms:
                        augmented_text = augmented_text.replace(
                            aspect, synonyms[aspect.lower()]
                        )
                        
                        # Update aspects list
                        new_aspects = [synonyms[aspect.lower()] if a == aspect else a 
                                     for a in aspects]
                        
                        augmented_texts.append(augmented_text)
                        augmented_aspects.append(new_aspects)
                        augmented_sentiments.append(sentiments)
        
        return augmented_texts, augmented_aspects, augmented_sentiments
    
    def create_aspect_sentiment_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aspect-sentiment pairs from structured ABSA data.
        
        Args:
            df: DataFrame with text, aspects, and sentiments columns
            
        Returns:
            DataFrame with individual aspect-sentiment pairs
        """
        pairs = []
        
        for _, row in df.iterrows():
            text = row['text']
            
            # Handle different formats of aspects and sentiments
            if isinstance(row['aspects'], str):
                try:
                    aspects = eval(row['aspects'])
                except:
                    aspects = [row['aspects']]
            else:
                aspects = row['aspects'] if row['aspects'] else []
            
            if isinstance(row['sentiments'], str):
                try:
                    sentiments = eval(row['sentiments'])
                except:
                    sentiments = [row['sentiments']]
            else:
                sentiments = row['sentiments'] if row['sentiments'] else []
            
            # Create pairs
            for aspect, sentiment in zip(aspects, sentiments):
                if aspect and sentiment:
                    pairs.append({
                        'text': text,
                        'aspect': aspect,
                        'sentiment': sentiment,
                        'input_text': f"Aspect: {aspect}. Sentence: {text}"
                    })
        
        return pd.DataFrame(pairs)
    
    def validate_data(self, df: pd.DataFrame, task: str = "ate") -> Dict:
        """
        Validate ABSA dataset for quality and consistency.
        
        Args:
            df: Input DataFrame
            task: Task type for validation
            
        Returns:
            Validation report
        """
        report = {
            'total_samples': len(df),
            'issues': [],
            'statistics': {}
        }
        
        if task == "ate":
            # Check for missing aspects
            missing_aspects = df['aspects'].isna().sum()
            if missing_aspects > 0:
                report['issues'].append(f"{missing_aspects} samples with missing aspects")
            
            # Aspect statistics
            all_aspects = []
            for aspects in df['aspects'].dropna():
                if isinstance(aspects, str):
                    try:
                        aspect_list = eval(aspects)
                        all_aspects.extend(aspect_list)
                    except:
                        all_aspects.append(aspects)
                else:
                    all_aspects.extend(aspects)
            
            aspect_counts = Counter(all_aspects)
            report['statistics']['most_common_aspects'] = aspect_counts.most_common(10)
            report['statistics']['unique_aspects'] = len(aspect_counts)
            report['statistics']['avg_aspects_per_text'] = len(all_aspects) / len(df)
        
        elif task == "asc":
            # Check sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            report['statistics']['sentiment_distribution'] = sentiment_counts.to_dict()
            
            # Check for imbalanced data
            min_class = sentiment_counts.min()
            max_class = sentiment_counts.max()
            if max_class / min_class > 3:
                report['issues'].append("Significant class imbalance detected")
        
        # Text length statistics
        text_lengths = [len(text.split()) for text in df['text']]
        report['statistics']['text_length'] = {
            'mean': np.mean(text_lengths),
            'std': np.std(text_lengths),
            'min': np.min(text_lengths),
            'max': np.max(text_lengths)
        }
        
        return report
    
    def load_ate_data(self, data_dir, train_size=None, val_size=None, test_size=None):
        """
        Load BIO-tagged data for ATE from splits.
        Returns: train_data, val_data, test_data (as torch Dataset)
        """
        class SimpleATEDataset(Dataset):
            def __init__(self, df, preprocessor, max_length=128):
                # Use 'clean_text' if present, else fallback to 'text'
                if 'clean_text' in df.columns:
                    self.texts = df['clean_text'].tolist()
                elif 'text' in df.columns:
                    self.texts = df['text'].tolist()
                else:
                    raise ValueError("No 'clean_text' or 'text' column found in ATE split CSV.")
                # Use bio_tags if present, else try to generate from aspects
                if 'bio_tags' in df.columns:
                    # bio_tags may be stored as a string representation of a list
                    def parse_tags(x):
                        if isinstance(x, str):
                            try:
                                tags = eval(x)
                                if isinstance(tags, list):
                                    return tags
                            except:
                                pass
                        return []
                    self.bio_tags = df['bio_tags'].apply(parse_tags).tolist()
                else:
                    # Fallback: try to generate from aspects
                    if 'aspects' in df.columns:
                        self.bio_tags = [preprocessor.create_bio_tags(t, eval(a) if isinstance(a, str) else []) for t, a in zip(self.texts, df['aspects'])]
                    else:
                        self.bio_tags = [['O'] * len(t.split()) for t in self.texts]
                self.preprocessor = preprocessor
                self.max_length = max_length
                # Convert bio_tags to label ids
                tag2id = {'B-ASP': 0, 'I-ASP': 1, 'B-OP': 2, 'I-OP': 3, 'O': 4}
                self.labels = [[tag2id.get(tag, 4) for tag in tags] for tags in self.bio_tags]
                # Prepare data
                self.data = self.preprocessor.prepare_ate_data(self.texts, [[] for _ in self.texts], max_length=self.max_length)
                # Overwrite labels with correct ones
                self.data['labels'] = torch.tensor([self._pad_labels(l, self.max_length) for l in self.labels], dtype=torch.long)
            def _pad_labels(self, label_ids, max_length):
                # Pad or truncate label_ids to max_length, using -100 for ignored positions
                if len(label_ids) >= max_length:
                    return label_ids[:max_length]
                return label_ids + [-100] * (max_length - len(label_ids))
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                item = {
                    'input_ids': self.data['input_ids'][idx],
                    'attention_mask': self.data['attention_mask'][idx],
                    'labels': self.data['labels'][idx]
                }
                # Include NER tags if available in the prepared data
                if 'ner_tags' in self.data:
                    item['ner_tags'] = self.data['ner_tags'][idx]
                return item
        train_df = pd.read_csv(f"{data_dir}/train.csv")
        val_df = pd.read_csv(f"{data_dir}/val.csv")
        test_df = pd.read_csv(f"{data_dir}/test.csv")
        train_data = SimpleATEDataset(train_df, self, max_length=128)
        val_data = SimpleATEDataset(val_df, self, max_length=128)
        test_data = SimpleATEDataset(test_df, self, max_length=128)
        return train_data, val_data, test_data

    def load_asc_data(self, data_dir, train_size=None, val_size=None, test_size=None, max_length=None):
        """
        Load ASC aspect-sentiment pairs from splits.
        Returns: train_data, val_data, test_data (as torch Dataset)
        """
        import pandas as pd
        from torch.utils.data import Dataset
        # Load CSVs
        train_df = pd.read_csv(f"{data_dir}/asc_pairs.csv")
        val_df = pd.read_csv(f"{data_dir}/asc_pairs.csv") # fallback if no split
        test_df = pd.read_csv(f"{data_dir}/asc_pairs.csv")

        # Determine max_length
        if max_length is None:
            max_length = 256  # fallback default
        # Try to get from config if available
        if hasattr(self, 'max_length') and self.max_length is not None:
            max_length = self.max_length

        # Prepare data for ASC (tokenization, label mapping)
        train_data = self.prepare_asc_data(
            train_df['text'].tolist(),
            train_df['aspect'].tolist(),
            train_df['sentiment'].tolist(),
            max_length=max_length
        )
        val_data = self.prepare_asc_data(
            val_df['text'].tolist(),
            val_df['aspect'].tolist(),
            val_df['sentiment'].tolist(),
            max_length=max_length
        )
        test_data = self.prepare_asc_data(
            test_df['text'].tolist(),
            test_df['aspect'].tolist(),
            test_df['sentiment'].tolist(),
            max_length=max_length
        )
        # Return as ABSADataset for correct __getitem__
        return ABSADataset(train_data, task="asc"), ABSADataset(val_data, task="asc"), ABSADataset(test_data, task="asc")

    def load_end_to_end_data(self, data_dir, train_size=None, val_size=None, test_size=None, max_length=None, domain=None):
        """
        Load end-to-end ABSA data for multitask (aspect+sentiment) training.
        Returns: train_data, val_data, test_data (as torch Dataset)
        """
        import pandas as pd
        from torch.utils.data import Dataset
        aspect_cat_map = self.aspect_domains.get('restaurant', None)
        # Try to get from config if available
        if hasattr(self, 'aspect_categories'):
            aspect_cat_map = self.aspect_categories
        else:
            # Fallback: use config from bert_end_to_end.yaml
            aspect_cat_map = {
                'food': 0, 'service': 1, 'price': 2, 'ambience': 3, 'menu': 4, 'place': 5, 'staff': 6, 'miscellaneous': 7
            }
        num_aspect_categories = len(aspect_cat_map)
        num_sentiment_labels = 3  # negative, neutral, positive
        if max_length is None:
            max_length = 128
        # Patch: support per-domain end2end splits, avoid double domain
        split_dir = Path(data_dir)
        if domain is not None:
            if split_dir.name == domain:
                split_dir = split_dir / 'end2end'
            else:
                split_dir = split_dir / domain / 'end2end'
        else:
            split_dir = split_dir / 'end2end'
        train_df = pd.read_csv(split_dir / 'train.csv')
        val_df = pd.read_csv(split_dir / 'val.csv')
        test_df = pd.read_csv(split_dir / 'test.csv')
        def parse_list(val):
            if isinstance(val, str):
                try:
                    return eval(val)
                except:
                    return []
            return val if isinstance(val, list) else []
        class EndToEndABSADataset(Dataset):
            def __init__(self, df, tokenizer, max_length, aspect_cat_map, num_sentiment_labels):
                self.texts = df['text'].tolist()
                self.aspects = df['aspects'].apply(parse_list).tolist()
                self.sentiments = df['sentiments'].apply(parse_list).tolist()
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.aspect_cat_map = aspect_cat_map
                self.num_aspect_categories = len(aspect_cat_map)
                self.num_sentiment_labels = num_sentiment_labels
            def __len__(self):
                return len(self.texts)
            def __getitem__(self, idx):
                text = self.texts[idx]
                aspects = self.aspects[idx]
                sentiments = self.sentiments[idx]
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                # aspect_labels: multi-hot vector [num_aspect_categories]
                aspect_labels = torch.zeros(self.num_aspect_categories, dtype=torch.float)
                sentiment_labels = torch.full((self.num_aspect_categories,), -100, dtype=torch.long)
                for asp, sent in zip(aspects, sentiments):
                    if asp in self.aspect_cat_map:
                        idx_ = self.aspect_cat_map[asp]
                        aspect_labels[idx_] = 1.0
                        # Normalize sentiment
                        norm_sent = 1  # default neutral
                        s = str(sent).lower().strip()
                        if s in ['negative', 'neg', '0', '0.0']:
                            norm_sent = 0
                        elif s in ['positive', 'pos', '2', '2.0', '1']:
                            norm_sent = 2
                        sentiment_labels[idx_] = norm_sent
                return {
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'aspect_labels': aspect_labels,
                    'sentiment_labels': sentiment_labels
                }
        train_data = EndToEndABSADataset(train_df, self.tokenizer, max_length, aspect_cat_map, num_sentiment_labels)
        val_data = EndToEndABSADataset(val_df, self.tokenizer, max_length, aspect_cat_map, num_sentiment_labels)
        test_data = EndToEndABSADataset(test_df, self.tokenizer, max_length, aspect_cat_map, num_sentiment_labels)
        return train_data, val_data, test_data

    def _align_ner_tags_with_absa_tokens(self, ner_tokens: List[str], ner_tags: List[str], absa_tokens: List[str]) -> List[str]:
        """
        Align NER tags with ABSA tokens using longest common subsequence approach.
        
        Args:
            ner_tokens: Tokens from NER tokenization
            ner_tags: NER tags corresponding to ner_tokens
            absa_tokens: Tokens from ABSA tokenization
            
        Returns:
            List of NER tags aligned with ABSA tokens
        """
        if len(ner_tokens) == len(absa_tokens):
            return ner_tags
            
        aligned_tags = []
        ner_idx = 0
        
        for absa_token in absa_tokens:
            if ner_idx < len(ner_tokens):
                ner_token = ner_tokens[ner_idx]
                
                # Exact match
                if absa_token.lower() == ner_token.lower():
                    aligned_tags.append(ner_tags[ner_idx])
                    ner_idx += 1
                # ABSA token is substring of NER token (NER token was split)
                elif ner_token.lower().startswith(absa_token.lower()):
                    aligned_tags.append(ner_tags[ner_idx])
                    # Don't increment ner_idx yet, might need it for next ABSA token
                # NER token is substring of ABSA token (multiple NER tokens combined)
                elif absa_token.lower().startswith(ner_token.lower()):
                    aligned_tags.append(ner_tags[ner_idx])
                    ner_idx += 1
                    # Try to consume more NER tokens if they continue the ABSA token
                    while (ner_idx < len(ner_tokens) and 
                           absa_token.lower().find(ner_tokens[ner_idx].lower()) != -1):
                        ner_idx += 1
                # Fuzzy match - look ahead for similar tokens
                else:
                    found_match = False
                    for lookahead in range(min(3, len(ner_tokens) - ner_idx)):
                        if ner_tokens[ner_idx + lookahead].lower() == absa_token.lower():
                            # Use the tag from the matched token
                            aligned_tags.append(ner_tags[ner_idx + lookahead])
                            ner_idx = ner_idx + lookahead + 1
                            found_match = True
                            break
                    
                    if not found_match:
                        # No good match found, use 'O' tag
                        aligned_tags.append('O')
                        # Advance ner_idx conservatively
                        if ner_idx < len(ner_tokens) - 1:
                            ner_idx += 1
            else:
                # No more NER tokens, use 'O'
                aligned_tags.append('O')
        
        return aligned_tags

    # ...existing code...
        
class ABSADataset(Dataset):
    """
    PyTorch Dataset for ABSA tasks.
    """
    
    def __init__(self, data: Dict, task: str = "ate"):
        """
        Initialize dataset.
        
        Args:
            data: Preprocessed data dictionary
            task: Task type ('ate', 'asc')
        """
        self.task = task
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = data['labels']
        
        # Include NER tags if available
        self.ner_tags = data.get('ner_tags', None)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
        
        # Include NER tags if available
        if self.ner_tags is not None:
            item['ner_tags'] = self.ner_tags[idx]
            
        return item


class ABSADataLoader:
    """
    Data loader for ABSA datasets with preprocessing pipeline.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessor = ABSAPreprocessor()
        self.logger = logging.getLogger(__name__)
    
    def load_semeval_data(self, data_path: str, task: str = "ate") -> pd.DataFrame:
        """
        Load SemEval ABSA datasets.
        
        Args:
            data_path: Path to data file
            task: Task type
            
        Returns:
            Loaded DataFrame
        """
        data_path = Path(data_path)
        
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix == '.json':
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Validate required columns
        required_cols = ['text']
        if task == "ate":
            required_cols.extend(['aspects'])
        elif task == "asc":
            required_cols.extend(['aspect', 'sentiment'])
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def prepare_datasets(self, train_path: str, val_path: str = None, 
                        test_path: str = None, task: str = "ate") -> Dict:
        """
        Prepare train/val/test datasets.
        
        Args:
            train_path: Training data path
            val_path: Validation data path
            test_path: Test data path
            task: Task type
            
        Returns:
            Dictionary with prepared datasets
        """
        datasets = {}
        
        # Load training data
        train_df = self.load_semeval_data(train_path, task)
        self.logger.info(f"Loaded {len(train_df)} training samples")
        
        if task == "ate":
            train_data = self.preprocessor.prepare_ate_data(
                train_df['text'].tolist(),
                train_df['aspects'].tolist(),
                max_length=self.config['data']['max_seq_length']
            )
        elif task == "asc":
            train_data = self.preprocessor.prepare_asc_data(
                train_df['text'].tolist(),
                train_df['aspect'].tolist(),
                train_df['sentiment'].tolist(),
                max_length=self.config['data']['max_seq_length']
            )
        
        datasets['train'] = ABSADataset(train_data, task)
        
        # Load validation data
        if val_path:
            val_df = self.load_semeval_data(val_path, task)
            if task == "ate":
                val_data = self.preprocessor.prepare_ate_data(
                    val_df['text'].tolist(),
                    val_df['aspects'].tolist(),
                    max_length=self.config['data']['max_seq_length']
                )
            elif task == "asc":
                val_data = self.preprocessor.prepare_asc_data(
                    val_df['text'].tolist(),
                    val_df['aspect'].tolist(),
                    val_df['sentiment'].tolist(),
                    max_length=self.config['data']['max_seq_length']
                )
            datasets['val'] = ABSADataset(val_data, task)
        
        # Load test data
        if test_path:
            test_df = self.load_semeval_data(test_path, task)
            if task == "ate":
                test_data = self.preprocessor.prepare_ate_data(
                    test_df['text'].tolist(),
                    test_df['aspects'].tolist(),
                    max_length=self.config['data']['max_seq_length']
                )
            elif task == "asc":
                test_data = self.preprocessor.prepare_asc_data(
                    test_df['text'].tolist(),
                    test_df['aspect'].tolist(),
                    test_df['sentiment'].tolist(),
                    max_length=self.config['data']['max_seq_length']
                )
            datasets['test'] = ABSADataset(test_data, task)
        
        return datasets

def collate_end2end_absa_batch(batch):
    """Collate function for end-to-end multitask ABSA batches."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    aspect_labels = torch.stack([item['aspect_labels'] for item in batch])
    sentiment_labels = torch.stack([item['sentiment_labels'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'aspect_labels': aspect_labels,
        'sentiment_labels': sentiment_labels
    }
