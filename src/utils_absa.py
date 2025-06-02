"""
ABSA Utility Functions (moved from notebook for modularity)
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
import json

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
    def __init__(self, task: str = 'ate'):
        self.task = task
        self.stop_words = set(stopwords.words('english'))
        self.aspect_domains = {
            'restaurant': ['food', 'service', 'ambiance', 'price', 'location', 'staff'],
            'laptop': ['battery', 'screen', 'keyboard', 'performance', 'price', 'design'],
            'hotel': ['room', 'service', 'location', 'price', 'staff', 'amenities']
        }
        self.sentiment_mappings = {
            'positive': ['positive', 'pos', '1', 1, 2],
            'negative': ['negative', 'neg', '0', 0, 0],
            'neutral': ['neutral', 'neu', 'conflict', '2', 2, 1]
        }
    def clean_text(self, text: str) -> str:
        if not text or pd.isna(text):
            return ''
        text = str(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://\S+', '[URL]', text)
        text = re.sub(r'[^\w\s\.!?,;:\-\(\)\[\]\'\"]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        contractions = {"won't": 'will not', "can't": 'cannot', "n't": ' not', "'re": ' are', "'ve": ' have', "'ll": ' will', "'d": ' would', "'m": ' am'}
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        return text.strip()
    def create_bio_tags(self, text: str, aspects: List[str]) -> Tuple[List[str], List[str]]:
        tokens = word_tokenize(text.lower())
        tags = ['O'] * len(tokens)
        for aspect in aspects:
            aspect_tokens = word_tokenize(aspect.lower())
            for i in range(len(tokens) - len(aspect_tokens) + 1):
                if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                    tags[i] = 'B-ASP'
                    for j in range(1, len(aspect_tokens)):
                        if i + j < len(tags):
                            tags[i + j] = 'I-ASP'
                    break
        return tokens, tags
    def normalize_sentiment(self, sentiment: Union[str, int]) -> int:
        sentiment = str(sentiment).lower().strip()
        for normalized, variants in self.sentiment_mappings.items():
            if sentiment in [str(v).lower() for v in variants]:
                if normalized == 'negative':
                    return 0
                elif normalized == 'neutral':
                    return 1
                elif normalized == 'positive':
                    return 2
        return 1
    def create_aspect_sentiment_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        pairs = []
        for _, row in df.iterrows():
            text = row['text']
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
            for aspect, sentiment in zip(aspects, sentiments):
                if aspect and sentiment:
                    formatted_input = f'Aspect: {aspect}. Sentence: {text}'
                    pairs.append({
                        'input_text': formatted_input,
                        'aspect': aspect,
                        'text': text,
                        'sentiment': sentiment,
                        'normalized_sentiment': self.normalize_sentiment(sentiment)
                    })
        return pd.DataFrame(pairs)
    def validate_data(self, df: pd.DataFrame, task: str = 'ate') -> Dict:
        report = {'total_samples': len(df), 'issues': [], 'statistics': {}}
        if task == 'ate':
            missing_aspects = df['aspects'].isna().sum()
            if missing_aspects > 0:
                report['issues'].append(f'{missing_aspects} samples with missing aspects')
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
        elif task == 'asc':
            sentiment_counts = df['sentiment'].value_counts()
            report['statistics']['sentiment_distribution'] = sentiment_counts.to_dict()
            min_class = sentiment_counts.min()
            max_class = sentiment_counts.max()
            if max_class / min_class > 3:
                report['issues'].append('Significant class imbalance detected')
        text_lengths = [len(text.split()) for text in df['text']]
        report['statistics']['text_length'] = {
            'mean': np.mean(text_lengths),
            'std': np.std(text_lengths),
            'min': np.min(text_lengths),
            'max': np.max(text_lengths)
        }
        return report

def parse_semeval_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for sentence in root.findall('.//sentence'):
        text = sentence.find('text').text if sentence.find('text') is not None else ''
        aspects = []
        sentiments = []
        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is not None:
            for aspect_term in aspect_terms.findall('aspectTerm'):
                term = aspect_term.get('term', '')
                polarity = aspect_term.get('polarity', 'neutral')
                aspects.append(term)
                sentiments.append(polarity)
        aspect_categories = sentence.find('aspectCategories')
        if aspect_categories is not None:
            for aspect_cat in aspect_categories.findall('aspectCategory'):
                category = aspect_cat.get('category', '')
                polarity = aspect_cat.get('polarity', 'neutral')
                aspects.append(category)
                sentiments.append(polarity)
        data.append({'text': text, 'aspects': aspects, 'sentiments': sentiments, 'num_aspects': len(aspects)})
    return data

def load_absa_data(file_path):
    if str(file_path).endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        processed_data = []
        for item in data:
            text = ' '.join(item['token']) if 'token' in item else item.get('text', '')
            if 'aspects' in item and item['aspects']:
                for aspect in item['aspects']:
                    processed_data.append({
                        'text': text,
                        'aspect_term': ' '.join(aspect['term']) if 'term' in aspect else aspect,
                        'polarity': aspect.get('polarity', None),
                        'aspect_start': aspect.get('from', None),
                        'aspect_end': aspect.get('to', None),
                        'text_length': len(text),
                        'word_count': len(text.split()),
                        'pos_tags': item.get('pos', None)
                    })
            else:
                processed_data.append({
                    'text': text,
                    'aspect_term': None,
                    'polarity': None,
                    'aspect_start': None,
                    'aspect_end': None,
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'pos_tags': item.get('pos', None)
                })
        return pd.DataFrame(processed_data)
    elif str(file_path).endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError(f'Unsupported file type: {file_path}')
