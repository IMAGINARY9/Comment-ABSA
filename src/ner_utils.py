
'''
Utility for integrating NER model predictions into Comment-ABSA.
'''
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json, Tokenizer
from nltk.tokenize import word_tokenize
import logging

logger = logging.getLogger(__name__)

class CommentABSANERExtractor:    
    def __init__(self, model_path: str, word_tokenizer_path: str, tag_vocab_path: str, ner_max_seq_length: int = 128):
        '''
        Initializes the NER extractor.

        Args:
            model_path: Path to the trained NER Keras model (.keras or .h5).
            word_tokenizer_path: Path to the Keras Tokenizer JSON file for words.
            tag_vocab_path: Path to the JSON file mapping NER tags to IDs.
            ner_max_seq_length: Maximum sequence length used by the NER model.
        '''
        logger.info(f"Loading NER model from: {model_path}")
        self.model = load_model(model_path)
        logger.info("NER model loaded successfully.")
        
        logger.info(f"Loading NER word tokenizer from: {word_tokenizer_path}")
        try:
            with open(word_tokenizer_path, 'r', encoding='utf-8') as f:
                tokenizer_config = json.load(f)
                self.word_tokenizer = tokenizer_from_json(json.dumps(tokenizer_config))
            logger.info("NER word tokenizer loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load NER tokenizer with error: {e}")
            logger.info("Creating basic tokenizer fallback...")
            # Create a basic tokenizer as fallback
            self.word_tokenizer = Tokenizer(oov_token="<UNK>")
            # Add a simple word index for basic functionality
            self.word_tokenizer.word_index = {"<UNK>": 1}
            logger.info("Basic NER tokenizer fallback created.")

        logger.info(f"Loading NER tag vocabulary from: {tag_vocab_path}")
        with open(tag_vocab_path, 'r', encoding='utf-8') as f:
            self.tag_vocab = json.load(f)  # Expected format: {"tag": index}
        logger.info("NER tag vocabulary loaded successfully.")
        
        self.idx2tag = {int(idx): tag for tag, idx in self.tag_vocab.items()} # Ensure idx is int for lookup
        # Ensure PAD is handled, common practice is index 0 for PAD in NER if not explicitly in vocab
        if 0 not in self.idx2tag:
             self.idx2tag[0] = 'PAD' 
        elif self.idx2tag[0].upper() != 'PAD': # If 0 is something else, warn or remap
            logger.warning(f"Index 0 in NER tag_vocab is {self.idx2tag[0]}, not PAD. Ensure this is intended.")

        self.ner_max_seq_length = ner_max_seq_length
        self.oov_token_id = self.word_tokenizer.word_index.get(self.word_tokenizer.oov_token, 1) if self.word_tokenizer.oov_token else 1

    def predict_ner_tags(self, text: str) -> tuple[list[str], list[str]]:
        '''
        Predicts NER tags for a given text using word_tokenize.

        Args:
            text: The input string.

        Returns:
            A tuple containing: 
                - tokens (list[str]): The list of tokens from word_tokenize.
                - predicted_tags (list[str]): The list of predicted NER tags for each token.
        '''
        if not text.strip():
            return [], []
            
        tokens = word_tokenize(text) # NLTK's word_tokenize for consistency

        # Convert tokens to sequences using the loaded Keras NER tokenizer
        # Handles lowercasing and OOV tokens as defined by the NER tokenizer
        sequence = [self.word_tokenizer.word_index.get(token, self.oov_token_id) 
                    for token in tokens] # Keras tokenizer usually handles lowercasing during fit
                                        # If NER tokenizer was not fit on lowercased text, adjust here or ensure consistency.
                                        # Assuming NER tokenizer handles casing as it was trained.

        if not sequence: # If all tokens were OOV and no OOV token ID, or empty text
            return tokens, ['O'] * len(tokens)

        padded_sequence = pad_sequences([sequence], maxlen=self.ner_max_seq_length, padding='post', truncating='post')
        
        raw_predictions = self.model.predict(padded_sequence, verbose=0) # verbose=0 to suppress Keras progress bar
        
        predicted_indices = np.argmax(raw_predictions, axis=-1)[0] 
        
        predicted_tags = []
        for i in range(len(tokens)): # Iterate up to the number of original tokens
            if i < len(predicted_indices):
                tag_idx = predicted_indices[i]
                predicted_tags.append(self.idx2tag.get(tag_idx, 'O')) # Default to 'O' if tag_idx is unknown
            else: # Should not happen if NER max_seq_length >= len(tokens)
                predicted_tags.append('O')
        
        return tokens, predicted_tags
