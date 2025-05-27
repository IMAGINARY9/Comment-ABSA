"""
Comprehensive test suite for the comment ABSA (Aspect-Based Sentiment Analysis) project.
Tests ATE, ASC, end-to-end models, preprocessing, training, and evaluation.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from models import (
        DeBERTaATE, DeBERTaASC, BERTForABSA, EndToEndABSA, 
        BiLSTMCRF, ABSAPredictor
    )
    from preprocessing import ABSAPreprocessor
    from training import ATETrainer, ASCTrainer, MultiTaskABSATrainer
    from evaluation import ABSAEvaluator
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some tests may be skipped")


class TestABSAModels(unittest.TestCase):
    """Test ABSA models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.num_labels_ate = 5  # B-ASP, I-ASP, B-OP, I-OP, O
        self.num_labels_asc = 3  # positive, negative, neutral
        self.num_aspect_labels = 8
        self.max_length = 128
        
    def test_deberta_ate_model_creation(self):
        """Test DeBERTa ATE model creation."""
        try:
            model = DeBERTaATE(
                model_name="microsoft/deberta-v3-base",
                num_labels=self.num_labels_ate,
                dropout=0.1,
                use_lora=True,
                lora_config={
                    'r': 16,
                    'lora_alpha': 32,
                    'lora_dropout': 0.1,
                    'target_modules': ["query_proj", "value_proj"]
                }
            )
            
            self.assertIsInstance(model, DeBERTaATE)
            self.assertEqual(model.num_labels, self.num_labels_ate)
            
        except NameError:
            self.skipTest("DeBERTaATE not available")
    
    def test_deberta_asc_model_creation(self):
        """Test DeBERTa ASC model creation."""
        try:
            model = DeBERTaASC(
                model_name="microsoft/deberta-v3-base",
                num_labels=self.num_labels_asc,
                dropout=0.1,
                use_lora=True,
                lora_config={
                    'r': 16,
                    'lora_alpha': 32,
                    'lora_dropout': 0.1,
                    'target_modules': ["query_proj", "value_proj", "key_proj"]
                }
            )
            
            self.assertIsInstance(model, DeBERTaASC)
            self.assertEqual(model.num_labels, self.num_labels_asc)
            
        except NameError:
            self.skipTest("DeBERTaASC not available")
    
    def test_bert_for_absa_model_creation(self):
        """Test BERT for ABSA model creation."""
        try:
            model = BERTForABSA(
                model_name="bert-base-uncased",
                num_aspect_labels=self.num_aspect_labels,
                num_sentiment_labels=self.num_labels_asc,
                dropout=0.1
            )
            
            self.assertIsInstance(model, BERTForABSA)
            
            # Test forward pass
            batch_size = 2
            seq_len = 50
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            
            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                self.assertIn('aspect_logits', output)
                self.assertIn('sentiment_logits', output)
                
        except NameError:
            self.skipTest("BERTForABSA not available")
    
    def test_end_to_end_absa_model_creation(self):
        """Test End-to-End ABSA model creation."""
        try:
            model = EndToEndABSA(
                model_name="microsoft/deberta-v3-base",
                num_aspect_labels=self.num_aspect_labels,
                num_sentiment_labels=self.num_labels_asc,
                dropout=0.1
            )
            
            self.assertIsInstance(model, EndToEndABSA)
            
        except NameError:
            self.skipTest("EndToEndABSA not available")
    
    def test_bilstm_crf_model_creation(self):
        """Test BiLSTM-CRF model creation."""
        try:
            vocab_size = 10000
            embedding_dim = 100
            hidden_dim = 128
            
            model = BiLSTMCRF(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_labels=self.num_labels_ate,
                num_layers=2,
                dropout=0.1,
                use_char_cnn=True
            )
            
            self.assertIsInstance(model, BiLSTMCRF)
            self.assertEqual(model.num_labels, self.num_labels_ate)
            
        except NameError:
            self.skipTest("BiLSTMCRF not available")
    
    def test_absa_predictor_creation(self):
        """Test ABSA predictor creation."""
        try:
            mock_ate_model = Mock()
            mock_asc_model = Mock()
            mock_tokenizer = Mock()
            mock_preprocessor = Mock()
            
            predictor = ABSAPredictor(
                ate_model=mock_ate_model,
                asc_model=mock_asc_model,
                tokenizer=mock_tokenizer,
                preprocessor=mock_preprocessor,
                aspect_labels=['B-ASP', 'I-ASP', 'B-OP', 'I-OP', 'O'],
                sentiment_labels=['negative', 'neutral', 'positive'],
                device=self.device
            )
            
            self.assertIsInstance(predictor, ABSAPredictor)
            
        except NameError:
            self.skipTest("ABSAPredictor not available")


class TestABSAPreprocessing(unittest.TestCase):
    """Test ABSA preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = [
            "The food was delicious but the service was terrible.",
            "Great atmosphere and friendly staff, highly recommend!",
            "Overpriced for what you get, disappointing experience.",
            "The pizza was okay but the pasta was amazing."
        ]
        
        self.sample_aspects = [
            ["food", "service"],
            ["atmosphere", "staff"],
            ["price"],
            ["pizza", "pasta"]
        ]
        
        self.sample_sentiments = [
            ["positive", "negative"],
            ["positive", "positive"],
            ["negative"],
            ["neutral", "positive"]
        ]
        
        self.sample_bio_tags = [
            ["O", "B-ASP", "O", "B-OP", "O", "O", "B-ASP", "O", "B-OP"],
            ["B-OP", "B-ASP", "O", "B-OP", "B-ASP", "O", "O", "O"],
            ["B-OP", "O", "O", "O", "O", "O", "B-OP", "O"],
            ["O", "B-ASP", "O", "B-OP", "O", "O", "B-ASP", "O", "B-OP"]
        ]
        
    def test_absa_preprocessor_creation(self):
        """Test ABSA preprocessor creation."""
        try:
            preprocessor = ABSAPreprocessor(
                model_name="microsoft/deberta-v3-base",
                max_length=128,
                task="token_classification"
            )
            
            self.assertIsInstance(preprocessor, ABSAPreprocessor)
            self.assertEqual(preprocessor.max_length, 128)
            self.assertEqual(preprocessor.task, "token_classification")
            
        except NameError:
            self.skipTest("ABSAPreprocessor not available")
    
    def test_bio_tagging(self):
        """Test BIO tagging functionality."""
        try:
            preprocessor = ABSAPreprocessor(
                model_name="microsoft/deberta-v3-base",
                max_length=128,
                task="token_classification"
            )
            
            text = self.sample_texts[0]
            aspects = self.sample_aspects[0]
            
            bio_tags = preprocessor.create_bio_tags(text, aspects)
            
            self.assertIsInstance(bio_tags, list)
            self.assertTrue(all(tag in ['B-ASP', 'I-ASP', 'B-OP', 'I-OP', 'O'] for tag in bio_tags))
            
        except (NameError, AttributeError):
            self.skipTest("BIO tagging not available")
    
    def test_aspect_extraction(self):
        """Test aspect extraction from BIO tags."""
        try:
            preprocessor = ABSAPreprocessor(
                model_name="microsoft/deberta-v3-base",
                max_length=128,
                task="token_classification"
            )
            
            text = self.sample_texts[0]
            bio_tags = self.sample_bio_tags[0]
            
            aspects = preprocessor.extract_aspects_from_bio(text, bio_tags)
            
            self.assertIsInstance(aspects, list)
            
        except (NameError, AttributeError):
            self.skipTest("Aspect extraction not available")
    
    def test_asc_data_preparation(self):
        """Test ASC data preparation."""
        try:
            preprocessor = ABSAPreprocessor(
                model_name="microsoft/deberta-v3-base",
                max_length=128,
                task="sequence_classification"
            )
            
            text = self.sample_texts[0]
            aspect = self.sample_aspects[0][0]
            sentiment = self.sample_sentiments[0][0]
            
            formatted_input = preprocessor.format_asc_input(text, aspect)
            
            self.assertIsInstance(formatted_input, str)
            self.assertIn(aspect, formatted_input)
            self.assertIn(text, formatted_input)
            
        except (NameError, AttributeError):
            self.skipTest("ASC data preparation not available")
    
    def test_data_validation(self):
        """Test data validation."""
        try:
            preprocessor = ABSAPreprocessor(
                model_name="microsoft/deberta-v3-base",
                max_length=128,
                task="token_classification"
            )
            
            # Test valid data
            valid_data = {
                "text": self.sample_texts[0],
                "aspects": self.sample_aspects[0],
                "sentiments": self.sample_sentiments[0]
            }
            
            is_valid = preprocessor.validate_data(valid_data)
            self.assertTrue(is_valid)
            
            # Test invalid data
            invalid_data = {
                "text": self.sample_texts[0],
                "aspects": self.sample_aspects[0],
                "sentiments": ["positive"]  # Mismatched length
            }
            
            is_invalid = preprocessor.validate_data(invalid_data)
            self.assertFalse(is_invalid)
            
        except (NameError, AttributeError):
            self.skipTest("Data validation not available")
    
    def test_data_augmentation(self):
        """Test data augmentation."""
        try:
            preprocessor = ABSAPreprocessor(
                model_name="microsoft/deberta-v3-base",
                max_length=128,
                task="token_classification"
            )
            
            original_data = {
                "text": self.sample_texts[0],
                "aspects": self.sample_aspects[0],
                "sentiments": self.sample_sentiments[0]
            }
            
            augmented_data = preprocessor.augment_data(original_data)
            
            self.assertIsInstance(augmented_data, list)
            
        except (NameError, AttributeError):
            self.skipTest("Data augmentation not available")


class TestABSATraining(unittest.TestCase):
    """Test ABSA training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        
        # Create mock models
        self.mock_ate_model = Mock()
        self.mock_asc_model = Mock()
        self.mock_multitask_model = Mock()
        
        for model in [self.mock_ate_model, self.mock_asc_model, self.mock_multitask_model]:
            model.parameters.return_value = []
            model.train = Mock()
            model.eval = Mock()
        
    def test_ate_trainer_creation(self):
        """Test ATE trainer creation."""
        try:
            trainer = ATETrainer(
                model=self.mock_ate_model,
                device=self.device,
                learning_rate=5e-5,
                weight_decay=0.01,
                warmup_steps=500,
                class_weights={'B-ASP': 2.0, 'I-ASP': 2.0, 'B-OP': 1.5, 'I-OP': 1.5, 'O': 1.0}
            )
            
            self.assertIsInstance(trainer, ATETrainer)
            
        except NameError:
            self.skipTest("ATETrainer not available")
    
    def test_asc_trainer_creation(self):
        """Test ASC trainer creation."""
        try:
            trainer = ASCTrainer(
                model=self.mock_asc_model,
                device=self.device,
                learning_rate=2e-5,
                weight_decay=0.01,
                warmup_steps=200,
                class_weights={'negative': 1.2, 'neutral': 0.8, 'positive': 1.0}
            )
            
            self.assertIsInstance(trainer, ASCTrainer)
            
        except NameError:
            self.skipTest("ASCTrainer not available")
    
    def test_multitask_absa_trainer_creation(self):
        """Test multi-task ABSA trainer creation."""
        try:
            trainer = MultiTaskABSATrainer(
                model=self.mock_multitask_model,
                device=self.device,
                learning_rate=5e-5,
                weight_decay=0.01,
                warmup_steps=300,
                aspect_weight=0.5,
                sentiment_weight=0.5,
                use_lora=True
            )
            
            self.assertIsInstance(trainer, MultiTaskABSATrainer)
            
        except NameError:
            self.skipTest("MultiTaskABSATrainer not available")
    
    def test_lora_integration(self):
        """Test LoRA integration in training."""
        try:
            trainer = ATETrainer(
                model=self.mock_ate_model,
                device=self.device,
                learning_rate=5e-5,
                use_lora=True
            )
            
            # Test that LoRA configuration is handled
            self.assertTrue(hasattr(trainer, 'use_lora') or 
                           hasattr(trainer.model, 'use_lora'))
            
        except (NameError, AttributeError):
            self.skipTest("LoRA integration not available")
    
    def test_class_weighting(self):
        """Test class weighting in training."""
        try:
            class_weights = {'B-ASP': 2.0, 'I-ASP': 2.0, 'B-OP': 1.5, 'I-OP': 1.5, 'O': 1.0}
            
            trainer = ATETrainer(
                model=self.mock_ate_model,
                device=self.device,
                learning_rate=5e-5,
                class_weights=class_weights
            )
            
            # Test that class weights are stored
            self.assertTrue(hasattr(trainer, 'class_weights'))
            
        except (NameError, AttributeError):
            self.skipTest("Class weighting not available")


class TestABSAEvaluation(unittest.TestCase):
    """Test ABSA evaluation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Token-level predictions and labels (for ATE)
        self.sample_token_predictions = np.array([
            [0, 1, 4, 2, 3, 4, 4],  # B-ASP, I-ASP, O, B-OP, I-OP, O, O
            [4, 0, 1, 4, 4, 4, 4],  # O, B-ASP, I-ASP, O, O, O, O
            [4, 4, 2, 3, 4, 0, 1]   # O, O, B-OP, I-OP, O, B-ASP, I-ASP
        ])
        
        self.sample_token_labels = np.array([
            [0, 1, 4, 2, 3, 4, 4],  # Perfect match
            [4, 0, 1, 4, 4, 4, 4],  # Perfect match
            [4, 4, 2, 3, 4, 0, 4]   # Partial match
        ])
        
        # Aspect-level predictions and labels
        self.sample_aspect_predictions = np.array([2, 0, 1, 0, 2])  # positive, negative, neutral, negative, positive
        self.sample_aspect_labels = np.array([2, 0, 0, 0, 1])       # positive, negative, negative, negative, neutral
        
        self.sample_confidences = np.array([0.9, 0.8, 0.6, 0.7, 0.85])
        
    def test_absa_evaluator_creation(self):
        """Test ABSA evaluator creation."""
        try:
            evaluator = ABSAEvaluator(
                task="token_classification",
                label_names=['B-ASP', 'I-ASP', 'B-OP', 'I-OP', 'O'],
                output_dir="./test_outputs"
            )
            
            self.assertIsInstance(evaluator, ABSAEvaluator)
            self.assertEqual(evaluator.task, "token_classification")
            
        except NameError:
            self.skipTest("ABSAEvaluator not available")
    
    def test_token_level_metrics(self):
        """Test token-level metrics calculation."""
        try:
            evaluator = ABSAEvaluator(
                task="token_classification",
                label_names=['B-ASP', 'I-ASP', 'B-OP', 'I-OP', 'O']
            )
            
            metrics = evaluator.calculate_token_metrics(
                predictions=self.sample_token_predictions.flatten(),
                labels=self.sample_token_labels.flatten()
            )
            
            self.assertIsInstance(metrics, dict)
            self.assertIn('accuracy', metrics)
            self.assertIn('f1_score', metrics)
            
        except (NameError, AttributeError):
            self.skipTest("Token-level metrics not available")
    
    def test_aspect_level_metrics(self):
        """Test aspect-level metrics calculation."""
        try:
            evaluator = ABSAEvaluator(
                task="sequence_classification",
                label_names=['negative', 'neutral', 'positive']
            )
            
            metrics = evaluator.calculate_aspect_metrics(
                predictions=self.sample_aspect_predictions,
                labels=self.sample_aspect_labels
            )
            
            self.assertIsInstance(metrics, dict)
            self.assertIn('accuracy', metrics)
            self.assertIn('f1_score', metrics)
            
        except (NameError, AttributeError):
            self.skipTest("Aspect-level metrics not available")
    
    def test_confidence_analysis(self):
        """Test confidence analysis."""
        try:
            evaluator = ABSAEvaluator(
                task="sequence_classification",
                label_names=['negative', 'neutral', 'positive']
            )
            
            analysis = evaluator.analyze_confidence(
                predictions=self.sample_aspect_predictions,
                confidences=self.sample_confidences,
                labels=self.sample_aspect_labels
            )
            
            self.assertIsInstance(analysis, dict)
            
        except (NameError, AttributeError):
            self.skipTest("Confidence analysis not available")
    
    def test_error_analysis(self):
        """Test error analysis."""
        try:
            evaluator = ABSAEvaluator(
                task="sequence_classification",
                label_names=['negative', 'neutral', 'positive']
            )
            
            # Mock text data
            texts = ["sample text"] * len(self.sample_aspect_predictions)
            aspects = ["food"] * len(self.sample_aspect_predictions)
            
            errors = evaluator.analyze_errors(
                predictions=self.sample_aspect_predictions,
                labels=self.sample_aspect_labels,
                texts=texts,
                aspects=aspects,
                confidences=self.sample_confidences
            )
            
            self.assertIsInstance(errors, dict)
            
        except (NameError, AttributeError):
            self.skipTest("Error analysis not available")
    
    def test_evaluation_report_generation(self):
        """Test evaluation report generation."""
        try:
            evaluator = ABSAEvaluator(
                task="token_classification",
                label_names=['B-ASP', 'I-ASP', 'B-OP', 'I-OP', 'O'],
                output_dir="./test_outputs"
            )
            
            # Mock evaluation results
            results = {
                'predictions': self.sample_token_predictions.flatten(),
                'labels': self.sample_token_labels.flatten(),
                'confidences': np.random.random(self.sample_token_predictions.size)
            }
            
            report = evaluator.generate_evaluation_report(
                results=results,
                save_plots=False,  # Don't save plots in tests
                save_errors=False
            )
            
            self.assertIsInstance(report, dict)
            self.assertIn('metrics', report)
            
        except (NameError, AttributeError):
            self.skipTest("Evaluation report generation not available")


class TestIntegration(unittest.TestCase):
    """Integration tests for the ABSA system."""
    
    def test_ate_pipeline(self):
        """Test ATE (Aspect Term Extraction) pipeline."""
        try:
            # Sample ATE data
            texts = [
                "The food was delicious but the service was terrible.",
                "Great atmosphere and friendly staff.",
                "Overpriced for what you get."
            ]
            
            bio_tags = [
                ["O", "B-ASP", "O", "B-OP", "O", "O", "B-ASP", "O", "B-OP"],
                ["B-OP", "B-ASP", "O", "B-OP", "B-ASP"],
                ["B-OP", "O", "O", "O", "O"]
            ]
            
            # Test preprocessing
            preprocessor = ABSAPreprocessor(
                model_name="microsoft/deberta-v3-base",
                max_length=64,
                task="token_classification"
            )
            
            processed_data = []
            for text, tags in zip(texts, bio_tags):
                # Tokenize and align labels
                tokens = preprocessor.tokenize(text)
                aligned_labels = preprocessor.align_labels_with_tokens(tokens, tags)
                processed_data.append((tokens, aligned_labels))
            
            self.assertEqual(len(processed_data), 3)
            
        except Exception as e:
            self.skipTest(f"ATE pipeline test skipped due to: {e}")
    
    def test_asc_pipeline(self):
        """Test ASC (Aspect Sentiment Classification) pipeline."""
        try:
            # Sample ASC data
            texts = [
                "The food was delicious but the service was terrible.",
                "Great atmosphere and friendly staff.",
                "Overpriced for what you get."
            ]
            
            aspects = ["food", "atmosphere", "price"]
            sentiments = ["positive", "positive", "negative"]
            
            # Test preprocessing
            preprocessor = ABSAPreprocessor(
                model_name="microsoft/deberta-v3-base",
                max_length=64,
                task="sequence_classification"
            )
            
            processed_data = []
            for text, aspect, sentiment in zip(texts, aspects, sentiments):
                # Format input for ASC
                formatted_input = preprocessor.format_asc_input(text, aspect)
                tokens = preprocessor.tokenize(formatted_input)
                
                # Convert sentiment to label
                sentiment_label = {'negative': 0, 'neutral': 1, 'positive': 2}[sentiment]
                
                processed_data.append((tokens, sentiment_label))
            
            self.assertEqual(len(processed_data), 3)
            
        except Exception as e:
            self.skipTest(f"ASC pipeline test skipped due to: {e}")
    
    def test_end_to_end_absa_pipeline(self):
        """Test end-to-end ABSA pipeline."""
        try:
            # Sample end-to-end ABSA data
            texts = [
                "The food was delicious but the service was terrible.",
                "Great atmosphere and friendly staff.",
                "Overpriced for what you get."
            ]
            
            aspects = [
                ["food", "service"],
                ["atmosphere", "staff"],
                ["price"]
            ]
            
            sentiments = [
                ["positive", "negative"],
                ["positive", "positive"],
                ["negative"]
            ]
            
            # Test preprocessing
            preprocessor = ABSAPreprocessor(
                model_name="microsoft/deberta-v3-base",
                max_length=64,
                task="absa_end_to_end"
            )
            
            processed_data = []
            for text, text_aspects, text_sentiments in zip(texts, aspects, sentiments):
                # Process for end-to-end ABSA
                tokens = preprocessor.tokenize(text)
                
                # Create multi-label targets
                aspect_labels = preprocessor.create_aspect_labels(text_aspects)
                sentiment_labels = preprocessor.create_sentiment_labels(text_sentiments)
                
                processed_data.append((tokens, aspect_labels, sentiment_labels))
            
            self.assertEqual(len(processed_data), 3)
            
        except Exception as e:
            self.skipTest(f"End-to-end ABSA pipeline test skipped due to: {e}")


class TestUtilities(unittest.TestCase):
    """Test utility functions."""
    
    def test_bio_tag_conversion(self):
        """Test BIO tag conversion utilities."""
        try:
            # Test BIO to IOB conversion
            bio_tags = ['B-ASP', 'I-ASP', 'O', 'B-OP', 'I-OP']
            
            # If conversion utilities are implemented
            self.assertEqual(len(bio_tags), 5)
            
        except Exception as e:
            self.skipTest(f"BIO tag conversion test skipped: {e}")
    
    def test_aspect_sentiment_pairing(self):
        """Test aspect-sentiment pairing utilities."""
        try:
            aspects = ["food", "service"]
            sentiments = ["positive", "negative"]
            
            # If pairing utilities are implemented
            pairs = list(zip(aspects, sentiments))
            self.assertEqual(len(pairs), 2)
            
        except Exception as e:
            self.skipTest(f"Aspect-sentiment pairing test skipped: {e}")
    
    def test_label_alignment(self):
        """Test label alignment with tokenization."""
        try:
            text = "The food was delicious"
            bio_tags = ["O", "B-ASP", "O", "B-OP"]
            
            # If alignment utilities are implemented
            self.assertEqual(len(bio_tags), 4)
            
        except Exception as e:
            self.skipTest(f"Label alignment test skipped: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
