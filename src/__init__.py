"""
Aspect-Based Sentiment Analysis (ABSA) Package

This package provides tools for aspect-based sentiment analysis including
aspect term extraction, aspect sentiment classification, and end-to-end models.
"""

__version__ = "0.1.0"
__author__ = "Sentiment Analysis Team"

from .preprocessing import ABSAPreprocessor
from .models import (
    DeBERTaATE, DeBERTaASC, BERTForABSA, 
    EndToEndABSA, BiLSTMCRF
)
from .training import ABSATrainer, ATETrainer, ASCTrainer
from .evaluation import ABSAEvaluator

__all__ = [
    "ABSAPreprocessor",
    "DeBERTaATE",
    "DeBERTaASC", 
    "BERTForABSA",
    "EndToEndABSA",
    "BiLSTMCRF",
    "ABSATrainer",
    "ATETrainer",
    "ASCTrainer", 
    "ABSAEvaluator"
]
