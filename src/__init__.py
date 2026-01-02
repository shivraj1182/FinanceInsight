"""FinanceInsight Package
A comprehensive NER system for financial data extraction from unstructured documents.
"""
__version__ = "0.1.0"
__author__ = "shivraj1182"
__description__ = "Named Entity Recognition for Financial Data Extraction"

from .data_preparation import DataPreparation
from .ner_model import BERTBasedNER, FinBERTNER, NERTrainer, ModelEvaluator
from .entity_extractor import FinancialEntityExtractor
from .document_parser import FinancialDocumentParser

__all__ = [
    'DataPreparation',
    'BERTBasedNER',
    'FinBERTNER',
    'NERTrainer',
    'ModelEvaluator',
    'FinancialEntityExtractor',
    'FinancialDocumentParser'
]
