"""Data Preparation Module for FinanceInsight

This module handles data collection, preprocessing, and preparation
for NER model training.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('punkt')


class DataPreparation:
    """Handles data preparation for financial NER tasks."""
    
    def __init__(self):
        """Initialize data preparation module."""
        self.financial_entities = [
            'COMPANY', 'STOCK_TICKER', 'REVENUE', 'EARNINGS',
            'MARKET_CAP', 'DATE', 'PERSON', 'LOCATION', 'FINANCIAL_METRIC'
        ]
    
    def load_financial_documents(self, file_path: str) -> List[str]:
        """Load financial documents from file.
        
        Args:
            file_path: Path to financial document file
            
        Returns:
            List of document texts
        """
        documents = []
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                documents = df['text'].tolist()
            elif file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    documents = f.readlines()
        except Exception as e:
            print(f"Error loading documents: {e}")
        return documents
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess financial text.
        
        Args:
            text: Raw financial text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^a-zA-Z0-9$%€£\s.,;:-]', '', text)
        return text
    
    def tokenize_documents(self, documents: List[str]) -> List[List[str]]:
        """Tokenize documents into sentences and words.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of tokenized documents
        """
        tokenized = []
        for doc in documents:
            sentences = sent_tokenize(doc)
            doc_tokens = [word_tokenize(sent) for sent in sentences]
            tokenized.append(doc_tokens)
        return tokenized
    
    def prepare_training_data(self, documents: List[str], 
                            labels: List[List[str]] = None) -> Tuple[List[str], List[List[str]]]:
        """Prepare training data with labels.
        
        Args:
            documents: List of document texts
            labels: Optional labels for entities
            
        Returns:
            Tuple of (processed_documents, entity_labels)
        """
        processed_docs = []
        for doc in documents:
            processed = self.preprocess_text(doc)
            processed_docs.append(processed)
        
        if labels is None:
            labels = [['O'] * len(doc.split()) for doc in processed_docs]
        
        return processed_docs, labels
    
    def create_vocab(self, documents: List[str]) -> Dict[str, int]:
        """Create vocabulary from documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            Dictionary mapping words to indices
        """
        vocab = {}
        idx = 0
        for doc in documents:
            words = doc.split()
            for word in words:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab


if __name__ == "__main__":
    # Example usage
    prep = DataPreparation()
    print("Data Preparation module initialized")
