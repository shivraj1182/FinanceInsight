"""Data Preparation Module for FinanceInsight

This module handles data collection, preprocessing, tokenization, and preparation
for NER model training with support for multiple file formats.
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)


class DataPreparation:
    """Handles data preparation for financial NER model training.
    
    Supports loading, preprocessing, tokenization, and augmentation of financial text data.
    """

    def __init__(self):
        """Initialize data preparation module."""
        self.financial_entities = [
            'COMPANY', 'STOCK_TICKER', 'REVENUE', 'EARNINGS',
            'MARKET_CAP', 'DATE', 'PERSON', 'LOCATION', 'FINANCIAL_METRIC'
        ]
        self.stop_words = set(stopwords.words('english'))
        self.data = None
        logger.info("DataPreparation module initialized")

    def load_financial_documents(
        self, file_path: str, file_type: str = 'csv'
    ) -> List[str]:
        """Load financial documents from file.

        Args:
            file_path: Path to the file
            file_type: Type of file (csv, txt, json)

        Returns:
            List of documents
        """
        try:
            if file_type == 'csv':
                df = pd.read_csv(file_path)
                documents = df['text'].tolist() if 'text' in df.columns else df.iloc[:, 0].tolist()
            elif file_type == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents = f.readlines()
            elif file_type == 'json':
                df = pd.read_json(file_path)
                documents = df['text'].tolist() if 'text' in df.columns else df.iloc[:, 0].tolist()
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []

    def clean_text(self, text: str) -> str:
        """Clean and normalize financial text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^a-zA-Z0-9\s$€¥£%()\-.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize_text(self, text: str, level: str = 'word') -> List[str]:
        """Tokenize financial text.

        Args:
            text: Input text
            level: Tokenization level (word, sentence)

        Returns:
            List of tokens
        """
        if level == 'sentence':
            return sent_tokenize(text)
        elif level == 'word':
            return word_tokenize(text)
        else:
            raise ValueError(f"Unknown tokenization level: {level}")

    def preprocess_documents(self, documents: List[str]) -> List[Dict[str, any]]:
        """Preprocess documents for NER model.

        Args:
            documents: List of raw documents

        Returns:
            List of processed documents with metadata
        """
        processed_docs = []
        
        for idx, doc in enumerate(documents):
            # Clean text
            cleaned = self.clean_text(doc)
            
            # Tokenize into sentences
            sentences = self.tokenize_text(cleaned, level='sentence')
            
            # Process each sentence
            for sent_idx, sentence in enumerate(sentences):
                # Tokenize words
                tokens = self.tokenize_text(sentence, level='word')
                
                processed_docs.append({
                    'doc_id': idx,
                    'sent_id': sent_idx,
                    'text': sentence,
                    'tokens': tokens,
                    'token_count': len(tokens),
                    'original_length': len(doc)
                })
        
        logger.info(f"Processed {len(processed_docs)} sentences from {len(documents)} documents")
        return processed_docs

    def create_ner_labels(
        self,
        documents: List[Dict],
        entity_annotations: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Create NER labels for documents.

        Args:
            documents: Processed documents
            entity_annotations: Entity annotations (optional)

        Returns:
            Documents with NER labels
        """
        labeled_docs = []
        
        for doc in documents:
            tokens = doc['tokens']
            # Initialize with O (outside) labels
            labels = ['O'] * len(tokens)
            
            if entity_annotations:
                # Apply entity annotations if provided
                for annotation in entity_annotations:
                    if annotation['doc_id'] == doc['doc_id']:
                        # BIO tagging scheme
                        start = annotation['start']
                        end = annotation['end']
                        entity_type = annotation['type']
                        
                        # Find tokens in range and label them
                        char_idx = 0
                        for token_idx, token in enumerate(tokens):
                            token_start = char_idx
                            token_end = char_idx + len(token)
                            char_idx = token_end + 1  # +1 for space
                            
                            if token_start >= start and token_end <= end:
                                if token_idx == 0 or labels[token_idx-1] != f'I-{entity_type}':
                                    labels[token_idx] = f'B-{entity_type}'
                                else:
                                    labels[token_idx] = f'I-{entity_type}'
            
            labeled_docs.append({
                **doc,
                'labels': labels,
                'label_distribution': self._get_label_distribution(labels)
            })
        
        return labeled_docs

    def _get_label_distribution(self, labels: List[str]) -> Dict[str, int]:
        """Get distribution of labels.

        Args:
            labels: List of labels

        Returns:
            Distribution dictionary
        """
        distribution = {}
        for label in labels:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def data_augmentation(self, documents: List[Dict], augmentation_factor: int = 2) -> List[Dict]:
        """Augment training data through various techniques.

        Args:
            documents: Input documents
            augmentation_factor: Factor to augment data

        Returns:
            Augmented documents
        """
        augmented = documents.copy()
        
        for _ in range(augmentation_factor - 1):
            for doc in documents:
                # Simple augmentation: random token shuffling (preserving order for financial safety)
                # For now, just duplicate with different doc_id
                augmented_doc = doc.copy()
                augmented_doc['doc_id'] = augmented_doc['doc_id'] * 1000 + len(augmented)
                augmented.append(augmented_doc)
        
        logger.info(f"Augmented data from {len(documents)} to {len(augmented)} documents")
        return augmented

    def split_data(
        self,
        documents: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train, validation, and test sets.

        Args:
            documents: Input documents
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train, validation, test) documents
        """
        np.random.seed(random_seed)
        indices = np.random.permutation(len(documents))
        
        train_size = int(len(documents) * train_ratio)
        val_size = int(len(documents) * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_data = [documents[i] for i in train_indices]
        val_data = [documents[i] for i in val_indices]
        test_data = [documents[i] for i in test_indices]
        
        logger.info(f"Split data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        return train_data, val_data, test_data

    def save_prepared_data(self, documents: List[Dict], output_path: str):
        """Save prepared data to disk.

        Args:
            documents: Prepared documents
            output_path: Output file path
        """
        df = pd.DataFrame(documents)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(documents)} documents to {output_path}")


if __name__ == "__main__":
    print("Data Preparation module initialized")
