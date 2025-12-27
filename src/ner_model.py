"""NER Model Module for FinanceInsight

This module contains the core NER models using transformers, with specific
implementation for FinBERT (domain-specific BERT for financial texts).
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertTokenizer,
    BertModel,
)
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix as sk_confusion_matrix
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FinBERT Model Configuration
FINBERT_MODEL_NAME = "ProsusAI/finbert"
BERT_MODEL_NAME = "bert-base-uncased"


class FinBERTNER(nn.Module):
    """FinBERT-based NER model specifically optimized for financial text.
    
    This model uses FinBERT (Financial BERT) which is a pre-trained NLP model
    trained on financial text and is particularly effective for financial entity
    recognition tasks.
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = FINBERT_MODEL_NAME,
        dropout_rate: float = 0.2,
        device: str = "cpu",
    ):
        """Initialize FinBERT-based NER model.

        Args:
            num_labels: Number of entity labels
            model_name: HuggingFace model identifier (default: FinBERT)
            dropout_rate: Dropout rate for regularization
            device: Device to run model on (cpu/cuda)
        """
        super().__init__()
        self.num_labels = num_labels
        self.device = device
        self.model_name = model_name

        try:
            # Load FinBERT tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.warning(
                f"Could not load {model_name}: {e}. Falling back to BERT-base."
            )
            self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
            self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)

        # Get hidden size from the loaded model
        self.hidden_size = self.bert.config.hidden_size

        # Classification layers
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for valid tokens
            token_type_ids: Segment IDs for BERT

        Returns:
            Logits for each token (batch_size, seq_length, num_labels)
        """
        # Get BERT output
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        # Use the last hidden state for classification
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        # Apply dropout and classify
        dropout_output = self.dropout(last_hidden_state)
        logits = self.classifier(dropout_output)  # (batch_size, seq_length, num_labels)

        return logits

    def get_tokenizer(self):
        """Get the tokenizer instance."""
        return self.tokenizer

    def save_model(self, output_path: str):
        """Save the model to disk.

        Args:
            output_path: Path to save the model
        """
        torch.save(self.state_dict(), output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")

    def load_model(self, model_path: str):
        """Load the model from disk.

        Args:
            model_path: Path to load the model from
        """
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"Model loaded from {model_path}")


class BERTBasedNER(nn.Module):
    """BERT-based NER model for financial entity extraction.
    
    A more generic BERT implementation that can be used as a baseline
    or when FinBERT is not available.
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = BERT_MODEL_NAME,
        hidden_size: int = 768,
        dropout_rate: float = 0.1,
        device: str = "cpu",
    ):
        """Initialize BERT-based NER model.

        Args:
            num_labels: Number of entity labels
            model_name: HuggingFace model identifier
            hidden_size: Hidden size of the model
            dropout_rate: Dropout rate
            device: Device to run on
        """
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.device = device

        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

        # Classification layers
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Logits for entity classification
        """
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        encoder_output = outputs.last_hidden_state
        dropout_output = self.dropout(encoder_output)
        logits = self.classifier(dropout_output)
        return logits


class NERTrainer:
    """Trainer class for NER models with training, validation, and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 2e-5,
        device: str = "cpu",
        num_epochs: int = 3,
    ):
        """Initialize NER trainer.

        Args:
            model: NER model to train
            learning_rate: Learning rate for optimizer
            device: Device to train on
            num_epochs: Number of training epochs
        """
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.training_losses = []
        self.validation_metrics = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Execute one training epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            # Compute loss (reshape for sequence labeling)
            batch_size, seq_length, num_labels = logits.shape
            loss = self.loss_fn(
                logits.view(-1, num_labels), labels.view(-1)
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.training_losses.append(avg_loss)
        logger.info(f"Epoch average loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(
        self, val_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on validation data.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = ModelEvaluator.calculate_metrics(
            all_predictions, all_labels
        )
        self.validation_metrics.append(metrics)
        logger.info(f"Validation metrics: {metrics}")
        return metrics


class ModelEvaluator:
    """Evaluates NER model performance with various metrics."""

    @staticmethod
    def calculate_metrics(
        predictions: List, labels: List
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score.

        Args:
            predictions: Model predictions
            labels: Ground truth labels

        Returns:
            Dictionary with metric values
        """
        # Flatten predictions and labels
        flat_predictions = []
        flat_labels = []

        for pred, label in zip(predictions, labels):
            if isinstance(pred, (list, np.ndarray)):
                flat_predictions.extend(pred)
                flat_labels.extend(label)
            else:
                flat_predictions.append(pred)
                flat_labels.append(label)

        if len(flat_predictions) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_labels,
            flat_predictions,
            average="weighted",
            zero_division=0,
        )

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    @staticmethod
    def confusion_matrix(
        predictions: List[int], labels: List[int]
    ) -> np.ndarray:
        """Generate confusion matrix.

        Args:
            predictions: Model predictions
            labels: Ground truth labels

        Returns:
            Confusion matrix
        """
        return sk_confusion_matrix(labels, predictions)


if __name__ == "__main__":
    print("NER Model module initialized")
    print(f"FinBERT model: {FINBERT_MODEL_NAME}")
    print(f"BERT model: {BERT_MODEL_NAME}")
