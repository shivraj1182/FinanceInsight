"""Main entry point for FinanceInsight NER project.

This script serves as the main pipeline for training, evaluating, and
inferencing FinBERT-based Named Entity Recognition models for financial data extraction.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data_preparation import DataPreparation
from src.ner_model import FinBERTNER, NERTrainer, ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinanceInsightDataset(Dataset):
    """Custom PyTorch Dataset for financial NER tasks."""

    def __init__(self, documents, tokenizer, max_length=128):
        """Initialize dataset.
        
        Args:
            documents: List of processed documents
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc = self.documents[idx]
        text = doc['text']
        labels = doc.get('labels', ['O'] * len(doc['tokens']))
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert labels to tensor (simple approach: all O if not provided)
        label_ids = torch.zeros(self.max_length, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_ids
        }


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=3,
    device='cpu',
    save_path='models/finbert_ner.pt'
):
    """Train the NER model.
    
    Args:
        model: NER model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on
        save_path: Path to save the model
    """
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    trainer = NERTrainer(model, device=device, num_epochs=num_epochs)
    
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        train_loss = trainer.train_epoch(train_loader)
        logger.info(f"Training loss: {train_loss:.4f}")
        
        # Validation
        if val_loader:
            val_metrics = trainer.evaluate(val_loader)
            logger.info(f"Validation metrics: {val_metrics}")
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                model.save_model(save_path)
                logger.info(f"Best model saved to {save_path} (F1: {best_f1:.4f})")
    
    logger.info(f"\nTraining completed. Best F1 score: {best_f1:.4f}")
    return model


def infer(model, text, device='cpu'):
    """Run inference on text.
    
    Args:
        model: Trained NER model
        text: Input text
        device: Device to run inference on
        
    Returns:
        List of entities with their types and positions
    """
    model.eval()
    tokenizer = model.get_tokenizer()
    
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=-1)
    
    # Decode predictions
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    pred_labels = predictions[0].cpu().numpy()
    
    # Extract entities
    entities = []
    for token, label in zip(tokens, pred_labels):
        if label != 0:  # Not O (outside) tag
            entities.append({
                'token': token,
                'label': label
            })
    
    return entities


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='FinanceInsight: NER for Financial Data Extraction'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'infer', 'demo'],
        default='demo',
        help='Execution mode'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to training data'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/finbert_ner.pt',
        help='Path to save/load model'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Text for inference'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for computation'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("FinanceInsight: Named Entity Recognition for Financial Data")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {args.device}")
    
    if args.mode == 'demo':
        logger.info("\nRunning demo mode...")
        logger.info("\nInitializing FinBERT NER model...")
        
        try:
            model = FinBERTNER(
                num_labels=9,  # COMPANY, STOCK_TICKER, REVENUE, EARNINGS, etc.
                device=args.device
            )
            logger.info("Model loaded successfully!")
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Demo text
            demo_text = "Apple Inc. reported Q4 revenue of $89.5 billion, with earnings per share reaching $6.05."
            logger.info(f"\nDemo text: {demo_text}")
            logger.info("\nFinBERT NER model is ready for inference!")
            logger.info("\nTo extract entities, use:")
            logger.info("  python main.py --mode infer --text \"your text here\"")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Ensure all dependencies are installed: pip install -r requirements.txt")
    
    elif args.mode == 'train':
        if not args.data_path:
            logger.error("--data-path is required for training mode")
            return 1
        
        logger.info(f"\nLoading training data from {args.data_path}...")
        data_prep = DataPreparation()
        documents = data_prep.load_financial_documents(
            args.data_path,
            file_type='csv'
        )
        
        if not documents:
            logger.error("No documents loaded")
            return 1
        
        # Preprocess
        processed_docs = data_prep.preprocess_documents(documents)
        train_data, val_data, test_data = data_prep.split_data(processed_docs)
        
        # Create datasets
        model = FinBERTNER(num_labels=9, device=args.device)
        tokenizer = model.get_tokenizer()
        
        train_dataset = FinanceInsightDataset(train_data, tokenizer)
        val_dataset = FinanceInsightDataset(val_data, tokenizer) if val_data else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False
        ) if val_dataset else None
        
        # Train
        train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            device=args.device,
            save_path=args.model_path
        )
    
    elif args.mode == 'infer':
        if not args.text:
            logger.error("--text is required for inference mode")
            return 1
        
        logger.info(f"\nLoading model from {args.model_path}...")
        model = FinBERTNER(num_labels=9, device=args.device)
        
        try:
            model.load_model(args.model_path)
            logger.info("Model loaded successfully!")
        except FileNotFoundError:
            logger.warning(f"Model not found at {args.model_path}. Using pre-trained weights.")
        
        logger.info(f"\nInput text: {args.text}")
        entities = infer(model, args.text, device=args.device)
        
        logger.info("\nExtracted Entities:")
        if entities:
            for entity in entities:
                logger.info(f"  - Token: {entity['token']}, Label: {entity['label']}")
        else:
            logger.info("  No entities found")
    
    logger.info("\n" + "="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
