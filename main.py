"""Main entry point for FinanceInsight NER project."""

import sys
from src.data_preparation import DataPreparation
from src.ner_model import BERTBasedNER
from src.entity_extractor import FinancialEntityExtractor
from src.document_parser import FinancialDocumentParser


def main():
    """Main function to run the FinanceInsight pipeline."""
    print("="*60)
    print("FinanceInsight: NER for Financial Data Extraction")
    print("="*60)
    
    # Initialize modules
    print("\nInitializing modules...")
    data_prep = DataPreparation()
    entity_extractor = FinancialEntityExtractor()
    doc_parser = FinancialDocumentParser()
    
    print("All modules initialized successfully!")
    print("\nAvailable modules:")
    print("  - DataPreparation: For data loading and preprocessing")
    print("  - FinancialEntityExtractor: For entity extraction")
    print("  - FinancialDocumentParser: For document parsing")
    print("  - BERTBasedNER: For NER model training")
    print("\nTo get started, see README.md for usage examples.")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
