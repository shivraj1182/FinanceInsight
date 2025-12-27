# FinanceInsight

**Named Entity Recognition (NER) for Financial Data Extraction**

FinanceInsight is a comprehensive Python project that develops advanced NER models specifically designed to extract critical financial entities and data from unstructured financial documents. The project focuses on identifying and extracting key financial information such as company names, stock prices, revenue figures, earnings data, market capitalizations, and financial events from various financial documents.

## Project Overview

This project implements state-of-the-art transformer-based NER models (BERT, FinBERT) to effectively extract structured financial data from unstructured text sources including:
- Earnings reports
- SEC filings (10-K, 10-Q)
- Financial news articles
- Analyst reports
- Annual reports

## Features

### Core Components

1. **Data Preparation Module** (`src/data_preparation.py`)
   - Document loading and preprocessing
   - Tokenization and lemmatization
   - Vocabulary creation
   - Data augmentation for training
   - Support for CSV and TXT formats

2. **NER Models** (`src/ner_model.py`)
   - BERT-based NER for financial entities
   - FinBERT-based NER (domain-specific)
   - Training pipeline with loss computation
   - Model evaluation and metric calculation
   - Confusion matrix generation

3. **Entity Extraction** (`src/entity_extractor.py`)
   - Financial entity recognition (companies, stock tickers, currencies)
   - User-defined entity extraction
   - Financial event detection (M&A, IPO, stock splits)
   - Temporal expression extraction
   - Entity linking to external databases
   - Entity enrichment with additional information

4. **Document Parsing** (`src/document_parser.py`)
   - Financial document segmentation
   - Section identification (MDA, Financial Statements, Risk Factors)
   - Table extraction and parsing
   - Balance sheet, income statement, and cash flow parsing
   - Document structure analysis

## Project Structure

```
FinanceInsight/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── data_preparation.py         # Data loading and preprocessing
│   ├── ner_model.py               # NER model implementations
│   ├── entity_extractor.py        # Entity extraction logic
│   └── document_parser.py         # Document parsing utilities
├── main.py                         # Entry point
├── requirements.txt                # Project dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shivraj1182/FinanceInsight.git
cd FinanceInsight
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Running the Project

```bash
python main.py
```

### Using Individual Modules

```python
from src.data_preparation import DataPreparation
from src.entity_extractor import FinancialEntityExtractor
from src.document_parser import FinancialDocumentParser

# Initialize modules
data_prep = DataPreparation()
extractor = FinancialEntityExtractor()
parser = FinancialDocumentParser()

# Extract entities from text
text = "Apple Inc. reported Q4 revenue of $89.5 billion."
entities = extractor.extract_entities(text)
print(entities)
```

## Model Performance

### Evaluation Metrics
- **Precision**: Accuracy of predicted entities
- **Recall**: Coverage of actual entities
- **F1-Score**: Harmonic mean of precision and recall

### Supported Entity Types
- COMPANY: Company names
- STOCK_TICKER: Stock ticker symbols
- REVENUE: Revenue figures
- EARNINGS: Earnings data
- MARKET_CAP: Market capitalization
- DATE: Important dates and periods
- PERSON: Individual names
- LOCATION: Geographic locations
- FINANCIAL_METRIC: Financial ratios and metrics

## Technologies Used

- **PyTorch**: Deep learning framework
- **Transformers**: Pre-trained language models (BERT, FinBERT)
- **NLTK**: Natural Language Toolkit
- **spaCy**: NLP library
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities

## Project Timeline

### Milestone 1 (Weeks 1-2): Data Preparation
- Data collection and preprocessing
- Exploratory Data Analysis (EDA)
- Data augmentation

### Milestone 2 (Weeks 3-4): NER Model Development
- Model selection and evaluation
- Training with transfer learning
- Error analysis and refinement

### Milestone 3 (Weeks 5-6): Custom Entity Extraction
- User-defined entity extraction
- Financial event detection
- Database integration

### Milestone 4 (Weeks 7-8): Document Parsing & Deployment
- Document segmentation
- Table parsing
- Production deployment

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

## License

This project is open source and available under the MIT License.

## Author

**shivraj1182** - GitHub: [@shivraj1182](https://github.com/shivraj1182)

## Acknowledgments

- Built with transformer-based NER models
- Inspired by financial NLP research
- Special thanks to the open-source community

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Last Updated**: December 27, 2025
