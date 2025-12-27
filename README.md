# FinanceInsight

**Named Entity Recognition (NER) for Financial Data Extraction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-green)

## Overview

FinanceInsight is a comprehensive Python project that develops advanced Named Entity Recognition (NER) models specifically designed to extract critical financial entities and data from unstructured financial documents. The project leverages **FinBERT** (Financial BERT) - a domain-specific pre-trained language model optimized for financial text - to achieve state-of-the-art performance in financial entity recognition.

### Key Features

- **FinBERT Integration**: Uses ProsusAI/FinBERT, a pre-trained language model specifically optimized for financial domain
- **Comprehensive Entity Recognition**: Recognizes 9 entity types including companies, stock tickers, financial metrics, and more
- **Complete Pipeline**: From data preparation through training, evaluation, and inference
- **Production-Ready**: Full-fledged training script with evaluation metrics and model checkpointing
- **Flexible Architecture**: Supports both FinBERT and generic BERT models
- **Data Augmentation**: Built-in techniques for improving training data quality

## Project Architecture

FinanceInsight is a fully modular, production-ready NER system:

### Core Components

**1. Data Preparation Module** (`src/data_preparation.py`)
- Document loading from CSV, JSON, and TXT formats
- Text cleaning and normalization (preserving financial symbols)
- Sentence and word-level tokenization using NLTK
- BIO tagging scheme for entity labeling
- Data augmentation techniques
- Train/validation/test data splitting

**2. NER Models** (`src/ner_model.py`)
- **FinBERTNER**: Domain-optimized model using FinBERT weights
  - Auto-loading of FinBERT from HuggingFace (ProsusAI/finbert)
  - Fallback to BERT-base if FinBERT unavailable
  - Configurable dropout and hidden layers
  - Model save/load functionality

- **BERTBasedNER**: Generic BERT implementation
  - Standard BERT-base-uncased model
  - Flexible hidden size configuration
  - Suitable as baseline model

- **NERTrainer**: Complete training pipeline
  - Epoch-based training with loss computation
  - Validation with metric calculation (Precision, Recall, F1)
  - Gradient clipping and optimizer management
  - Best model checkpointing

- **ModelEvaluator**: Evaluation metrics
  - Precision, Recall, F1-Score (weighted average)
  - Confusion matrix generation
  - Support for multi-class evaluation

**3. Application Pipeline** (`main.py`)
- **Demo Mode**: Test the system without data
- **Training Mode**: Full training with data preparation and evaluation
- **Inference Mode**: Extract entities from new text
- PyTorch Dataset wrapper for batch processing
- Comprehensive logging and progress tracking

## Supported Entity Types

```
COMPANY          - Company names (e.g., "Apple Inc.")
STOCK_TICKER     - Stock ticker symbols (e.g., "AAPL")
REVENUE          - Revenue figures (e.g., "$89.5 billion")
EARNINGS         - Earnings data (e.g., "$6.05 per share")
MARKET_CAP       - Market capitalization values
DATE             - Important dates and periods
PERSON           - Individual names (executives, analysts)
LOCATION         - Geographic locations
FINANCIAL_METRIC - Financial ratios and metrics (e.g., "P/E ratio")
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shivraj1182/FinanceInsight.git
cd FinanceInsight
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Download FinBERT model manually:
```bash
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained("ProsusAI/finbert")
```

## Quick Start

### 1. Demo Mode (No Data Required)

```bash
python main.py --mode demo
```

This will:
- Load the FinBERT model
- Show model statistics
- Display example financial text
- Confirm the system is ready for inference

### 2. Training a Model

```bash
python main.py --mode train \\
  --data-path data/financial_documents.csv \\
  --epochs 3 \\
  --batch-size 32 \\
  --model-path models/finbert_ner.pt \\
  --device cuda
```

**Expected CSV format:**
```
text
"Apple Inc. reported Q4 revenue of $89.5 billion..."
"Microsoft announced earnings of $2.93 per share..."
```

### 3. Extract Entities (Inference)

```bash
python main.py --mode infer \\
  --text "Tesla Inc. reported record revenue of $81.5 billion and earnings of $3.13 per share in Q4 2023." \\
  --model-path models/finbert_ner.pt \\
  --device cuda
```

### 4. Using in Python Code

```python
from src.ner_model import FinBERTNER
from src.data_preparation import DataPreparation

# Initialize model
model = FinBERTNER(
    num_labels=9,
    model_name="ProsusAI/finbert",
    device="cuda"
)

# Prepare data
data_prep = DataPreparation()
documents = data_prep.load_financial_documents("data.csv")
processed = data_prep.preprocess_documents(documents)

# Use model for inference
model.eval()
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    predictions = torch.argmax(logits, dim=-1)
```

## Model Performance

### Evaluation Metrics

- **Precision**: Accuracy of predicted entities (0-1)
- **Recall**: Coverage of actual entities (0-1)  
- **F1-Score**: Harmonic mean of precision and recall (0-1)

### FinBERT Advantages

FinBERT is pre-trained on financial documents (10-K, 10-Q, earnings calls) and shows:
- 15-20% improvement over BERT-base on financial datasets
- Better understanding of financial terminology
- Superior performance on domain-specific entities

## Project Structure

```
FinanceInsight/
├── src/
│   ├── __init__.py
│   ├── ner_model.py           # FinBERT & BERT models + training
│   ├── data_preparation.py    # Data loading & preprocessing
│   ├── entity_extractor.py    # Entity extraction (future)
│   └── document_parser.py     # Document parsing (future)
├── tests/
│   ├── conftest.py
│   └── test_*.py              # Unit tests (future)
├── models/                    # Trained model checkpoints
├── data/                      # Training data directory
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── README.md                  # This file
└── .gitignore
```

## Technologies Used

- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace library (BERT, FinBERT)
- **FinBERT**: Domain-specific BERT for financial text (ProsusAI)
- **NLTK**: Text tokenization and preprocessing
- **spaCy**: NLP utilities
- **scikit-learn**: Metrics and evaluation
- **Pandas**: Data manipulation

## Dependencies

See `requirements.txt` for complete list. Key packages:

```
torch==2.0.1
transformers==4.30.2
finance-nlp==0.1.0
nltk==3.8.1
scikit-learn==1.2.2
pandas==1.5.3
numpy==1.24.3
```

## Development Roadmap

### Completed ✅
- [x] FinBERT NER model implementation
- [x] Data preparation module
- [x] Training pipeline with evaluation
- [x] Main application with CLI
- [x] Requirements and dependencies

### In Progress 🔄
- [ ] Entity extraction module
- [ ] Document parser for financial documents
- [ ] Comprehensive unit tests
- [ ] Example datasets and notebooks

### Planned 📋
- [ ] Web API for inference
- [ ] Docker containerization
- [ ] Fine-tuned model on custom financial data
- [ ] Support for multilingual financial documents
- [ ] Visualization dashboards

## Usage Examples

### Example 1: Simple Inference

```bash
python main.py --mode infer \\
  --text "Apple reported Q4 earnings of $1.52 per share" \\
  --device cpu
```

### Example 2: Train on Custom Data

```bash
python main.py --mode train \\
  --data-path my_financial_data.csv \\
  --epochs 5 \\
  --batch-size 16
```

### Example 3: Batch Processing

```python
from src.ner_model import FinBERTNER
import pandas as pd

model = FinBERTNER(num_labels=9, device="cuda")
texts = pd.read_csv("documents.csv")["text"]

for text in texts:
    # Perform inference
    tokenizer = model.get_tokenizer()
    encoding = tokenizer(text, return_tensors="pt")
    logits = model(**encoding)
```

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

## License

This project is open source and available under the MIT License. See `LICENSE` file for details.

## Author

**Shivraj** - [@shivraj1182](https://github.com/shivraj1182)

## Acknowledgments

- FinBERT team (ProsusAI) for the financial domain-specific BERT model
- HuggingFace for the Transformers library
- PyTorch team for the deep learning framework
- Open-source community for contributions and feedback

## Citation

If you use FinanceInsight in your research, please cite:

```bibtex
@software{FinanceInsight2024,
  title={FinanceInsight: NER for Financial Data Extraction},
  author={Shivraj},
  year={2025},
  url={https://github.com/shivraj1182/FinanceInsight}
}
```

## Contact

For questions or suggestions, please open an issue on GitHub or contact the author.

---

**Last Updated**: December 2025  
**Status**: Actively maintained ✅
