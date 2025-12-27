"""Unit tests for FinanceInsight modules.

Tests for data preparation, NER models, entity extraction, and document parsing.
"""

import pytest
from src.data_preparation import DataPreparation
from src.entity_extractor import FinancialEntityExtractor
from src.document_parser import FinancialDocumentParser


class TestDataPreparation:
    """Tests for DataPreparation module."""

    def setup_method(self):
        """Setup test fixtures."""
        self.data_prep = DataPreparation()
        self.sample_text = "Apple Inc. reported Q4 revenue of $89.5 billion."

    def test_initialization(self):
        """Test module initialization."""
        assert self.data_prep is not None
        assert len(self.data_prep.financial_entities) == 9

    def test_clean_text(self):
        """Test text cleaning."""
        dirty_text = "Apple Inc. http://example.com email@test.com!"
        cleaned = self.data_prep.clean_text(dirty_text)
        assert "http" not in cleaned
        assert "@" not in cleaned

    def test_tokenize_text_word(self):
        """Test word tokenization."""
        tokens = self.data_prep.tokenize_text(self.sample_text, level='word')
        assert len(tokens) > 0
        assert 'Apple' in tokens or 'apple' in tokens.lower()

    def test_tokenize_text_sentence(self):
        """Test sentence tokenization."""
        multi_sentence = "Apple Inc. reported earnings. Revenue grew significantly."
        sentences = self.data_prep.tokenize_text(multi_sentence, level='sentence')
        assert len(sentences) >= 2

    def test_preprocess_documents(self):
        """Test document preprocessing."""
        docs = [self.sample_text]
        processed = self.data_prep.preprocess_documents(docs)
        assert len(processed) > 0
        assert 'tokens' in processed[0]
        assert 'text' in processed[0]

    def test_split_data(self):
        """Test data splitting."""
        docs = [{'text': f'Document {i}'} for i in range(100)]
        train, val, test = self.data_prep.split_data(docs)
        assert len(train) + len(val) + len(test) == 100
        assert len(train) > len(val) > len(test)


class TestFinancialEntityExtractor:
    """Tests for FinancialEntityExtractor module."""

    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = FinancialEntityExtractor()
        self.financial_text = "Apple Inc. reported Q4 2023 revenue of $89.5 billion. Stock ticker AAPL."

    def test_initialization(self):
        """Test extractor initialization."""
        assert self.extractor is not None
        assert 'COMPANY' in self.extractor.patterns
        assert 'STOCK_TICKER' in self.extractor.patterns

    def test_extract_entities(self):
        """Test entity extraction."""
        entities = self.extractor.extract_entities(self.financial_text)
        assert len(entities) > 0
        # Check that at least one entity was found
        entity_types = [e.entity_type for e in entities]
        assert len(entity_types) > 0

    def test_extract_financial_metrics(self):
        """Test financial metrics extraction."""
        metrics = self.extractor.extract_financial_metrics(self.financial_text)
        assert 'revenues' in metrics
        assert 'dates' in metrics
        assert 'currencies' in metrics

    def test_extract_currencies(self):
        """Test currency extraction."""
        text = "The company earned $100 million and €50 million."
        currencies = self.extractor._extract_currencies(text)
        assert len(currencies) >= 2
        values = [c['currency'] for c in currencies]
        assert '$' in values or 'USD' in values

    def test_extract_percentages(self):
        """Test percentage extraction."""
        text = "Revenue grew 25.5% while costs increased 10%."
        percentages = self.extractor._extract_percentages(text)
        assert len(percentages) >= 2


class TestFinancialDocumentParser:
    """Tests for FinancialDocumentParser module."""

    def setup_method(self):
        """Setup test fixtures."""
        self.parser = FinancialDocumentParser()
        self.sample_10k = """10-K
Item 1 BUSINESS DESCRIPTION
The company operates in technology.
Item 7 MANAGEMENT DISCUSSION AND ANALYSIS
Revenue increased 20%.
Item 8 FINANCIAL STATEMENTS
Balance Sheet
Assets: $1 billion
"""

    def test_initialization(self):
        """Test parser initialization."""
        assert self.parser is not None
        assert 'BUSINESS_DESCRIPTION' in self.parser.section_patterns
        assert 'MDA' in self.parser.section_patterns

    def test_detect_document_type(self):
        """Test document type detection."""
        doc_type = self.parser._detect_document_type(self.sample_10k)
        assert doc_type == '10-K'

    def test_parse_document(self):
        """Test document parsing."""
        result = self.parser.parse_document(self.sample_10k)
        assert 'document_type' in result
        assert 'sections' in result
        assert result['document_type'] == '10-K'

    def test_extract_financial_data(self):
        """Test financial data extraction."""
        data = self.parser.extract_financial_data(self.sample_10k)
        assert 'financial_figures' in data
        assert 'dates' in data

    def test_extract_mda_section(self):
        """Test MDA section extraction."""
        mda = self.parser.extract_mda_section(self.sample_10k)
        assert mda is not None
        assert 'Revenue' in mda


class TestIntegration:
    """Integration tests for multiple modules."""

    def setup_method(self):
        """Setup test fixtures."""
        self.data_prep = DataPreparation()
        self.extractor = FinancialEntityExtractor()
        self.parser = FinancialDocumentParser()

    def test_end_to_end_pipeline(self):
        """Test complete pipeline."""
        raw_text = "Apple Inc. reported Q4 2023 revenue of $89.5 billion."
        
        # Data preparation
        cleaned = self.data_prep.clean_text(raw_text)
        assert len(cleaned) > 0
        
        # Entity extraction
        entities = self.extractor.extract_entities(cleaned)
        assert len(entities) > 0
        
        # Metrics extraction
        metrics = self.extractor.extract_financial_metrics(cleaned)
        assert len(metrics) > 0

    def test_document_parsing_pipeline(self):
        """Test document parsing pipeline."""
        document = "10-K Report\nItem 1 BUSINESS\nWe operate globally.\nItem 8 FINANCIAL STATEMENTS\nBalance Sheet:\nAssets: $1B"
        
        parsed = self.parser.parse_document(document)
        assert parsed['document_type'] == '10-K'
        assert len(parsed['sections']) > 0
        
        data = self.parser.extract_financial_data(document)
        assert len(data['financial_figures']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
