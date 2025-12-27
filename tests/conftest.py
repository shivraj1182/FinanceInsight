tests/conftest.py"""Pytest configuration and fixtures for FinanceInsight tests."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def sample_text():
    """Fixture providing sample financial text."""
    return """Apple Inc. reported Q4 revenue of $89.5 billion. The company's earnings per share 
    increased to $7.41. Market capitalization reached $3.2 trillion on January 15, 2024."""


@pytest.fixture
def financial_documents():
    """Fixture providing sample financial documents."""
    return [
        "Apple Inc. reported Q4 2023 revenue of $89.5 billion",
        "Microsoft Corporation earnings call on January 30, 2024",
        "Tesla Inc. stock split announced for Q2 2024",
        "Google (Alphabet) parent company merger discussion"
    ]


@pytest.fixture
def mock_model():
    """Fixture providing a mocked NER model."""
    model = MagicMock()
    model.forward.return_value = [[0.1, 0.9], [0.2, 0.8]]
    return model


@pytest.fixture
def sample_entities():
    """Fixture providing sample extracted entities."""
    return {
        'COMPANY': ['Apple Inc.', 'Microsoft Corporation', 'Tesla Inc.'],
        'STOCK_TICKER': ['AAPL', 'MSFT', 'TSLA'],
        'CURRENCY': ['$89.5 billion', '$7.41'],
        'PERCENTAGE': ['25%', '15.5%'],
        'DATE': ['January 15, 2024', 'Q4 2023']
    }


@pytest.fixture
def empty_text():
    """Fixture providing empty text."""
    return ""


@pytest.fixture
def special_characters_text():
    """Fixture providing text with special characters."""
    return """Company@123 Inc. reported revenue of $1,000,000 & earnings of €500K.
    P/E ratio: 25.5x, EPS: $5.25"""


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset imports before each test."""
    yield


class TestConfig:
    """Test configuration class."""
    TESTING = True
    DEBUG = False
    LOG_LEVEL = 'DEBUG'


@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return TestConfig()
