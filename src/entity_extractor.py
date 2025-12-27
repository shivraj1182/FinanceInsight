"""Entity Extraction Module for FinanceInsight

This module handles comprehensive entity extraction from financial texts,
including pattern-based extraction, NER model inference, and entity linking.
"""

import re
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FinancialEntity:
    """Represents an extracted financial entity."""
    entity_type: str
    text: str
    start_pos: int
    end_pos: int
    confidence: float
    source: str  # 'pattern' or 'ner'
    context: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'type': self.entity_type,
            'text': self.text,
            'start': self.start_pos,
            'end': self.end_pos,
            'confidence': self.confidence,
            'source': self.source,
            'context': self.context
        }


class FinancialEntityExtractor:
    """Extracts financial entities from text using patterns and NER."""

    def __init__(self, use_ner_model=False, ner_model=None):
        """Initialize the entity extractor.
        
        Args:
            use_ner_model: Whether to use NER model for extraction
            ner_model: Pre-trained NER model instance
        """
        self.use_ner_model = use_ner_model
        self.ner_model = ner_model
        
        # Define financial patterns
        self.patterns = {
            'COMPANY': [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Ltd|LLC|Co|Company|Corporation|Incorporated))\b',
                r'\b([A-Z]+\s+Inc|Corp|Co|Ltd)\b',
            ],
            'STOCK_TICKER': [
                r'\b([A-Z]{1,5})\b(?=\s|:|$)',  # Standalone ticker
                r'\(([A-Z]{1,5})\)',  # Ticker in parentheses
                r'ticker:\s*([A-Z]{1,5})',  # Explicit ticker label
            ],
            'REVENUE': [
                r'revenue[\s:]*\$?([\d,]+(?:\.\d{1,2})?\s*(?:million|billion|trillion|M|B|T|k|K))',
                r'sales[\s:]*\$?([\d,]+(?:\.\d{1,2})?\s*(?:million|billion|trillion|M|B|T))',
            ],
            'EARNINGS': [
                r'earnings?[\s:]*\$?([\d,]+(?:\.\d{1,2})?\s*(?:million|billion|M|B|trillion)?)',
                r'EPS[\s:]*\$?([\d,]+\.\d{1,2})',
                r'per share[\s:]*\$?([\d,]+\.\d{1,2})',
            ],
            'MARKET_CAP': [
                r'market cap[\s:]*\$?([\d,]+(?:\.\d{1,2})?\s*(?:million|billion|trillion|M|B|T))',
                r'valuation[\s:]*\$?([\d,]+(?:\.\d{1,2})?\s*(?:million|billion|trillion|M|B|T))',
            ],
            'DATE': [
                r'\b(Q[1-4]\s*[12]\d{3})',  # Q1 2023
                r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',  # 12/25/2023
                r'\b(\d{4}-\d{1,2}-\d{1,2})\b',  # 2023-12-25
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            ],
            'CURRENCY': [
                r'(\$|USD|€|EUR|¥|JPY|£|GBP)\s*([\d,]+(?:\.\d{2})?)',
            ],
            'FINANCIAL_METRIC': [
                r'(P/E|PE|price-to-earnings|dividend yield|ROI|ROE|debt-to-equity)\s*[\d.%]*',
                r'(EBITDA|FCF|OCF)\s*[\d.KMB]*',
            ],
        }
        
        logger.info("FinancialEntityExtractor initialized")

    def extract_entities(
        self, text: str, use_patterns: bool = True, use_ner: bool = False
    ) -> List[FinancialEntity]:
        """Extract financial entities from text.
        
        Args:
            text: Input financial text
            use_patterns: Use regex patterns for extraction
            use_ner: Use NER model if available
            
        Returns:
            List of extracted financial entities
        """
        entities = []
        
        # Pattern-based extraction
        if use_patterns:
            entities.extend(self._extract_by_patterns(text))
        
        # NER-based extraction
        if use_ner and self.use_ner_model and self.ner_model:
            entities.extend(self._extract_by_ner(text))
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.start_pos)
        
        logger.info(f"Extracted {len(entities)} entities from text")
        return entities

    def _extract_by_patterns(self, text: str) -> List[FinancialEntity]:
        """Extract entities using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity_text = match.group(1) if match.lastindex else match.group(0)
                    
                    entity = FinancialEntity(
                        entity_type=entity_type,
                        text=entity_text.strip(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.85,  # Pattern-based confidence
                        source='pattern',
                        context=text[max(0, match.start()-50):min(len(text), match.end()+50)]
                    )
                    entities.append(entity)
        
        return entities

    def _extract_by_ner(self, text: str) -> List[FinancialEntity]:
        """Extract entities using NER model.
        
        Args:
            text: Input text
            
        Returns:
            List of NER-extracted entities
        """
        entities = []
        
        if not self.ner_model:
            return entities
        
        try:
            # Get NER predictions (requires model and tokenizer)
            tokenizer = self.ner_model.get_tokenizer()
            
            # Simple tokenization for demo
            tokens = text.split()
            
            # In production, would use actual NER model inference
            # This is a placeholder for integration
            logger.debug(f"NER extraction called for {len(tokens)} tokens")
            
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
        
        return entities

    def _deduplicate_entities(
        self, entities: List[FinancialEntity]
    ) -> List[FinancialEntity]:
        """Remove duplicate entities, keeping highest confidence.
        
        Args:
            entities: List of entities
            
        Returns:
            Deduplicated list
        """
        seen = {}
        
        for entity in entities:
            key = (entity.start_pos, entity.end_pos, entity.entity_type)
            
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        
        return list(seen.values())

    def extract_financial_metrics(
        self, text: str
    ) -> Dict[str, List[Dict]]:
        """Extract specific financial metrics and their values.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of metric types and their values
        """
        metrics = {
            'revenues': self._extract_metric_value(text, 'revenue', ['revenue', 'sales']),
            'earnings': self._extract_metric_value(text, 'earnings', ['earnings', 'eps', 'net income']),
            'growth_rates': self._extract_percentages(text),
            'dates': self._extract_dates(text),
            'currencies': self._extract_currencies(text),
        }
        
        return metrics

    def _extract_metric_value(
        self, text: str, metric_type: str, keywords: List[str]
    ) -> List[Dict]:
        """Extract specific metric values from text.
        
        Args:
            text: Input text
            metric_type: Type of metric
            keywords: Keywords to search for
            
        Returns:
            List of metric values
        """
        results = []
        
        for keyword in keywords:
            pattern = rf'{keyword}[\s:]*\$?([\d,]+(?:\.\d{{1,2}})?\s*(?:million|billion|trillion|M|B|T)?)'
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                results.append({
                    'metric': metric_type,
                    'keyword': keyword,
                    'value': match.group(1).strip(),
                    'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
                })
        
        return results

    def _extract_percentages(self, text: str) -> List[Dict]:
        """Extract percentage values (growth rates, margins, etc.).
        
        Args:
            text: Input text
            
        Returns:
            List of percentage values
        """
        percentages = []
        
        for match in re.finditer(r'([\d.]+)%', text):
            percentages.append({
                'value': match.group(1),
                'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
            })
        
        return percentages

    def _extract_dates(self, text: str) -> List[str]:
        """Extract date patterns.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted dates
        """
        dates = []
        date_pattern = r'\b(Q[1-4]\s*[12]\d{3}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2})\b'
        
        for match in re.finditer(date_pattern, text):
            dates.append(match.group(1))
        
        return dates

    def _extract_currencies(self, text: str) -> List[Dict]:
        """Extract currency amounts.
        
        Args:
            text: Input text
            
        Returns:
            List of currency amounts
        """
        currencies = []
        currency_pattern = r'(\$|USD|€|EUR|¥|JPY|£|GBP)\s*([\d,]+(?:\.\d{2})?)'
        
        for match in re.finditer(currency_pattern, text):
            currencies.append({
                'currency': match.group(1),
                'amount': match.group(2)
            })
        
        return currencies

    def extract_key_entities(
        self, text: str
    ) -> Dict[str, List[str]]:
        """Extract key financial entities and group by type.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types to list of texts
        """
        entities = self.extract_entities(text)
        
        grouped = {}
        for entity in entities:
            if entity.entity_type not in grouped:
                grouped[entity.entity_type] = []
            grouped[entity.entity_type].append(entity.text)
        
        return grouped


if __name__ == "__main__":
    print("Entity Extraction module initialized")
