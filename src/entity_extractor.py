"""Entity Extraction Module for FinanceInsight

This module handles custom entity extraction from financial texts.
"""

from typing import List, Dict, Tuple
import re


class FinancialEntityExtractor:
    """Extracts financial entities from text."""
    
    def __init__(self):
        """Initialize the entity extractor."""
        self.financial_patterns = {
            'COMPANY': r'\b[A-Z][a-zA-Z\s&]+(Inc|Corp|Ltd|LLC|Co)\.?\b',
            'STOCK_TICKER': r'\b[A-Z]{1,5}\b',
            'CURRENCY': r'\$|€|¥|£',
            'PERCENTAGE': r'\b\d+(\.\d+)?%\b',
            'FINANCIAL_METRIC': r'\b(revenue|earnings|profit|loss|dividend|yield)\b'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities from text.
        
        Args:
            text: Input financial text
            
        Returns:
            Dictionary mapping entity types to extracted entities
        """
        entities = {}
        for entity_type, pattern in self.financial_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = matches
        return entities
    
    def extract_user_defined_entities(self, text: str, 
                                     entity_keywords: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Extract user-defined entities.
        
        Args:
            text: Input text
            entity_keywords: Dictionary of entity types and their keywords
            
        Returns:
            Extracted user-defined entities
        """
        results = {}
        for entity_type, keywords in entity_keywords.items():
            results[entity_type] = []
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, text, re.IGNORECASE)
                results[entity_type].extend(matches)
        return results
    
    def extract_financial_events(self, text: str) -> List[Dict[str, str]]:
        """Extract financial events from text.
        
        Args:
            text: Input text
            
        Returns:
            List of detected financial events
        """
        events = []
        event_patterns = {
            'MERGER_ACQUISITION': r'merge|acquisition|acquir',
            'IPO': r'IPO|initial public offering',
            'STOCK_SPLIT': r'stock split',
            'EARNINGS_CALL': r'earnings call|earnings announcement'
        }
        
        for event_type, pattern in event_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                events.append({'type': event_type, 'text': text})
        
        return events
    
    def extract_temporal_expressions(self, text: str) -> List[str]:
        """Extract temporal expressions and dates.
        
        Args:
            text: Input text
            
        Returns:
            List of temporal expressions
        """
        date_pattern = r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}'
        temporal_keywords = r'\b(Q[1-4]|quarter|year|month|fiscal)\s*\d{1,4}?\b'
        
        dates = re.findall(date_pattern, text)
        temporal = re.findall(temporal_keywords, text, re.IGNORECASE)
        
        return dates + temporal


class EntityLinker:
    """Links entities to external databases."""
    
    def __init__(self):
        """Initialize entity linker."""
        pass
    
    def link_to_database(self, entity: str, entity_type: str) -> Dict:
        """Link entity to external database.
        
        Args:
            entity: Entity to link
            entity_type: Type of entity
            
        Returns:
            Entity information from database
        """
        # Placeholder for database linking
        return {'entity': entity, 'type': entity_type, 'id': None}
    
    def enrich_entities(self, entities: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
        """Enrich entities with additional information.
        
        Args:
            entities: Extracted entities
            
        Returns:
            Enriched entities
        """
        enriched = {}
        for entity_type, entity_list in entities.items():
            enriched[entity_type] = []
            for entity in entity_list:
                enriched_entity = self.link_to_database(entity, entity_type)
                enriched[entity_type].append(enriched_entity)
        return enriched


if __name__ == "__main__":
    extractor = FinancialEntityExtractor()
    print("Entity Extractor module initialized")
