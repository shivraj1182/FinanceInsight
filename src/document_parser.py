"""Document Parsing Module for FinanceInsight

Handles parsing and segmentation of financial documents including SEC filings
(10-K, 10-Q), earnings reports, and financial statements.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Represents a section of a financial document."""
    section_type: str
    title: str
    content: str
    start_line: int
    end_line: int
    subsections: List['DocumentSection'] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'type': self.section_type,
            'title': self.title,
            'content': self.content[:200] + '...',  # Truncate for summary
            'start_line': self.start_line,
            'end_line': self.end_line,
            'subsections': len(self.subsections) if self.subsections else 0
        }


class FinancialDocumentParser:
    """Parses financial documents including 10-K, 10-Q, earnings reports."""

    def __init__(self):
        """Initialize the document parser."""
        # Section patterns for 10-K/10-Q
        self.section_patterns = {
            'BUSINESS_DESCRIPTION': r'(?:Item\s+1|BUSINESS DESCRIPTION)',
            'RISK_FACTORS': r'(?:Item\s+1A|RISK FACTORS)',
            'FINANCIAL_STATEMENTS': r'(?:Item\s+8|FINANCIAL STATEMENTS)',
            'MDA': r'(?:Item\s+7|MANAGEMENT.*DISCUSSION)',
            'BALANCE_SHEET': r'(?:CONSOLIDATED|BALANCE SHEET)',
            'INCOME_STATEMENT': r'(?:INCOME STATEMENT|STATEMENT OF EARNINGS)',
            'CASH_FLOW': r'(?:CASH FLOW|STATEMENT OF CASH FLOWS)',
            'MANAGEMENT': r'(?:Item\s+10|MANAGEMENT|EXECUTIVE OFFICERS)',
        }
        
        logger.info("FinancialDocumentParser initialized")

    def parse_document(self, document_text: str, doc_type: str = 'auto') -> Dict:
        """Parse a financial document.
        
        Args:
            document_text: Full document text
            doc_type: Type of document ('10-K', '10-Q', 'earnings', auto')
            
        Returns:
            Dictionary with parsed document structure
        """
        # Detect document type if not provided
        if doc_type == 'auto':
            doc_type = self._detect_document_type(document_text)
        
        lines = document_text.split('\n')
        sections = self._identify_sections(lines, doc_type)
        
        parsed = {
            'document_type': doc_type,
            'total_lines': len(lines),
            'sections': [s.to_dict() for s in sections],
            'section_objects': sections,
            'tables': self._extract_tables(document_text),
        }
        
        logger.info(f"Parsed {len(sections)} sections from {doc_type}")
        return parsed

    def _detect_document_type(self, text: str) -> str:
        """Detect the type of financial document.
        
        Args:
            text: Document text
            
        Returns:
            Document type string
        """
        text_upper = text.upper()[:5000]  # Check first 5000 chars
        
        if '10-K' in text_upper or 'ANNUAL REPORT' in text_upper:
            return '10-K'
        elif '10-Q' in text_upper or 'QUARTERLY REPORT' in text_upper:
            return '10-Q'
        elif 'EARNINGS' in text_upper or 'FINANCIAL RESULTS' in text_upper:
            return 'earnings'
        else:
            return 'financial_document'

    def _identify_sections(self, lines: List[str], doc_type: str) -> List[DocumentSection]:
        """Identify main sections in the document.
        
        Args:
            lines: List of document lines
            doc_type: Type of document
            
        Returns:
            List of identified sections
        """
        sections = []
        current_section = None
        section_start = 0
        
        for line_num, line in enumerate(lines):
            # Check if line matches any section pattern
            for section_type, pattern in self.section_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    # Save previous section
                    if current_section:
                        content = '\n'.join(lines[section_start:line_num])
                        section = DocumentSection(
                            section_type=current_section,
                            title=lines[section_start] if section_start < len(lines) else '',
                            content=content,
                            start_line=section_start,
                            end_line=line_num
                        )
                        sections.append(section)
                    
                    # Start new section
                    current_section = section_type
                    section_start = line_num
                    break
        
        # Add final section
        if current_section:
            content = '\n'.join(lines[section_start:])
            section = DocumentSection(
                section_type=current_section,
                title=lines[section_start] if section_start < len(lines) else '',
                content=content,
                start_line=section_start,
                end_line=len(lines)
            )
            sections.append(section)
        
        return sections

    def extract_financial_data(
        self, document_text: str
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Extract key financial data from document.
        
        Args:
            document_text: Document text
            
        Returns:
            Dictionary of extracted financial data
        """
        data = {
            'companies': self._extract_company_info(document_text),
            'financial_figures': self._extract_figures(document_text),
            'dates': self._extract_dates(document_text),
            'metrics': self._extract_metrics(document_text),
        }
        
        return data

    def _extract_company_info(self, text: str) -> List[Dict]:
        """Extract company information.
        
        Args:
            text: Document text
            
        Returns:
            List of company information
        """
        info = []
        
        # Look for common company info patterns
        patterns = {
            'cik': r'CIK[\s:]*([0-9]{10})',
            'ticker': r'TICKER[\s:]*([A-Z]{1,5})',
            'company_name': r'Company Name[\s:]*([^\n]+)',
        }
        
        for key, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                info.append({'type': key, 'value': match.group(1)})
        
        return info

    def _extract_figures(self, text: str) -> List[Dict]:
        """Extract financial figures.
        
        Args:
            text: Document text
            
        Returns:
            List of financial figures
        """
        figures = []
        
        # Match currency amounts
        pattern = r'(\$[\d,]+(?:\.\d{2})?|[\d,]+(?:\.\d{2})?\s*(?:million|billion|thousand|M|B|K))'
        
        for match in re.finditer(pattern, text):
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 100)
            context = text[context_start:context_end]
            
            figures.append({
                'value': match.group(1),
                'context': context.replace('\n', ' ')[:200]
            })
        
        return figures[:50]  # Limit to first 50

    def _extract_dates(self, text: str) -> List[str]:
        """Extract date references.
        
        Args:
            text: Document text
            
        Returns:
            List of dates
        """
        dates = []
        date_pattern = r'\b(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-\d{1,2}-\d{4}|Q[1-4]\s*\d{4}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b'
        
        for match in re.finditer(date_pattern, text):
            dates.append(match.group(1))
        
        return list(set(dates))  # Remove duplicates

    def _extract_metrics(self, text: str) -> Dict[str, List[Dict]]:
        """Extract financial metrics.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'earnings_per_share': [],
            'revenue_growth': [],
            'profit_margins': [],
            'debt_ratios': [],
        }
        
        # EPS pattern
        eps_pattern = r'(?:Earnings|EPS)[\s:]*\$?([\d.]+)'
        for match in re.finditer(eps_pattern, text, re.IGNORECASE):
            metrics['earnings_per_share'].append({'value': match.group(1)})
        
        # Growth rates
        growth_pattern = r'([\d.]+)%\s+(?:increase|growth|decline)'
        for match in re.finditer(growth_pattern, text, re.IGNORECASE):
            metrics['revenue_growth'].append({'value': match.group(1) + '%'})
        
        # Profit margins
        margin_pattern = r'(?:margin|profit)\s+of\s+([\d.]+)%'
        for match in re.finditer(margin_pattern, text, re.IGNORECASE):
            metrics['profit_margins'].append({'value': match.group(1) + '%'})
        
        return metrics

    def _extract_tables(self, text: str) -> List[Dict]:
        """Extract table structures from document.
        
        Args:
            text: Document text
            
        Returns:
            List of table information
        """
        tables = []
        
        # Simple table detection - look for consistent column alignment
        lines = text.split('\n')
        potential_tables = []
        
        for i, line in enumerate(lines):
            # Tables often have multiple spaces or tabs
            if re.search(r'\s{2,}', line) or '\t' in line:
                potential_tables.append(i)
        
        # Group consecutive lines as single table
        if potential_tables:
            table_info = {
                'detected': len(potential_tables),
                'lines': potential_tables[:10]  # Show first 10 table lines
            }
            tables.append(table_info)
        
        return tables

    def extract_mda_section(self, text: str) -> Optional[str]:
        """Extract Management Discussion & Analysis section.
        
        Args:
            text: Document text
            
        Returns:
            MDA section text or None
        """
        pattern = r'(?:Item\s+7|MD&A|MANAGEMENT.*DISCUSSION)([\s\S]*?)(?=Item\s+8|FINANCIAL STATEMENTS|$)'
        
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return None

    def extract_financial_statements(self, text: str) -> Dict[str, str]:
        """Extract financial statement sections.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of financial statements
        """
        statements = {}
        
        # Balance Sheet
        balance_sheet_pattern = r'(?:CONSOLIDATED\s+)?BALANCE SHEET([\s\S]*?)(?=INCOME STATEMENT|STATEMENT OF|$)'
        match = re.search(balance_sheet_pattern, text, re.IGNORECASE)
        if match:
            statements['balance_sheet'] = match.group(1).strip()[:500]
        
        # Income Statement
        income_pattern = r'(?:CONSOLIDATED\s+)?(?:INCOME STATEMENT|STATEMENT OF EARNINGS)([\s\S]*?)(?=CASH FLOW|STATEMENT OF|$)'
        match = re.search(income_pattern, text, re.IGNORECASE)
        if match:
            statements['income_statement'] = match.group(1).strip()[:500]
        
        # Cash Flow
        cashflow_pattern = r'(?:CONSOLIDATED\s+)?(?:CASH FLOW|STATEMENT OF CASH FLOWS)([\s\S]*?)(?=NOTES|$)'
        match = re.search(cashflow_pattern, text, re.IGNORECASE)
        if match:
            statements['cash_flow'] = match.group(1).strip()[:500]
        
        return statements


if __name__ == "__main__":
    print("Document Parser module initialized")
