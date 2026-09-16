import re
import os
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger("finance_agent.ingestion.sec_parser")

# Standard SEC 10-K Item patterns
ITEM_PATTERN = re.compile(
    r"\b(?:ITEM|Item)\s+([0-9]{1,2}[A-Z]?)(?:[\.\:\s\—\-]|\s*&#160;\s*)([^\n\r<]{0,120})",
    re.IGNORECASE,
)

PART_PATTERN = re.compile(
    r"\b(?:PART|Part)\s+([IVXLCDM]+)\b",
    re.IGNORECASE,
)


class SECFilingMetadata:
    def __init__(
        self,
        ticker: str,
        company_name: str,
        filing_type: str,
        fiscal_year: int,
        period_end_date: Optional[str] = None,
        cik: Optional[str] = None,
        source_filename: str = "",
    ):
        self.ticker = ticker
        self.company_name = company_name
        self.filing_type = filing_type
        self.fiscal_year = fiscal_year
        self.period_end_date = period_end_date
        self.cik = cik
        self.source_filename = source_filename

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "filing_type": self.filing_type,
            "fiscal_year": self.fiscal_year,
            "period_end_date": self.period_end_date,
            "cik": self.cik,
            "source_filename": self.source_filename,
        }


class ParsedBlock:
    """Represents a discrete semantic block: either narrative text or a financial table."""
    def __init__(
        self,
        block_type: str,  # 'text' or 'table'
        content: str,     # Markdown string or cleaned text
        raw_text: str = "",
        table_title: Optional[str] = None,
        table_data: Optional[List[List[str]]] = None,
        summary: Optional[str] = None,
    ):
        self.block_type = block_type
        self.content = content
        self.raw_text = raw_text or content
        self.table_title = table_title
        self.table_data = table_data or []
        self.summary = summary


class ParsedSection:
    def __init__(
        self,
        part: Optional[str],
        item: Optional[str],
        title: str,
        blocks: Optional[List[ParsedBlock]] = None,
    ):
        self.part = part
        self.item = item
        self.title = title
        self.blocks = blocks or []


class SECParser:
    """Parser for SEC iXBRL/HTML 10-K & 10-Q filings."""

    def __init__(self, parser_engine: str = "html.parser"):
        # Prefer lxml if installed, else fallback to html.parser
        try:
            import lxml  # noqa: F401
            self.parser_engine = "lxml"
        except ImportError:
            self.parser_engine = "html.parser"

    def extract_metadata(self, html_content: str, filename: str) -> SECFilingMetadata:
        """Dynamically extracts DEI metadata from the filing, with filename fallback."""
        # Search DEI tags in first 500k characters
        head_sample = html_content[:500000]

        # CIK
        cik_match = re.search(r"EntityCentralIndexKey[^>]*>([0-9]{10})<", head_sample)
        cik = cik_match.group(1) if cik_match else None

        # Fiscal Year
        year_match = re.search(r"DocumentFiscalYearFocus[^>]*>([0-9]{4})<", head_sample)
        fiscal_year = int(year_match.group(1)) if year_match else None

        # Document Type (10-K, 10-Q)
        type_match = re.search(r"DocumentType[^>]*>([A-Za-z0-9\-]+)<", head_sample)
        filing_type = type_match.group(1).strip() if type_match else "10-K"

        # Trading Symbol / Ticker
        ticker_match = re.search(r"TradingSymbol[^>]*>([A-Za-z0-9]+)<", head_sample)
        ticker = ticker_match.group(1).upper() if ticker_match else None

        # Company Name
        name_match = re.search(r"EntityRegistrantName[^>]*>([^<]+)<", head_sample)
        company_name = name_match.group(1).strip() if name_match else None

        # Period End Date
        period_match = re.search(r"DocumentPeriodEndDate[^>]*>([0-9]{4}-[0-9]{2}-[0-9]{2})<", head_sample)
        period_end_date = period_match.group(1) if period_match else None

        # Fallback from filename (e.g., 'aapl-20250927.htm' or 'tsla-20251231.htm')
        basename = os.path.basename(filename).lower()
        parts = basename.split("-")
        fallback_ticker = parts[0].upper() if len(parts) > 0 and len(parts[0]) <= 5 else "UNKNOWN"

        if not ticker:
            ticker = fallback_ticker

        if not company_name:
            if ticker == "AAPL":
                company_name = "Apple Inc."
            elif ticker == "TSLA":
                company_name = "Tesla, Inc."
            else:
                company_name = f"{ticker} Corp."

        if not fiscal_year and len(parts) > 1 and len(parts[1]) >= 4 and parts[1][:4].isdigit():
            fiscal_year = int(parts[1][:4])
        elif not fiscal_year:
            fiscal_year = 2025

        if not period_end_date and len(parts) > 1 and len(parts[1]) >= 8:
            date_str = parts[1][:8]
            if date_str.isdigit():
                period_end_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        return SECFilingMetadata(
            ticker=ticker,
            company_name=company_name,
            filing_type=filing_type,
            fiscal_year=fiscal_year,
            period_end_date=period_end_date,
            cik=cik,
            source_filename=os.path.basename(filename),
        )

    def is_spacer_table(self, table: Tag) -> bool:
        """Determines if a table is a purely decorative spacer (1-row line divider)."""
        rows = table.find_all("tr")
        if not rows:
            return True
        if len(rows) == 1:
            text = table.get_text(strip=True)
            if not text or len(text) < 3:
                return True
        # Check style for min-height or tiny dividers
        style = table.get("style", "").lower()
        if "height:1pt" in style or "height:2pt" in style or "height:3pt" in style:
            text = table.get_text(strip=True)
            if not text:
                return True
        return False

    def table_to_markdown(self, table: Tag) -> Tuple[str, List[List[str]], Optional[str]]:
        """Converts an HTML table into a clean Markdown table with row data extracted."""
        raw_rows = table.find_all("tr")
        grid: List[List[str]] = []

        for row in raw_rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            row_data = []
            for cell in cells:
                # Remove nested junk while keeping text
                text = " ".join(cell.get_text(" ", strip=True).split())
                # Normalize currency and percentages
                text = text.replace("$ ", "$").replace(" %", "%")
                row_data.append(text)
            # Filter out rows that are entirely empty
            if any(cell.strip() for cell in row_data):
                grid.append(row_data)

        if not grid:
            return "", [], None

        # Normalize column counts across rows
        max_cols = max(len(r) for r in grid)
        # If the table has only 1 column, it's likely a formatted paragraph
        if max_cols <= 1:
            full_text = "\n".join(" ".join(r) for r in grid if r)
            return full_text, grid, None

        padded_grid = [r + [""] * (max_cols - len(r)) for r in grid]

        # Extract title/caption if first row is a header title spanning all columns
        table_title = None
        start_row = 0
        if len(padded_grid) > 1 and sum(1 for c in padded_grid[0] if c.strip()) == 1:
            # First row has single merged title
            table_title = next(c for c in padded_grid[0] if c.strip())
            start_row = 1

        # Use the next row as column headers
        headers = padded_grid[start_row]
        # Clean empty header labels
        headers = [h if h.strip() else f"Col_{idx+1}" for idx, h in enumerate(headers)]

        md_lines = []
        if table_title:
            md_lines.append(f"### {table_title}\n")

        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join([":---"] * max_cols) + " |")

        for r in padded_grid[start_row + 1:]:
            md_lines.append("| " + " | ".join(r) + " |")

        markdown_text = "\n".join(md_lines)
        return markdown_text, padded_grid, table_title

    def generate_table_summary(
        self,
        table_title: Optional[str],
        grid: List[List[str]],
        metadata: SECFilingMetadata,
        item_context: str = "",
    ) -> str:
        """Generates a concise linearized description ('The Search Face') for vector embedding."""
        company = metadata.company_name
        year = metadata.fiscal_year
        title = table_title or "Financial Data Table"

        summary_parts = [
            f"{company} Form 10-K ({year}) - {item_context} - {title}:",
        ]

        # Extract up to 10 prominent metrics from rows
        for row in grid[1:12]:
            non_empty = [c for c in row if c.strip()]
            if len(non_empty) >= 2:
                metric_name = non_empty[0]
                values = non_empty[1:]
                # Format: "iPhone: $209,586 (2025), $201,183 (2024)"
                summary_parts.append(f"- {metric_name}: {', '.join(values[:4])}")

        return "\n".join(summary_parts)

    def parse(self, file_path: str) -> Tuple[SECFilingMetadata, List[ParsedSection]]:
        """Parses the entire SEC HTML file into structured sections and typed blocks."""
        logger.info(f"Reading file: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        metadata = self.extract_metadata(html_content, file_path)
        logger.info(f"Extracted metadata: {metadata.to_dict()}")

        # Remove <ix:header> section to eliminate 2500+ lines of XBRL taxonomy
        cleaned_html = re.sub(
            r"<ix:header>.*?</ix:header>",
            "",
            html_content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        soup = BeautifulSoup(cleaned_html, self.parser_engine)

        # Drop script and style elements
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()

        # Find all top-level content elements (divs, tables, p, h1-h6)
        body = soup.find("body") or soup

        sections: List[ParsedSection] = []
        current_part: Optional[str] = None
        current_item: Optional[str] = "Cover Page"
        current_title: str = "Document Overview"
        current_blocks: List[ParsedBlock] = []

        # Helper to flush current section
        def flush_section():
            nonlocal current_blocks, current_part, current_item, current_title
            if current_blocks:
                sections.append(
                    ParsedSection(
                        part=current_part,
                        item=current_item,
                        title=current_title,
                        blocks=current_blocks,
                    )
                )
                current_blocks = []

        # Traverse direct children of body in reading order
        for element in body.children:
            if not getattr(element, "name", None):
                continue

            # Check if this element is or contains one or more tables
            nested_tables = [element] if element.name == "table" else element.find_all("table")
            if nested_tables:
                for tbl in nested_tables:
                    if self.is_spacer_table(tbl):
                        continue

                    md_table, grid, title = self.table_to_markdown(tbl)
                    if not md_table or not grid:
                        continue

                    item_label = f"{current_item} ({current_title})" if current_item else ""
                    summary = self.generate_table_summary(title, grid, metadata, item_label)

                    current_blocks.append(
                        ParsedBlock(
                            block_type="table",
                            content=md_table,
                            raw_text=tbl.get_text(" ", strip=True),
                            table_title=title,
                            table_data=grid,
                            summary=summary,
                        )
                    )
                continue
            else:
                # Text container (div / p)
                text = " ".join(element.get_text(" ", strip=True).split())
                if not text:
                    continue

                # Check if this text marks a new PART
                part_match = PART_PATTERN.match(text)
                if part_match and len(text) < 40:
                    flush_section()
                    current_part = f"PART {part_match.group(1).upper()}"
                    current_title = f"{current_part} Overview"
                    continue

                # Check if this text marks a new ITEM
                item_match = ITEM_PATTERN.search(text)
                if item_match and len(text) < 150:
                    flush_section()
                    item_num = item_match.group(1).upper()
                    item_desc = item_match.group(2).strip(" .:-")
                    current_item = f"Item {item_num}"
                    current_title = item_desc if item_desc else f"Item {item_num}"
                    continue

                # Regular narrative block
                current_blocks.append(
                    ParsedBlock(
                        block_type="text",
                        content=text,
                    )
                )

        # Flush any remaining blocks
        flush_section()
        logger.info(f"Parsed {len(sections)} sections from {file_path}")
        return metadata, sections
