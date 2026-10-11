import re
import os
import html
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Iterator, List, Optional, Sequence, Tuple, Union
from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

from app.services.financial_tables import is_well_formed, parse_statement_rows, render_clean_statement

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger("finance_agent.ingestion.sec_parser")

# Item numbers defined by Form 10-K / 10-Q (Regulation S-K order). Anything else after "Item"
# ("Item 5.02", "Item 99") is a cross-reference or an exhibit entry, never a section heading.
SECTION_ITEM_ORDER = [
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
    "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16",
]

# Form 10-K Part each item belongs to.
TEN_K_PART_ITEMS: Dict[str, Tuple[str, ...]] = {
    "PART I": ("1", "1A", "1B", "1C", "2", "3", "4"),
    "PART II": ("5", "6", "7", "7A", "8", "9", "9A", "9B", "9C"),
    "PART III": ("10", "11", "12", "13", "14"),
    "PART IV": ("15", "16"),
}
ITEM_PART: Dict[str, str] = {f"Item {number}": part for part, numbers in TEN_K_PART_ITEMS.items() for number in numbers}

# Sections every complete filing of the form must carry; a parse missing any is flagged.
CORE_SECTIONS = {
    "10-K": ("Item 1", "Item 1A", "Item 7", "Item 8"),
    "10-Q": ("Item 1", "Item 2"),
}

# Headings are short; a longer block opening with "Item 7" is prose that cites the item.
HEADING_MAX_CHARS = 200

# "Item 1A. Risk Factors", "ITEM 7 — MD&A", "Item 1A:" and combined "Items 1 and 2. Business and
# Properties" (labelled with the first item). The letter suffix may be split off by markup
# ("Item 1 A."), but never swallows the title's first letter ("Item 1 Business").
ITEM_HEADING_PATTERN = re.compile(
    r"^ITEMS?\s*(\d{1,2}(?:\s?[A-C](?![A-Z]))?)"
    r"(?:\s*(?:,|and|&|to|through|-|–|—)\s*\d{1,2}(?:[A-C](?![A-Z]))?)*"
    r"\s*[.:\-–—]?\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)

PART_HEADING_PATTERN = re.compile(
    r"^PART\s+(IV|I{1,3})\b\s*[.:\-–—]?\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)

# A heading title ending in a page number is a table-of-contents entry ("Risk Factors 6").
_TOC_PAGE_SUFFIX = re.compile(r"\s(?:F-)?\d{1,3}$")
# Text that may sit between table-of-contents entries without being section content.
_TOC_FILLER = re.compile(r"^(?:page|table\s+of\s+contents|index)?\s*(?:F-)?(?:\d{1,3}|[ivxl]{1,6})?$", re.IGNORECASE)

# Elements that start a new block of text; any other tag is inline formatting.
_BLOCK_TAGS = [
    "div", "p", "table", "section", "article", "ul", "ol", "li", "dl", "dt", "dd", "blockquote",
    "center", "pre", "header", "footer", "main", "aside", "h1", "h2", "h3", "h4", "h5", "h6", "hr",
]


def is_heading_title(title: str) -> bool:
    """True when the text after "Item N" reads as a section title rather than a cross-reference.

    Real titles start with a capital ("Business", "RISK FACTORS", "[Reserved]"); cross-references
    continue the sentence in lower case ("Item 8 of Part II", "Item 15 of this Annual Report").
    An empty title is a heading whose title sits in the next element.
    """
    title = (title or "").strip()
    if not title:
        return True
    lead = title.lstrip("\"'“‘([ ")[:1]
    return lead.isalpha() and lead.isupper()


def classify_heading(text: str) -> Optional[Tuple[str, str, str]]:
    """Classifies a short block as a section heading.

    Returns ("part", "PART II", title), ("item", "Item 1A", title) or None. Table-of-contents entries
    (title ending in a page number), cross-references and unknown item numbers return None.
    """
    text = " ".join((text or "").split())
    if not text or len(text) > HEADING_MAX_CHARS:
        return None

    part_match = PART_HEADING_PATTERN.match(text)
    if part_match:
        title = part_match.group(2).strip()
        if not is_heading_title(title) or _TOC_PAGE_SUFFIX.search(title):
            return None
        return "part", f"PART {part_match.group(1).upper()}", title

    item_match = ITEM_HEADING_PATTERN.match(text)
    if not item_match:
        return None
    number = re.sub(r"\s+", "", item_match.group(1)).upper()
    title = item_match.group(2).strip(" .:-–—")
    if number not in SECTION_ITEM_ORDER or not is_heading_title(title) or _TOC_PAGE_SUFFIX.search(title):
        return None
    return "item", f"Item {number}", title


def missing_core_sections(filing_type: str, sections: Sequence["ParsedSection"]) -> List[str]:
    """Core items of the form that the parse never located (empty for forms without a core list)."""
    expected = CORE_SECTIONS.get((filing_type or "").upper(), ())
    found = {section.item for section in sections}
    return [item for item in expected if item not in found]


class FilingMetadataError(ValueError):
    """The filing does not identify itself (ticker, registrant, form or fiscal year)."""


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


@dataclass
class _Event:
    """One unit of the filing in reading order: a text block, a table, or a PART / Item heading."""
    kind: str  # 'text', 'table', 'part' or 'item'
    text: str = ""
    label: Optional[str] = None
    title: str = ""
    table: Optional[Tuple[str, List[List[str]], Optional[str], str]] = field(default=None, repr=False)


_MONTH_ABBREVIATIONS = {"sept": "sep"}
_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y")


def _parse_filing_date(value: Optional[str]) -> Optional[str]:
    """Normalizes a cover-page date ("September 27, 2025", "Sept. 27, 2025", "2025-09-27") to ISO."""
    if not value:
        return None
    cleaned = re.sub(r"\s+,", ",", " ".join(value.replace(".", " ").split()))
    first, _, rest = cleaned.partition(" ")
    cleaned = f"{_MONTH_ABBREVIATIONS.get(first.lower(), first)} {rest}".strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


_IX_FACT_TAG = re.compile(r"<(/?)ix:non(?:numeric|fraction)\b[^>]*?(/?)>", re.IGNORECASE)


def _dei_fact(html_content: str, name: str) -> Optional[str]:
    """Text of the first iXBRL `dei:<name>` fact, including facts nested inside it."""
    start = re.search(rf'<ix:non(?:numeric|fraction)\b[^>]*\bname="dei:{name}"[^>]*>', html_content, re.IGNORECASE)
    if not start:
        return None
    depth = 1
    for tag in _IX_FACT_TAG.finditer(html_content, start.end()):
        if tag.group(2):  # self-closing
            continue
        depth += -1 if tag.group(1) else 1
        if depth == 0:
            inner = re.sub(r"<[^>]+>", " ", html_content[start.end():tag.start()])
            value = " ".join(html.unescape(inner).split())
            return value or None
    return None


def _cover_lines(html_content: str) -> List[str]:
    """Visible cover-page lines (markup and the hidden iXBRL header removed)."""
    visible = re.sub(r"<ix:header>.*?</ix:header>", " ", html_content, flags=re.DOTALL | re.IGNORECASE)
    visible = re.sub(r"<(?:/?(?:div|p|tr|br|table|li|h\d)\b)[^>]*>", "\n", visible, flags=re.IGNORECASE)
    visible = html.unescape(re.sub(r"<[^>]+>", " ", visible))
    return [line for line in (" ".join(raw.split()) for raw in visible.splitlines()) if line]


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
        """Extracts filing identity from the iXBRL dei facts, then the cover page, then the EDGAR
        file name (`<ticker>-<yyyymmdd>.htm`). Raises FilingMetadataError when the ticker, registrant,
        form type or fiscal year cannot be derived from the filing itself."""
        head_sample = html_content[:1_000_000]
        basename = os.path.basename(filename)
        name_match = re.match(r"^([a-z]{1,5})-(\d{8})\b", basename.lower())
        cover: Optional[List[str]] = None

        def cover_lines() -> List[str]:
            nonlocal cover
            if cover is None:
                cover = _cover_lines(head_sample)
            return cover

        cik = _dei_fact(head_sample, "EntityCentralIndexKey")

        ticker = (_dei_fact(head_sample, "TradingSymbol") or "").split(" ")[0].upper() or None
        if not ticker and name_match:
            ticker = name_match.group(1).upper()
            logger.warning(f"{basename}: no dei:TradingSymbol; ticker {ticker} taken from the EDGAR file name.")
        if not ticker:
            raise FilingMetadataError(f"{basename}: cannot determine the ticker (no dei:TradingSymbol, file name not <ticker>-<yyyymmdd>.htm).")

        company_name = _dei_fact(head_sample, "EntityRegistrantName")
        if not company_name:
            lines = cover_lines()
            anchor = next((i for i, line in enumerate(lines) if re.search(r"exact\s+name\s+of\s+registrant", line, re.IGNORECASE)), None)
            if anchor:
                company_name = lines[anchor - 1]
                logger.warning(f"{basename}: no dei:EntityRegistrantName; registrant '{company_name}' read from the cover page.")
        if not company_name:
            raise FilingMetadataError(f"{basename}: cannot determine the registrant name (no dei:EntityRegistrantName or cover-page name).")

        filing_type = _dei_fact(head_sample, "DocumentType")
        if not filing_type:
            form_match = next((m for m in (re.search(r"\bFORM\s+(10-K(?:/A)?|10-Q(?:/A)?|20-F|40-F)\b", line, re.IGNORECASE) for line in cover_lines()[:80]) if m), None)
            if form_match:
                filing_type = form_match.group(1).upper()
                logger.warning(f"{basename}: no dei:DocumentType; form {filing_type} read from the cover page.")
        if not filing_type:
            raise FilingMetadataError(f"{basename}: cannot determine the form type (no dei:DocumentType or cover-page FORM line).")

        period_end_date = _parse_filing_date(_dei_fact(head_sample, "DocumentPeriodEndDate"))
        if not period_end_date:
            ended = next((m for m in (re.search(r"(?:fiscal\s+year|quarterly\s+period|period)\s+ended:?\s+([A-Za-z]{3,9}\.?\s+\d{1,2}\s*,\s*\d{4})", line, re.IGNORECASE) for line in cover_lines()[:80]) if m), None)
            period_end_date = _parse_filing_date(ended.group(1)) if ended else None
        if not period_end_date and name_match:
            stamp = name_match.group(2)
            period_end_date = _parse_filing_date(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}")
        if not period_end_date:
            logger.warning(f"{basename}: period end date not found in dei facts, cover page or file name.")

        fiscal_year_text = _dei_fact(head_sample, "DocumentFiscalYearFocus") or ""
        fiscal_year = int(fiscal_year_text) if re.fullmatch(r"(?:19|20)\d{2}", fiscal_year_text) else None
        if fiscal_year is None and period_end_date:
            fiscal_year = int(period_end_date[:4])
            logger.warning(f"{basename}: no dei:DocumentFiscalYearFocus; fiscal year {fiscal_year} taken from the period end date {period_end_date}.")
        if fiscal_year is None:
            raise FilingMetadataError(f"{basename}: cannot determine the fiscal year (no dei:DocumentFiscalYearFocus or period end date).")

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

        # Financial statements: collapse split "$" / ")" cells and colspan misalignment into
        # a compact `| Line item | 2025 | 2024 |` table when every row maps cleanly onto the periods.
        statement = parse_statement_rows(padded_grid[start_row:], title=table_title)
        if is_well_formed(statement, min_ratio=0.9):
            return render_clean_statement(statement, period_label=""), padded_grid, table_title

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

        body = soup.find("body") or soup
        events = self._collect_events(body)
        self._attach_split_titles(events)
        self._demote_toc_headings(events)
        sections = self._assemble_sections(events, metadata)

        logger.info(f"Parsed {len(sections)} sections from {file_path}")
        missing = missing_core_sections(metadata.filing_type, sections)
        if missing:
            logger.warning(
                f"{metadata.ticker} {metadata.filing_type} FY{metadata.fiscal_year}: section headings not found for "
                f"{', '.join(missing)}; retrieval for these sections will fall back to whole-filing search."
            )
        return metadata, sections

    def _iter_blocks(self, container: Tag) -> Iterator[Union[Tag, str]]:
        """Yields the filing's leaf blocks in reading order: tables, block elements holding only inline
        content, and runs of loose inline text (as strings). Wrappers (a page div, an iXBRL text block)
        are descended into, so text sitting next to a table and headings nested inside wrappers are
        never lost."""
        inline_run: List[str] = []

        def flush_run() -> Iterator[str]:
            text = " ".join(" ".join(inline_run).split())
            inline_run.clear()
            if text:
                yield text

        for child in container.children:
            if isinstance(child, NavigableString):
                if child.__class__ is NavigableString:  # skip comments / CDATA / doctype
                    inline_run.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            if child.name == "table":
                yield from flush_run()
                yield child
            elif child.find(_BLOCK_TAGS) is not None:
                yield from flush_run()
                yield from self._iter_blocks(child)
            elif child.name in _BLOCK_TAGS:
                yield from flush_run()
                yield child
            else:
                inline_run.append(child.get_text(" ", strip=True))
        yield from flush_run()

    def _table_event(self, tbl: Tag) -> Optional[_Event]:
        """A heading laid out as a table ("| Item 1A. | Risk Factors |") becomes a heading event;
        any other non-spacer table becomes a table event."""
        if self.is_spacer_table(tbl):
            return None
        cell_text = " ".join(
            " ".join(cell.get_text(" ", strip=True).split()) for cell in tbl.find_all(["td", "th"])
        )
        heading = classify_heading(cell_text)
        if heading:
            kind, label, title = heading
            return _Event(kind=kind, text=cell_text, label=label, title=title)

        md_table, grid, title = self.table_to_markdown(tbl)
        if not md_table or not grid:
            return None
        return _Event(kind="table", table=(md_table, grid, title, tbl.get_text(" ", strip=True)))

    def _collect_events(self, body: Tag) -> List[_Event]:
        events: List[_Event] = []
        for block in self._iter_blocks(body):
            if isinstance(block, Tag) and block.name == "table":
                event = self._table_event(block)
                if event:
                    events.append(event)
                continue

            text = " ".join((block if isinstance(block, str) else block.get_text(" ", strip=True)).split())
            if not text:
                continue
            heading = classify_heading(text)
            if heading is None:
                events.append(_Event(kind="text", text=text))
                continue
            kind, label, title = heading
            nested = classify_heading(title) if kind == "part" else None
            if nested and nested[0] == "item":
                # "PART III ITEM 10. DIRECTORS ..." in one element opens both the part and the item.
                events.append(_Event(kind="part", text=text, label=label, title=""))
                events.append(_Event(kind="item", text=title, label=nested[1], title=nested[2]))
            else:
                events.append(_Event(kind=kind, text=text, label=label, title=title))
        return events

    @staticmethod
    def _attach_split_titles(events: List[_Event]) -> None:
        """Joins headings split across elements ("Item 1A." then "Risk Factors") into one heading."""
        i = 0
        while i < len(events) - 1:
            event, following = events[i], events[i + 1]
            if (
                event.kind == "item"
                and not event.title
                and following.kind == "text"
                and len(following.text) <= HEADING_MAX_CHARS // 2
                and is_heading_title(following.text)
                and not following.text.endswith((".", ";", ":"))
                and not _TOC_PAGE_SUFFIX.search(following.text)
            ):
                event.title = following.text
                event.text = f"{event.text} {following.text}"
                del events[i + 1]
            i += 1

    @staticmethod
    def _demote_toc_headings(events: List[_Event]) -> None:
        """Turns table-of-contents entries back into plain text.

        Headings separated only by page numbers or PART labels form a run. A heading in a run is a
        TOC entry when its item is headed again later in the filing (the real section follows), and
        a whole run is a TOC when it lists items from three or more Parts: real empty sections
        ("Item 6. [Reserved]", Part III items incorporated by reference) never span the document.
        """
        last_seen: Dict[str, int] = {}
        for idx, event in enumerate(events):
            if event.kind == "item":
                last_seen[event.label] = idx

        heading_idx = [idx for idx, event in enumerate(events) if event.kind == "item"]
        runs: List[List[int]] = []
        for pos, idx in enumerate(heading_idx):
            if pos and all(
                e.kind == "part" or (e.kind == "text" and _TOC_FILLER.match(e.text))
                for e in events[heading_idx[pos - 1] + 1:idx]
            ):
                runs[-1].append(idx)
            else:
                runs.append([idx])

        demote: List[int] = []
        for run in runs:
            parts = {ITEM_PART.get(events[idx].label) for idx in run}
            if len(parts - {None}) >= 3:
                demote.extend(run)
            else:
                demote.extend(idx for idx in run[:-1] if last_seen[events[idx].label] > idx)
        for idx in demote:
            events[idx].kind, events[idx].label, events[idx].title = "text", None, ""

    def _assemble_sections(self, events: Sequence[_Event], metadata: SECFilingMetadata) -> List[ParsedSection]:
        sections: List[ParsedSection] = []
        current_part: Optional[str] = None
        current_item: Optional[str] = "Cover Page"
        current_title: str = "Document Overview"
        current_blocks: List[ParsedBlock] = []

        def flush_section():
            nonlocal current_blocks
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

        for event in events:
            if event.kind == "part":
                flush_section()
                current_part = event.label
                current_title = f"{current_part} Overview"
            elif event.kind == "item":
                flush_section()
                current_item = event.label
                current_title = event.title or event.label
            elif event.kind == "table":
                md_table, grid, title, raw_text = event.table
                item_label = f"{current_item} ({current_title})" if current_item else ""
                current_blocks.append(
                    ParsedBlock(
                        block_type="table",
                        content=md_table,
                        raw_text=raw_text,
                        table_title=title,
                        table_data=grid,
                        summary=self.generate_table_summary(title, grid, metadata, item_label),
                    )
                )
            else:
                current_blocks.append(ParsedBlock(block_type="text", content=event.text))

        flush_section()
        return sections
