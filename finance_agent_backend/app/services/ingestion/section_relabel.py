"""Re-derives section labels (part, item, title, breadcrumb) for chunks that are already stored.

Filings ingested before the parser recognised table-rendered headings ("| Item 1A. | Risk Factors |")
or rejected cross-references ("... Item 8 of Part II ...") carry wrong `item` labels. Their chunk
text is still right: every heading table forced a chunk boundary, so the stored chunks only need new
labels, never new content. This module replays the stored chunks in order:

- a stored heading table opens its item (the same rule the parser now applies);
- a stored PART boundary keeps the running item and switches the part, as the parser does;
- a stored section whose title is a cross-reference fragment ("of this report)") is an artifact:
  it continues the running section, or, when it opened a new part, takes the only standard Form
  10-K item that fits between its neighbours;
- every other stored section boundary was a real text heading and is kept.

Only labels and the label-dependent search text (`content_tsv`) change; embeddings are untouched.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from app.services.financial_tables import markdown_to_rows
from app.services.ingestion.chunker import SECChunker
from app.services.ingestion.sec_parser import (
    SECTION_ITEM_ORDER,
    TEN_K_PART_ITEMS,
    ParsedSection,
    SECFilingMetadata,
    classify_heading,
    is_heading_title,
)

logger = logging.getLogger("finance_agent.ingestion.section_relabel")

COVER_ITEM = "Cover Page"

# Standard Form 10-K item titles; used only to name a section whose own heading was overwritten by
# a cross-reference at a part boundary.
TEN_K_ITEM_TITLES: Dict[str, str] = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "[Reserved]", "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk", "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}


@dataclass(frozen=True)
class StoredChunk:
    """The label-relevant view of a stored DocumentChunk."""
    chunk_index: int
    chunk_type: str
    content: str
    part: Optional[str]
    item: Optional[str]
    section_title: Optional[str]
    table_title: Optional[str] = None


@dataclass(frozen=True)
class SectionLabel:
    part: Optional[str]
    item: Optional[str]
    title: str


@dataclass
class ChunkLabels:
    part: Optional[str]
    item: Optional[str]
    sub_section: Optional[str]
    breadcrumb: str
    section_title: Optional[str]


@dataclass
class ChunkChange:
    chunk_id: str
    chunk_index: int
    before: ChunkLabels
    after: ChunkLabels


@dataclass
class RelabelPlan:
    """Planned label changes for one filing."""
    label: str
    total_chunks: int
    before_counts: Counter
    after_counts: Counter
    changes: List[ChunkChange] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def heading_from_table_chunk(content: str) -> Optional[Tuple[str, str, str]]:
    """Classifies a stored table chunk as a PART / Item heading (see `classify_heading`)."""
    _, rows = markdown_to_rows(content)
    return classify_heading(" ".join(cell for row in rows for cell in row if cell))


def _item_rank(item: Optional[str]) -> int:
    number = (item or "").replace("Item", "").strip()
    return SECTION_ITEM_ORDER.index(number) if number in SECTION_ITEM_ORDER else -1


def _stored_label(chunk: StoredChunk) -> SectionLabel:
    return SectionLabel(chunk.part, chunk.item, chunk.section_title or "")


def _is_artifact(label: SectionLabel) -> bool:
    """A stored section opened by a cross-reference ("Item 8 of this report") instead of a heading."""
    return bool(label.item) and label.item != COVER_ITEM and not is_heading_title(label.title)


def _is_part_boundary(label: SectionLabel, previous: SectionLabel) -> bool:
    return bool(label.part) and label.part != previous.part and label.title == f"{label.part} Overview"


def _next_known_item(chunks: Sequence[StoredChunk], start: int) -> Optional[str]:
    """The first item after `start` that is opened by a heading table or a genuine stored heading."""
    previous = _stored_label(chunks[start])
    for chunk in chunks[start + 1:]:
        if chunk.chunk_type == "table":
            heading = heading_from_table_chunk(chunk.content)
            if heading and heading[0] == "item":
                return heading[1]
        stored = _stored_label(chunk)
        if stored != previous and not _is_artifact(stored) and not _is_part_boundary(stored, previous):
            return stored.item
        previous = stored
    return None


def _infer_ten_k_item(part: Optional[str], after_item: Optional[str], before_item: Optional[str]) -> Optional[str]:
    """The single standard item of `part` ranked strictly between the two neighbouring items."""
    low = _item_rank(after_item)
    high = _item_rank(before_item) if before_item else len(SECTION_ITEM_ORDER)
    candidates = [n for n in TEN_K_PART_ITEMS.get(part or "", ()) if low < SECTION_ITEM_ORDER.index(n) < high]
    return candidates[0] if len(candidates) == 1 else None


def derive_section_labels(chunks: Sequence[StoredChunk], filing_type: str = "10-K") -> Tuple[List[SectionLabel], List[str]]:
    """Section label for every chunk (same order as `chunks`, sorted by chunk_index) plus warnings
    for artifact sections whose true item could not be determined (those keep their stored label)."""
    labels: List[SectionLabel] = []
    warnings: List[str] = []
    current: Optional[SectionLabel] = None
    previous_stored: Optional[SectionLabel] = None
    # Item a cross-reference artifact wrongly opened; the parser carried it across PART boundaries.
    artifact_item: Optional[str] = None
    ten_k = (filing_type or "").upper().startswith("10-K")

    for idx, chunk in enumerate(chunks):
        stored = _stored_label(chunk)
        if current is None:
            current = stored
        elif stored != previous_stored:
            if _is_part_boundary(stored, previous_stored):
                # The stored item is the last heading before the part (whose own chunks may have been
                # too short to keep), unless it was carried over from an artifact.
                item = current.item if stored.item == artifact_item else stored.item
                current = SectionLabel(stored.part, item, stored.title)
            elif not _is_artifact(stored):
                current = stored
                artifact_item = None
            elif stored.part == current.part:
                artifact_item = stored.item  # cross-reference inside the running section
            else:
                artifact_item = stored.item
                inferred = _infer_ten_k_item(stored.part, current.item, _next_known_item(chunks, idx)) if ten_k else None
                if inferred:
                    current = SectionLabel(stored.part, f"Item {inferred}", TEN_K_ITEM_TITLES[inferred])
                else:
                    current = stored
                    warnings.append(
                        f"chunk {chunk.chunk_index}: section '{stored.item}: {stored.title}' looks like a "
                        f"cross-reference but its item cannot be inferred; label left unchanged."
                    )
                    logger.warning(warnings[-1])
        previous_stored = stored

        if chunk.chunk_type == "table":
            heading = heading_from_table_chunk(chunk.content)
            if heading and heading[0] == "item":
                current = SectionLabel(current.part, heading[1], heading[2] or heading[1])
            elif heading and heading[0] == "part":
                current = SectionLabel(heading[1], current.item, f"{heading[1]} Overview")
        labels.append(current)

    return labels, warnings


def plan_relabel(
    metadata: SECFilingMetadata,
    rows: Sequence[Tuple[str, StoredChunk, str]],
) -> RelabelPlan:
    """Builds the relabel plan for one filing. `rows` are (chunk_id, stored chunk, stored breadcrumb)."""
    ordered = sorted(rows, key=lambda row: row[1].chunk_index)
    chunks = [chunk for _, chunk, _ in ordered]
    labels, warnings = derive_section_labels(chunks, metadata.filing_type)
    chunker = SECChunker()

    plan = RelabelPlan(
        label=f"{metadata.ticker} {metadata.filing_type} FY{metadata.fiscal_year} ({metadata.source_filename})",
        total_chunks=len(chunks),
        before_counts=Counter(chunk.item for chunk in chunks),
        after_counts=Counter(label.item for label in labels),
        warnings=warnings,
    )
    for (chunk_id, chunk, breadcrumb), label in zip(ordered, labels):
        section = ParsedSection(part=label.part, item=label.item, title=label.title)
        sub_section = (chunk.table_title or label.title) if chunk.chunk_type == "table" else label.title
        before = ChunkLabels(chunk.part, chunk.item, None, breadcrumb, chunk.section_title)
        after = ChunkLabels(label.part, label.item, sub_section, chunker.build_breadcrumb(metadata, section), label.title)
        if (before.part, before.item, before.breadcrumb, before.section_title) != (after.part, after.item, after.breadcrumb, after.section_title):
            plan.changes.append(ChunkChange(chunk_id=chunk_id, chunk_index=chunk.chunk_index, before=before, after=after))
    return plan


def format_plan(plan: RelabelPlan) -> str:
    """Human-readable before/after item counts for one filing."""
    lines = [f"{plan.label}: {plan.total_chunks} chunks, {len(plan.changes)} to relabel"]
    items = sorted(set(plan.before_counts) | set(plan.after_counts), key=lambda item: (item != COVER_ITEM, _item_rank(item), str(item)))
    for item in items:
        before, after = plan.before_counts.get(item, 0), plan.after_counts.get(item, 0)
        marker = "" if before == after else "  *"
        lines.append(f"    {str(item):<12} {before:>4} -> {after:<4}{marker}")
    lines.extend(f"    WARNING {warning}" for warning in plan.warnings)
    return "\n".join(lines)
