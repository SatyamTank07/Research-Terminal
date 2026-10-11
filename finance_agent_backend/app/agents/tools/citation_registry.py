"""Numbered filing citations for one assistant message.

Producers register a `SourceSpec` and get back an inline markdown marker `[[n]](#cite-n)` (never with
surrounding whitespace). `finalize` resolves chunk metadata in one query, drops sources whose chunks no longer
exist (removing their markers byte-exactly), renumbers the survivors 1..N in order of first appearance and
returns the markdown together with the `SourceRef` list persisted as `ChatMessage.sources`.
"""

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from app.database import SessionLocal
from app.models import Document, DocumentChunk

logger = logging.getLogger("finance_agent.agents.citations")

SourceKind = Literal["verbatim", "derived", "market"]
MAX_MARKERS = 3
_MARKER = re.compile(r"\[\[(\d+)\]\]\(#cite-\1\)")


def marker(n: int) -> str:
    """Inline markdown marker for source `n`."""
    return f"[[{n}]](#cite-{n})"


def strip_markers(text: str) -> str:
    """Removes every citation marker."""
    return _MARKER.sub("", text)


@dataclass(frozen=True)
class SourceSpec:
    """What a producer knows about a source; chunk metadata is resolved later in one query."""

    kind: SourceKind
    chunk_ids: Tuple[str, ...] = ()
    quote: Optional[str] = None
    derived_from: Optional[str] = None
    provider: Optional[str] = None
    label: Optional[str] = None

    @property
    def key(self) -> Tuple[str, Tuple[str, ...], str]:
        """Dedupe key: quotes by text, derived sources by chunk set, market inputs by label and provider."""
        if self.kind == "verbatim":
            detail = " ".join((self.quote or "").split()).lower()
        elif self.kind == "market":
            detail = f"{self.label}|{self.provider}"
        else:
            detail = ""
        return self.kind, tuple(sorted(set(self.chunk_ids))), detail


class SourceChunk(BaseModel):
    """Location of one cited chunk inside an ingested filing."""

    chunk_id: str
    document_id: str
    chunk_index: int
    ticker: str
    fiscal_year: int
    item: Optional[str] = None
    breadcrumb: Optional[str] = None


class SourceRef(BaseModel):
    """One numbered source as persisted in ChatMessage.sources and rendered by the frontend."""

    n: int
    kind: SourceKind
    label: str
    document_id: Optional[str] = None
    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    chunks: List[SourceChunk] = Field(default_factory=list)
    quote: Optional[str] = None
    derived_from: Optional[str] = None
    provider: Optional[str] = None


ChunkResolver = Callable[[List[str]], Dict[str, SourceChunk]]


def resolve_chunk_metadata(chunk_ids: List[str]) -> Dict[str, SourceChunk]:
    """chunk id -> location metadata for the ids that still exist (one query).

    A database failure raises: returning {} would silently drop every filing source, while the caller's
    failure path publishes the report without markers and says so.
    """
    if not chunk_ids:
        return {}
    db = SessionLocal()
    try:
        rows = (
            db.query(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
                DocumentChunk.item,
                DocumentChunk.breadcrumb,
                Document.ticker,
                Document.fiscal_year,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .filter(DocumentChunk.id.in_(chunk_ids))
            .all()
        )
    finally:
        db.close()
    return {
        r.id: SourceChunk(
            chunk_id=r.id,
            document_id=r.document_id,
            chunk_index=r.chunk_index,
            ticker=r.ticker,
            fiscal_year=r.fiscal_year,
            item=r.item,
            breadcrumb=r.breadcrumb,
        )
        for r in rows
    }


class CitationRegistry:
    """Numbers the sources cited by one message."""

    def __init__(self, resolver: Optional[ChunkResolver] = None):
        self._resolver = resolver
        self._specs: List[SourceSpec] = []
        self._numbers: Dict[Tuple[str, Tuple[str, ...], str], int] = {}

    def cite(self, spec: Optional[SourceSpec]) -> str:
        """Marker for `spec`; empty when there is nothing to point at."""
        if spec is None or (spec.kind != "market" and not spec.chunk_ids):
            return ""
        n = self._numbers.get(spec.key)
        if n is None:
            self._specs.append(spec)
            n = self._numbers[spec.key] = len(self._specs)
        return marker(n)

    def cite_all(self, specs: Iterable[Optional[SourceSpec]], limit: int = MAX_MARKERS) -> str:
        """Concatenated markers for distinct specs, at most `limit`."""
        out: List[str] = []
        for spec in specs:
            m = self.cite(spec)
            if m and m not in out:
                out.append(m)
            if len(out) >= limit:
                break
        return "".join(out)

    def finalize(self, markdown: str) -> Tuple[str, List[SourceRef]]:
        """Resolves chunks, drops dead sources with their markers and renumbers survivors by first appearance."""
        resolver = self._resolver or resolve_chunk_metadata
        ids = sorted({cid for spec in self._specs for cid in spec.chunk_ids})
        meta = resolver(ids) if ids else {}
        renumbered: Dict[int, int] = {}
        refs: List[SourceRef] = []

        def replace(match: re.Match) -> str:
            old = int(match.group(1))
            if old not in renumbered:
                spec = self._specs[old - 1] if 0 < old <= len(self._specs) else None
                ref = self._build_ref(spec, meta, len(refs) + 1) if spec else None
                renumbered[old] = ref.n if ref else 0
                if ref:
                    refs.append(ref)
            new = renumbered[old]
            return marker(new) if new else ""

        text = _MARKER.sub(replace, markdown)
        dropped = sum(1 for n in renumbered.values() if n == 0)
        if dropped:
            logger.warning(f"Citations: dropped {dropped} source(s) whose filing chunks no longer exist.")
        return text, refs

    @staticmethod
    def _build_ref(spec: SourceSpec, meta: Dict[str, SourceChunk], n: int) -> Optional[SourceRef]:
        chunks = [meta[cid] for cid in dict.fromkeys(spec.chunk_ids) if cid in meta]
        if spec.kind != "market" and not chunks:
            return None
        first = chunks[0] if chunks else None
        if first is not None:
            default_label = first.breadcrumb or first.item or "SEC filing"
        else:
            default_label = spec.label or spec.provider or "Market data"
        return SourceRef(
            n=n,
            kind=spec.kind,
            label=spec.label or default_label,
            document_id=first.document_id if first else None,
            ticker=first.ticker if first else None,
            fiscal_year=first.fiscal_year if first else None,
            chunks=chunks,
            quote=spec.quote if spec.kind == "verbatim" else None,
            derived_from=spec.derived_from if spec.kind == "derived" else None,
            provider=spec.provider if spec.kind == "market" else None,
        )
