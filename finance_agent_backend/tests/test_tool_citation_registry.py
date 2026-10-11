"""Numbered filing citations (app/agents/tools/citation_registry.py)."""

import unittest
from unittest.mock import MagicMock, patch

from app.agents.tools.citation_registry import (
    CitationRegistry,
    resolve_chunk_metadata,
    SourceChunk,
    SourceSpec,
    marker,
    strip_markers,
)


def _chunk(cid: str, idx: int = 0) -> SourceChunk:
    return SourceChunk(
        chunk_id=cid, document_id="doc-1", chunk_index=idx, ticker="ACME", fiscal_year=2025,
        item="Item 7", breadcrumb=f"ACME > Item 7 > {cid}",
    )


def _resolver(known):
    return MagicMock(side_effect=lambda ids: {cid: _chunk(cid, i) for i, cid in enumerate(ids) if cid in known})


QUOTE = SourceSpec(kind="verbatim", chunk_ids=("c1",), quote="Net sales  increased 6%")
DERIVED = SourceSpec(kind="derived", chunk_ids=("c2", "c3"), derived_from="Computed from the audited income statement (FY2025)")
MARKET = SourceSpec(kind="market", label="Share price", provider="yfinance_live")


class TestCitationRegistry(unittest.TestCase):
    def test_marker_format(self):
        self.assertEqual(marker(3), "[[3]](#cite-3)")

    def test_numbering_and_dedupe(self):
        reg = CitationRegistry(resolver=_resolver(set()))
        self.assertEqual(reg.cite(QUOTE), marker(1))
        self.assertEqual(reg.cite(DERIVED), marker(2))
        # Same quote modulo whitespace/case -> same number.
        self.assertEqual(reg.cite(SourceSpec(kind="verbatim", chunk_ids=("c1",), quote="net sales increased 6%")), marker(1))
        # Derived sources dedupe on the chunk set only; the first description wins.
        self.assertEqual(reg.cite(SourceSpec(kind="derived", chunk_ids=("c3", "c2"), derived_from="other")), marker(2))
        self.assertEqual(reg.cite(MARKET), marker(3))
        self.assertEqual(reg.cite(SourceSpec(kind="market", label="Beta", provider="yfinance_live")), marker(4))

    def test_nothing_to_point_at_returns_empty(self):
        reg = CitationRegistry(resolver=_resolver(set()))
        self.assertEqual(reg.cite(None), "")
        self.assertEqual(reg.cite(SourceSpec(kind="derived", chunk_ids=())), "")
        self.assertEqual(reg.cite(SourceSpec(kind="verbatim", chunk_ids=(), quote="x")), "")

    def test_cite_all_distinct_and_limited(self):
        reg = CitationRegistry(resolver=_resolver(set()))
        specs = [QUOTE, None, QUOTE, DERIVED, MARKET, SourceSpec(kind="verbatim", chunk_ids=("c9",), quote="x")]
        self.assertEqual(reg.cite_all(specs), marker(1) + marker(2) + marker(3))

    def test_finalize_renumbers_by_first_appearance_and_resolves_once(self):
        resolver = _resolver({"c1", "c2", "c3"})
        reg = CitationRegistry(resolver=resolver)
        q, d, m = reg.cite(QUOTE), reg.cite(DERIVED), reg.cite(MARKET)
        text, refs = reg.finalize(f"Price{m}. Sales{q}. Table{d}. Again{q}.")
        self.assertEqual(text, f"Price{marker(1)}. Sales{marker(2)}. Table{marker(3)}. Again{marker(2)}.")
        self.assertEqual([r.n for r in refs], [1, 2, 3])
        self.assertEqual([r.kind for r in refs], ["market", "verbatim", "derived"])
        resolver.assert_called_once_with(["c1", "c2", "c3"])
        self.assertEqual(refs[0].provider, "yfinance_live")
        self.assertEqual(refs[0].chunks, [])
        self.assertEqual(refs[1].quote, "Net sales  increased 6%")
        self.assertEqual(refs[1].document_id, "doc-1")
        self.assertEqual(refs[1].ticker, "ACME")
        self.assertEqual([c.chunk_id for c in refs[2].chunks], ["c2", "c3"])
        self.assertEqual(refs[2].derived_from, "Computed from the audited income statement (FY2025)")

    def test_dead_source_marker_removed_exactly(self):
        reg = CitationRegistry(resolver=_resolver({"c2", "c3"}))
        q, d = reg.cite(QUOTE), reg.cite(DERIVED)
        text, refs = reg.finalize(f"Sales grew{q}. Margins held{d}.")
        self.assertEqual(text, f"Sales grew. Margins held{marker(1)}.")
        self.assertEqual([r.n for r in refs], [1])

    def test_partial_chunk_loss_keeps_source(self):
        reg = CitationRegistry(resolver=_resolver({"c3"}))
        d = reg.cite(DERIVED)
        _, refs = reg.finalize(f"x{d}")
        self.assertEqual([c.chunk_id for c in refs[0].chunks], ["c3"])

    def test_unused_specs_are_not_published(self):
        reg = CitationRegistry(resolver=_resolver({"c1", "c2", "c3"}))
        reg.cite(QUOTE)
        d = reg.cite(DERIVED)
        _, refs = reg.finalize(f"only{d}")
        self.assertEqual([r.kind for r in refs], ["derived"])

    def test_labels(self):
        reg = CitationRegistry(resolver=_resolver({"c1"}))
        q = reg.cite(QUOTE)
        m = reg.cite(MARKET)
        labelled = reg.cite(SourceSpec(kind="verbatim", chunk_ids=("c1",), quote="other", label="Custom"))
        _, refs = reg.finalize(q + m + labelled)
        self.assertEqual([r.label for r in refs], ["ACME > Item 7 > c1", "Share price", "Custom"])

    def test_no_chunks_skips_resolver(self):
        resolver = _resolver(set())
        reg = CitationRegistry(resolver=resolver)
        text, refs = reg.finalize(f"p{reg.cite(MARKET)}")
        resolver.assert_not_called()
        self.assertEqual(text, f"p{marker(1)}")

    def test_metadata_lookup_failure_raises_instead_of_dropping_every_source(self):
        session = MagicMock()
        session.query.side_effect = RuntimeError("database unavailable")
        with patch("app.agents.tools.citation_registry.SessionLocal", return_value=session):
            with self.assertRaises(RuntimeError):
                resolve_chunk_metadata(["c1"])
        session.close.assert_called_once()

    def test_strip_markers(self):
        self.assertEqual(strip_markers(f"a{marker(1)}{marker(12)}."), "a.")
