"""Assembles the report: source markers on the verified prose, the route layout, the numbered sources, the output."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.agents.shared.citations import CitationRegistry, SourceRef, strip_markers
from app.agents.shared.figures import annotate_sentences, build_figure_index
from app.agents.shared.prompts import render_prompt
from app.agents.specialized.lead_synthesizer.report_citations import (
    CITATIONS_FAILED_NOTE,
    EVIDENCE_NOTE,
    NO_CITATIONS_NOTE,
    NullCiter,
    ReportCiter,
)
from app.agents.specialized.lead_synthesizer.state_lead_synthesizer import Final10KResearchReport, ThreePillarThesis
from app.agents.specialized.lead_synthesizer.write import SynthesisProvenance
from app.agents.specialized.valuation_specialist.state_valuation_specialist import DCFValuationOutput

logger = logging.getLogger("finance_agent.agents.lead_synthesizer")


def cite_prose(
    prose: Dict[str, Optional[str]], fiscal_year: int, artifacts: Dict[str, Any], notes: List[str]
) -> Tuple[Any, Dict[str, Optional[str]]]:
    """Builds the report citer and appends source markers to verified prose; never raises."""
    try:
        citer = ReportCiter(CitationRegistry(), fiscal_year, **artifacts)
        index = build_figure_index(citer.figure_entries())
        cited = {
            key: annotate_sentences(text, index, citer.registry.cite_all) if text else text
            for key, text in prose.items()
        }
        return citer, cited
    except Exception as e:
        logger.exception(f"Citation build failed ({e!r}); publishing without filing markers.")
        notes.append("Filing citations could not be built for this report.")
        return NullCiter(), dict(prose)


def render_report(
    layout: str,
    ticker: str,
    company_name: str,
    fiscal_year: int,
    query_type: str,
    year_substituted: bool,
    prose: Dict[str, Optional[str]],
    notes: List[str],
    provenance: SynthesisProvenance,
    removed_sentences: int,
    citer: Any,
    artifacts: Dict[str, Any],
) -> str:
    """Renders report/<layout>.j2: every table from the artifacts, the cited prose between them."""
    markdown = render_prompt(
        "lead_synthesizer", f"report/{layout}.j2",
        ticker=ticker,
        company_name=company_name,
        fiscal_year=fiscal_year,
        query_type=query_type,
        year_substituted=year_substituted,
        p=prose,
        notes=notes,
        provenance=provenance,
        removed_sentences=removed_sentences,
        cite=citer,
        **artifacts,
    )
    return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"


def finalize_citations(citer: Any, markdown: str) -> Tuple[str, List[SourceRef]]:
    """Resolves and renumbers markers; on failure strips them, says so, and drops the evidence note."""
    if not isinstance(citer, ReportCiter):
        return strip_markers(markdown), []
    try:
        markdown, sources = citer.registry.finalize(markdown)
    except Exception as e:
        logger.exception(f"Citation finalize failed ({e!r}); publishing without filing markers.")
        markdown = strip_markers(markdown).replace(EVIDENCE_NOTE, NO_CITATIONS_NOTE)
        return markdown.rstrip("\n") + "\n\n" + CITATIONS_FAILED_NOTE + "\n", []
    if not any(s.kind != "market" for s in sources):
        markdown = markdown.replace(EVIDENCE_NOTE, NO_CITATIONS_NOTE)
    return markdown, sources


def assemble_report(
    ticker: str,
    company_name: str,
    fiscal_year: int,
    dcf_valuation: Optional[DCFValuationOutput],
    prose: Dict[str, Optional[str]],
    provenance: SynthesisProvenance,
    year_substituted: bool,
    markdown: str,
    sources: List[SourceRef],
) -> Final10KResearchReport:
    """The published output: the valuation headline (when the route valued), the thesis prose and the report."""
    val = dcf_valuation
    return Final10KResearchReport(
        ticker=ticker,
        company_name=company_name,
        fiscal_year=fiscal_year,
        implied_fair_value_per_share=val.implied_fair_value_per_share if val else None,
        current_share_price=val.current_share_price if val else None,
        upside_downside_pct=val.upside_downside_pct if val else None,
        valuation_stance=val.valuation_stance if val else None,
        three_pillar_thesis=ThreePillarThesis(
            pillar_1_business_moat=prose["pillar_1_business_moat"],
            pillar_2_financial_durability=prose["pillar_2_financial_durability"],
            pillar_3_valuation_asymmetry=prose["pillar_3_valuation_asymmetry"],
        ),
        executive_summary=prose["executive_summary"],
        synthesis_provenance=provenance,
        year_substituted=year_substituted,
        full_markdown_report=markdown,
        sources=sources,
    )
