"""Customer-concentration and pricing sentences from the rest of the filing (MD&A, notes, risk factors).

The judge sees them next to Item 1, and the concentration verify step falls back to them verbatim.
"""

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

from app.agents.shared.filing.sentences import normalize_text, split_sentences
from app.agents.specialized.business_strategist.context import FilingChunk
from app.agents.specialized.business_strategist.modules.market_power.schema import DisclosureSentence
from app.agents.specialized.business_strategist.modules.market_power.thresholds import (
    MAX_CONCENTRATION_SENTENCES,
    MAX_PRICING_SENTENCES,
    MAX_SENTENCE_CHARS,
)

logger = logging.getLogger("finance_agent.agents.business_strategist")

_CUSTOMER_WORD = re.compile(
    r"\bcustomers?\b|\bclients?\b|\bentit(?:y|ies)\b|\bdistributors?\b|\bpurchasers?\b|\bresellers?\b|\bwholesalers?\b|\bretailers?\b",
    re.IGNORECASE,
)
# A customer's share must be asserted with a verb ("accounted for 12%", "represented 22%",
# "no entity represented 10% or more"), not merely co-occur with a percentage in the sentence.
_SHARE_CLAIM = re.compile(
    r"(?:accounted\s+for|represent(?:ed|s|ing)?|comprised|constituted|made\s+up|amounted\s+to|exceed(?:ed|s)?"
    r"|more\s+than|in\s+excess\s+of)[^.;]{0,40}?(?:\d+(?:\.\d+)?\s?%|ten\s+percent)",
    re.IGNORECASE,
)
_REVENUE_WORD = re.compile(r"revenue|net\s+sales|\bsales\b", re.IGNORECASE)
_RECEIVABLE_WORD = re.compile(r"receivable", re.IGNORECASE)
_PRICING_WORD = re.compile(
    r"\bpric(?:e|es|ing)\b|average\s+selling\s+prices?|\basps?\b|pass(?:ed|ing)?\s+(?:through|on)\b", re.IGNORECASE
)
_PRICING_MOVE = re.compile(
    r"increas|decreas|reduc|higher|lower|premium|pressure|discount|raise|declin|pass|power|competitive|promotion|markdown",
    re.IGNORECASE,
)
_PRICING_EXCLUDE = re.compile(
    r"stock\s+price|share\s+price|price\s+of\s+our\s+(?:common\s+)?stock|exercise\s+price|purchase\s+price|strike\s+price"
    r"|transaction\s+price|selling\s+price\s+(?:basis|method)|standalone\s+selling|stand-alone\s+selling|fair\s+value"
    r"|forward-looking|price\s+risk|interest\s+rate|commodity\s+price|conversion\s+price|repurchase|offering\s+price",
    re.IGNORECASE,
)
# Input-cost sentences ("competitive pricing" from suppliers) are not about the company's own pricing,
# unless they say whether costs can be passed on to customers.
_INPUT_COST = re.compile(r"supplier|vendor|component|raw\s+material|wafer|commodit|procure", re.IGNORECASE)
_PASS_THROUGH = re.compile(r"pass(?:ed|ing)?\s+(?:through|on|along)|to\s+(?:our\s+)?customers", re.IGNORECASE)


def _is_pricing_sentence(sentence: str) -> bool:
    if not (_PRICING_WORD.search(sentence) and _PRICING_MOVE.search(sentence)) or _PRICING_EXCLUDE.search(sentence):
        return False
    return not _INPUT_COST.search(sentence) or bool(_PASS_THROUGH.search(sentence))


def _sentence_key(text: str) -> str:
    return re.sub(r"\s+%", "%", normalize_text(text))


def find_disclosure_sentences(
    chunks: Sequence[FilingChunk],
    items: Dict[str, str],
    exclude_chunk_ids: Optional[set] = None,
) -> List[DisclosureSentence]:
    """Customer-concentration and pricing sentences from narrative chunks outside `exclude_chunk_ids`.

    Concentration sentences are ranked revenue-based first, then receivables; within each kind
    sentences outside Item 1A (MD&A, notes) are preferred over risk-factor language.
    """
    exclude = exclude_chunk_ids or set()
    seen: set = set()
    found: List[Tuple[int, int, int, DisclosureSentence]] = []
    for chunk in chunks:
        if chunk.chunk_type != "narrative" or chunk.chunk_id in exclude:
            continue
        item = items.get(chunk.chunk_id) or chunk.item
        risk_rank = 1 if item == "Item 1A" else 0
        for sentence in split_sentences(chunk.content):
            key = _sentence_key(sentence)
            if key in seen:
                continue
            text = sentence[:MAX_SENTENCE_CHARS]
            if _CUSTOMER_WORD.search(sentence) and _SHARE_CLAIM.search(sentence):
                if _REVENUE_WORD.search(sentence):
                    kind, rank = "revenue_concentration", 0
                elif _RECEIVABLE_WORD.search(sentence):
                    kind, rank = "receivables_concentration", 1
                else:
                    kind = ""
                if kind:
                    seen.add(key)
                    found.append((rank, risk_rank, chunk.chunk_index, DisclosureSentence(chunk_id=chunk.chunk_id, item=item, text=text, kind=kind)))
                    continue
            if _is_pricing_sentence(sentence):
                seen.add(key)
                found.append((2, risk_rank, chunk.chunk_index, DisclosureSentence(chunk_id=chunk.chunk_id, item=item, text=text, kind="pricing")))
    found.sort(key=lambda x: (x[0], x[1], x[2]))
    concentration = [d for r, _, _, d in found if r < 2][:MAX_CONCENTRATION_SENTENCES]
    pricing = [d for r, _, _, d in found if r == 2][:MAX_PRICING_SENTENCES]
    return concentration + pricing


def gather_disclosures(filing: Sequence[FilingChunk], items: Dict[str, str], exclude: set) -> List[DisclosureSentence]:
    """The sentence screen over the whole filing outside `exclude`; an empty list when it fails."""
    try:
        return find_disclosure_sentences(filing, items, exclude_chunk_ids=exclude)
    except Exception as e:
        logger.warning(f"Disclosure sentence screen failed: {e}")
        return []
