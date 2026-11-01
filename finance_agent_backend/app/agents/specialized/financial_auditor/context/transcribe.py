"""llm_fallback reconciliation: line items the LLM left out of its transcription are filled from parsed statements."""

from typing import Any, Dict, List

from app.agents.specialized.financial_auditor.context.audit_inputs import (
    AnnualFinancialInput,
    BalanceSheetInput,
    PriorBalanceSheetInput,
)
from app.agents.specialized.financial_auditor.context.extract import ExtractionResult

GAP_FILL_DISAGREEMENT_PCT = 1.0


def gap_fill_transcribed_inputs(args: Dict[str, Any], extraction: ExtractionResult) -> Dict[str, Any]:
    """Fills line items the LLM omitted (None, or 0 for securities/debt) with deterministically extracted values."""
    log: List[str] = []
    changed = False
    try:
        annual = [
            AnnualFinancialInput.model_validate(r).model_dump(exclude_none=True)
            for r in args.get("annual_financials", [])
        ]
        bs = BalanceSheetInput.model_validate(args["balance_sheet"]).model_dump(exclude_none=True)
        prior_raw = args.get("prior_balance_sheet")
        prior = PriorBalanceSheetInput.model_validate(prior_raw).model_dump(exclude_none=True) if prior_raw else None
    except Exception as e:
        return {"inputs": None, "adjustments": [], "log": [f"Gap-fill skipped: LLM inputs invalid ({e})."], "changed": False}

    extracted = {r["fiscal_year"]: r for r in extraction.to_annual_inputs_all_years()}
    zero_is_missing = {"marketable_securities", "short_term_debt", "long_term_debt"}

    def merge(record: Dict[str, Any], source: Dict[str, Any], label: str):
        nonlocal changed
        for fname, value in source.items():
            if fname == "fiscal_year" or value is None:
                continue
            current = record.get(fname)
            if current is None or (fname in zero_is_missing and current == 0.0 and value > 0):
                record[fname] = value
                changed = True
                log.append(f"Gap-filled {label}.{fname} = {value:,.2f} from parsed statements (LLM omitted it).")
            elif isinstance(current, (int, float)) and value:
                diff = abs(current - value) / max(abs(value), 1e-9) * 100.0
                if diff > GAP_FILL_DISAGREEMENT_PCT:
                    log.append(
                        f"Disagreement on {label}.{fname}: LLM {current:,.2f} vs parsed {value:,.2f} "
                        f"({diff:.1f}%); LLM value kept."
                    )

    for record in annual:
        merge(record, extracted.get(record["fiscal_year"], {}), f"FY{record['fiscal_year']}")
    merge(bs, extraction.to_balance_sheet_input(), f"FY{bs['fiscal_year']}.balance_sheet")
    extracted_prior = extraction.to_prior_balance_sheet_input()
    if prior is None and extracted_prior and extracted_prior["fiscal_year"] == bs["fiscal_year"] - 1:
        prior = extracted_prior
        changed = True
        log.append(f"Added FY{prior['fiscal_year']} prior balance sheet from parsed statements.")

    return {
        "inputs": {"annual_financials": annual, "balance_sheet": bs, "prior_balance_sheet": prior},
        "adjustments": args.get("normalization_adjustments") or [],
        "log": log,
        "changed": changed,
    }
