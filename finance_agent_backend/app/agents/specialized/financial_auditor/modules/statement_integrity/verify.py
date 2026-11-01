"""Screens the auditor's proposed input corrections: only known fields on an existing record are applied."""

import copy
from typing import Any, Dict, List, Optional

from app.agents.specialized.financial_auditor.context.audit_inputs import (
    AnnualFinancialInput,
    BalanceSheetInput,
    PriorBalanceSheetInput,
)
from app.agents.specialized.financial_auditor.modules.statement_integrity.schema import InputCorrection

_CORRECTION_MODELS = {
    "annual": AnnualFinancialInput,
    "balance_sheet": BalanceSheetInput,
    "prior_balance_sheet": PriorBalanceSheetInput,
}


def apply_input_corrections(
    base_inputs: Dict[str, Any],
    corrections: Optional[List[InputCorrection]],
) -> Dict[str, Any]:
    """
    Applies cited corrections to extracted audit inputs (deep-copied).

    Returns {"inputs": corrected kwargs, "applied": [...], "rejected": [...]}.
    """
    inputs = copy.deepcopy(base_inputs)
    applied, rejected = [], []
    for corr in corrections or []:
        model = _CORRECTION_MODELS[corr.target]
        if corr.field not in model.model_fields or corr.field == "fiscal_year":
            rejected.append({**corr.model_dump(), "rejection": f"unknown field '{corr.field}'"})
            continue
        if corr.target == "annual":
            record = next((r for r in inputs["annual_financials"] if r.get("fiscal_year") == corr.fiscal_year), None)
        else:
            record = inputs.get(corr.target)
            if record is not None and record.get("fiscal_year") != corr.fiscal_year:
                record = None
        if record is None:
            rejected.append({**corr.model_dump(), "rejection": f"no {corr.target} record for FY{corr.fiscal_year}"})
            continue
        previous = record.get(corr.field)
        record[corr.field] = corr.value
        applied.append({**corr.model_dump(), "previous_value": previous})
    return {"inputs": inputs, "applied": applied, "rejected": rejected}
