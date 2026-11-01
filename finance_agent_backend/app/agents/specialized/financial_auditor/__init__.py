from app.agents.pipeline.contract import AgentSpec
from app.agents.specialized.financial_auditor.agent_financial_auditor import (
    FinancialAuditorAgent,
    audit_financial_statements,
    summarize_audit,
)
from app.agents.specialized.financial_auditor.modules.earnings_normalization import NormalizationAdjustment
from app.agents.specialized.financial_auditor.modules.tax_quality import STATUTORY_TAX_RATE
from app.agents.specialized.financial_auditor.state_financial_auditor import (
    AuditDataQuality,
    BalanceSheetSnapshot,
    FinancialAuditOutput,
    YearFinancials,
)

AGENT = AgentSpec(
    name="financial_auditor",
    label="Financial Auditor",
    output_key="financial_audit",
    run=audit_financial_statements,
    milestone=summarize_audit,
)

__all__ = [
    "AGENT",
    "FinancialAuditorAgent",
    "FinancialAuditOutput",
    "AuditDataQuality",
    "YearFinancials",
    "BalanceSheetSnapshot",
    "NormalizationAdjustment",
    "STATUTORY_TAX_RATE",
]
