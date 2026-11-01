"""The audit inputs every module reads: annual line items, the latest and prior balance sheets, and the window guards.

All dollar amounts in $ millions, shares in millions, CapEx as a positive magnitude. The field names, aliases
and descriptions form the `audit_financial_metrics_tool` schema the LLM transcribes into in llm_fallback mode.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

SHARE_SCALE_LIMIT = 100_000.0


def _validate_share_scale(v: Optional[float], name: str) -> Optional[float]:
    """Rejects share counts passed in thousands or raw units, and non-positive counts."""
    if v is None:
        return None
    if v > SHARE_SCALE_LIMIT:
        raise ValueError(
            f"{name} ({v:,.2f}) appears to be passed in thousands or raw units. "
            f"All shares must be standardized to MILLIONS (e.g. 15004.7 instead of 15004697). "
            f"Please divide by 1,000 (or 1,000,000) and retry."
        )
    if v <= 0:
        raise ValueError(f"{name} must be strictly positive (> 0), got: {v}")
    return float(v)


class AnnualFinancialInput(BaseModel):
    """Annual statement line items for one fiscal year ($ millions; shares in millions)."""

    fiscal_year: int = Field(..., validation_alias=AliasChoices("fiscal_year", "year", "fy"))
    revenue: float = Field(
        ..., description="Total net sales / total revenues",
        validation_alias=AliasChoices("revenue", "total_revenue", "net_sales", "total_net_sales", "sales"),
    )
    gross_profit: Optional[float] = Field(
        None, description="Reported gross profit; derived from cost_of_revenue when omitted",
        validation_alias=AliasChoices("gross_profit", "gross_margin", "gross_income"),
    )
    cost_of_revenue: Optional[float] = Field(
        None, description="Total cost of revenues / cost of sales",
        validation_alias=AliasChoices("cost_of_revenue", "cost_of_sales", "cogs", "total_cost_of_revenues"),
    )
    operating_expenses: Optional[float] = Field(
        None, description="Total operating expenses (below gross profit)",
        validation_alias=AliasChoices("operating_expenses", "total_operating_expenses", "opex"),
    )
    operating_income: float = Field(
        ..., description="Operating income / EBIT",
        validation_alias=AliasChoices("operating_income", "ebit", "operating_profit", "income_from_operations"),
    )
    interest_income: Optional[float] = Field(None, validation_alias=AliasChoices("interest_income"))
    interest_expense: Optional[float] = Field(
        None, description="Gross interest expense as a POSITIVE magnitude (None when not reported separately)",
        validation_alias=AliasChoices("interest_expense"),
    )
    pretax_income: Optional[float] = Field(
        None, description="Income before income taxes",
        validation_alias=AliasChoices(
            "pretax_income", "income_before_taxes", "income_before_provision_for_income_taxes",
            "income_before_income_tax", "earnings_before_taxes",
        ),
    )
    income_tax_expense: Optional[float] = Field(
        None, description="Provision for (benefit from) income taxes; negative for a benefit",
        validation_alias=AliasChoices(
            "income_tax_expense", "provision_for_income_taxes", "tax_expense", "income_tax_provision",
        ),
    )
    net_income: float = Field(
        ..., description="Net income attributable to common shareholders",
        validation_alias=AliasChoices("net_income", "net_income_attributable", "net_earnings", "net_profit"),
    )
    net_income_total: Optional[float] = Field(
        None, description="Consolidated net income including noncontrolling interests",
        validation_alias=AliasChoices("net_income_total", "consolidated_net_income"),
    )
    operating_cash_flow: float = Field(
        ..., description="Net cash provided by operating activities",
        validation_alias=AliasChoices(
            "operating_cash_flow", "cash_from_operations", "operating_activities_cash_flow",
            "cash_provided_by_operating_activities", "net_cash_provided_by_operating_activities",
        ),
    )
    capital_expenditures: float = Field(
        ..., description="Purchases of PP&E as a POSITIVE magnitude",
        validation_alias=AliasChoices(
            "capital_expenditures", "capex", "payments_for_acquisition_of_property_plant_and_equipment",
            "purchases_of_property_and_equipment", "additions_to_property_plant_and_equipment",
        ),
    )
    depreciation_amortization: Optional[float] = Field(
        None, validation_alias=AliasChoices("depreciation_amortization", "depreciation_and_amortization", "dna"),
    )
    stock_based_compensation: Optional[float] = Field(
        None, validation_alias=AliasChoices("stock_based_compensation", "share_based_compensation", "sbc"),
    )
    deferred_income_taxes: Optional[float] = Field(
        None, description="Deferred income taxes adjustment from the cash flow statement",
        validation_alias=AliasChoices("deferred_income_taxes", "deferred_taxes"),
    )
    accounts_receivable: Optional[float] = Field(
        None, validation_alias=AliasChoices("accounts_receivable", "ar", "accounts_receivable_net"),
    )
    inventories: Optional[float] = Field(None, validation_alias=AliasChoices("inventories", "inventory"))
    accounts_payable: Optional[float] = Field(None, validation_alias=AliasChoices("accounts_payable", "ap"))
    total_assets: Optional[float] = Field(None, validation_alias=AliasChoices("total_assets"))
    diluted_weighted_shares: Optional[float] = Field(
        None, description="Weighted-average diluted shares (millions)",
        validation_alias=AliasChoices("diluted_weighted_shares", "weighted_diluted_shares"),
    )
    period_end_shares_outstanding: Optional[float] = Field(
        None, description="Common shares outstanding at fiscal year end (millions)",
        validation_alias=AliasChoices("period_end_shares_outstanding", "shares_outstanding_period_end"),
    )

    @field_validator("capital_expenditures")
    @classmethod
    def validate_capex_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"capital_expenditures ({v:,.2f}) must be passed as a positive magnitude "
                f"representing cash outflows (e.g. 12715.0 instead of -12715.0). "
                f"If this line item represents proceeds from asset sales, do not pass it as CapEx."
            )
        return float(v)

    @field_validator("diluted_weighted_shares", "period_end_shares_outstanding")
    @classmethod
    def validate_shares(cls, v: Optional[float], info) -> Optional[float]:
        return _validate_share_scale(v, info.field_name)


class BalanceSheetInput(BaseModel):
    """Balance sheet snapshot for the latest fiscal year ($ millions; shares in millions)."""

    fiscal_year: int = Field(..., validation_alias=AliasChoices("fiscal_year", "year", "fy"))
    cash_and_equivalents: float = Field(
        ..., ge=0.0, validation_alias=AliasChoices("cash_and_equivalents", "cash", "cash_and_cash_equivalents"),
    )
    marketable_securities: Optional[float] = Field(
        0.0, ge=0.0,
        description="Current + non-current liquid marketable securities / short-term investments",
        validation_alias=AliasChoices("marketable_securities", "short_term_investments", "liquid_investments"),
    )
    short_term_debt: Optional[float] = Field(
        0.0, ge=0.0,
        validation_alias=AliasChoices(
            "short_term_debt", "commercial_paper", "current_debt", "current_portion_of_debt",
            "current_portion_of_term_debt",
        ),
    )
    long_term_debt: Optional[float] = Field(
        0.0, ge=0.0, validation_alias=AliasChoices("long_term_debt", "term_debt", "non_current_debt"),
    )
    stockholders_equity: float = Field(
        ..., description="Total stockholders' equity attributable to the parent (can be negative)",
        validation_alias=AliasChoices("stockholders_equity", "shareholders_equity", "equity"),
    )
    noncontrolling_interests: Optional[float] = Field(None, validation_alias=AliasChoices("noncontrolling_interests"))
    weighted_diluted_shares: Optional[float] = Field(
        None, description="Latest-year weighted-average diluted shares (millions)",
        validation_alias=AliasChoices("weighted_diluted_shares", "diluted_weighted_shares", "diluted_shares"),
    )
    period_end_shares_outstanding: Optional[float] = Field(
        None, description="Common shares outstanding at the balance sheet date (millions)",
        validation_alias=AliasChoices("period_end_shares_outstanding", "shares_outstanding"),
    )
    current_assets: Optional[float] = Field(
        None, ge=0.0, validation_alias=AliasChoices("current_assets", "total_current_assets"),
    )
    current_liabilities: Optional[float] = Field(
        None, ge=0.0, validation_alias=AliasChoices("current_liabilities", "total_current_liabilities"),
    )
    total_assets: Optional[float] = Field(None, ge=0.0, validation_alias=AliasChoices("total_assets"))

    @field_validator("marketable_securities", "short_term_debt", "long_term_debt", mode="before")
    @classmethod
    def coerce_none_to_zero(cls, v: Any) -> Any:
        return 0.0 if v is None else v

    @field_validator("weighted_diluted_shares", "period_end_shares_outstanding")
    @classmethod
    def validate_shares(cls, v: Optional[float], info) -> Optional[float]:
        return _validate_share_scale(v, info.field_name)

    @model_validator(mode="after")
    def require_a_share_count(self) -> "BalanceSheetInput":
        if self.weighted_diluted_shares is None and self.period_end_shares_outstanding is None:
            raise ValueError(
                "Provide weighted_diluted_shares (income statement, diluted weighted-average) and/or "
                "period_end_shares_outstanding (balance sheet common stock caption), in MILLIONS."
            )
        return self


class PriorBalanceSheetInput(BaseModel):
    """Prior fiscal-year balance sheet used for average invested capital / equity / assets."""

    fiscal_year: int = Field(..., validation_alias=AliasChoices("fiscal_year", "year", "fy"))
    cash_and_equivalents: float = Field(..., ge=0.0)
    marketable_securities: Optional[float] = Field(0.0, ge=0.0)
    short_term_debt: Optional[float] = Field(0.0, ge=0.0)
    long_term_debt: Optional[float] = Field(0.0, ge=0.0)
    stockholders_equity: float
    total_assets: Optional[float] = Field(None, ge=0.0)

    @field_validator("marketable_securities", "short_term_debt", "long_term_debt", mode="before")
    @classmethod
    def coerce_none_to_zero(cls, v: Any) -> Any:
        return 0.0 if v is None else v


def check_audit_window(
    annual_financials: List[AnnualFinancialInput],
    balance_sheet: BalanceSheetInput,
    prior_balance_sheet: Optional[PriorBalanceSheetInput] = None,
) -> None:
    """Rejects a gapped window and balance sheets that do not sit at the window's last (or prior) year."""
    if not annual_financials:
        raise ValueError("annual_financials must contain at least one fiscal year.")

    years = sorted(f.fiscal_year for f in annual_financials)
    expected_contiguous = list(range(years[0], years[-1] + 1))
    if years != expected_contiguous:
        raise ValueError(
            f"annual_financials must contain contiguous fiscal years with no gaps. "
            f"Supplied: {years}, expected: {expected_contiguous}"
        )
    if balance_sheet.fiscal_year != years[-1]:
        raise ValueError(
            f"balance_sheet.fiscal_year ({balance_sheet.fiscal_year}) must match the latest "
            f"annual_financials fiscal year ({years[-1]})."
        )
    if prior_balance_sheet is not None and prior_balance_sheet.fiscal_year != years[-1] - 1:
        raise ValueError(
            f"prior_balance_sheet.fiscal_year ({prior_balance_sheet.fiscal_year}) must equal "
            f"{years[-1] - 1} (the year before the latest balance sheet)."
        )


def validate_audit_inputs(
    inputs: Dict[str, Any],
) -> Tuple[List[AnnualFinancialInput], BalanceSheetInput, Optional[PriorBalanceSheetInput]]:
    """Validates plain-dict audit inputs (extraction or LLM transcription) into the input models."""
    prior = inputs.get("prior_balance_sheet")
    return (
        [AnnualFinancialInput.model_validate(r) for r in inputs["annual_financials"]],
        BalanceSheetInput.model_validate(inputs["balance_sheet"]),
        PriorBalanceSheetInput.model_validate(prior) if prior else None,
    )
