"""How much must it reinvest to fund that growth? CapEx fades to the steady state where net reinvestment = g / RONIC."""

from app.agents.specialized.forecasting_analyst.modules.reinvestment_path.calculate import apply_reinvestment_path

__all__ = ["apply_reinvestment_path"]
