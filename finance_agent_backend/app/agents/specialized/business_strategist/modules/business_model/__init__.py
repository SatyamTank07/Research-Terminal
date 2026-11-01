"""What does it sell, to whom, and how does it make money?"""

from app.agents.specialized.business_strategist.modules.business_model.schema import SegmentDetail
from app.agents.specialized.business_strategist.modules.business_model.verify import BusinessModel, verify_business_model

__all__ = ["BusinessModel", "SegmentDetail", "verify_business_model"]
