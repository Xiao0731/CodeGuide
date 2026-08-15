"""CodeGuide execution and teaching-contract rewards."""

from .execution import VerificationResult, verify_code
from .format import FormatComplianceReward, contract_score
from .grpo import (
    RewardBreakdown,
    RewardWeights,
    build_composite_reward,
    combine_scores,
    static_validity_score,
)

__all__ = [
    "FormatComplianceReward",
    "VerificationResult",
    "RewardBreakdown",
    "RewardWeights",
    "build_composite_reward",
    "combine_scores",
    "contract_score",
    "static_validity_score",
    "verify_code",
]
