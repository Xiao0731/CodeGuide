try:
    from .composite import CompositeReward, RewardOutput
    from .correctness import CodeCorrectnessReward
    from .format import FormatComplianceReward
    from .teaching import TeachingCompletenessReward
except ModuleNotFoundError:
    CompositeReward = None
    RewardOutput = None
    CodeCorrectnessReward = None
    FormatComplianceReward = None
    TeachingCompletenessReward = None

__all__ = [
    "CompositeReward",
    "RewardOutput",
    "CodeCorrectnessReward",
    "FormatComplianceReward",
    "TeachingCompletenessReward",
]
