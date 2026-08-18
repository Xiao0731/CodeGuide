#!/usr/bin/env python3
"""Canonical command-line entry point for CodeGuide GRPO training.

The implementation lives in :mod:`src.training.grpo_train`. Keeping this
file as a thin wrapper prevents historical entry points from diverging.

TRL 0.22.2 already records ``reward_std`` and ``frac_reward_zero_std`` for
GRPO groups. This wrapper adds only a warning policy on top of those native
metrics; it does not recompute rewards or alter the optimization objective.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.grpo_train import main

LOGGER = logging.getLogger("codeguide.grpo")


def _install_reward_collapse_warning() -> None:
    """Warn when TRL reports too many zero-advantage groups for many log windows."""
    import trl

    base_trainer = trl.GRPOTrainer
    ratio_threshold = float(
        os.environ.get("CODEGUIDE_COLLAPSE_RATIO_THRESHOLD", "0.5")
    )
    patience = int(os.environ.get("CODEGUIDE_COLLAPSE_PATIENCE", "20"))
    if not 0.0 <= ratio_threshold <= 1.0:
        raise ValueError("CODEGUIDE_COLLAPSE_RATIO_THRESHOLD must be in [0, 1]")
    if patience <= 0:
        raise ValueError("CODEGUIDE_COLLAPSE_PATIENCE must be positive")

    class MonitoredGRPOTrainer(base_trainer):
        """Add warnings while preserving TRL's native GRPO metrics and behavior."""

        _codeguide_zero_std_streak = 0

        def log(self, logs: dict[str, float], start_time=None) -> None:
            if self.model.training:
                values = self._metrics["train"].get("frac_reward_zero_std", [])
                if values:
                    ratio = float(sum(values) / len(values))
                    reward_std_values = self._metrics["train"].get("reward_std", [])
                    reward_std = (
                        float(sum(reward_std_values) / len(reward_std_values))
                        if reward_std_values
                        else float("nan")
                    )
                    if ratio > ratio_threshold:
                        self._codeguide_zero_std_streak += 1
                    else:
                        self._codeguide_zero_std_streak = 0

                    if (
                        self._codeguide_zero_std_streak == patience
                        and self.accelerator.is_main_process
                    ):
                        LOGGER.warning(
                            "GRPO group-reward collapse warning: "
                            "frac_reward_zero_std=%.3f > %.3f for %d consecutive "
                            "logging windows (reward_std=%.6f). This means many "
                            "prompts have identical rewards across their sampled "
                            "completions, so the reward-derived relative advantage "
                            "is zero for those groups. Check task difficulty, reward "
                            "saturation/sparsity and generation diversity.",
                            ratio,
                            ratio_threshold,
                            patience,
                            reward_std,
                        )
            return super().log(logs, start_time=start_time)

    # src.training.grpo_train imports GRPOTrainer lazily inside main(), so
    # replacing the exported class here affects only this canonical invocation.
    trl.GRPOTrainer = MonitoredGRPOTrainer


if __name__ == "__main__":
    # Keep --validate-only lightweight: it intentionally validates frozen
    # contracts without importing the GPU training stack.
    if "--validate-only" not in sys.argv:
        _install_reward_collapse_warning()
    main()
