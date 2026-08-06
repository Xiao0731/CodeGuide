from __future__ import annotations

import unittest

from src.reward.minimal_grpo_reward import (
    RewardWeights,
    combine_scores,
    static_validity_score,
)


class MinimalGrpoRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = RewardWeights()
        self.weights.validate()

    def test_invalid_python_has_no_static_reward(self):
        score = static_validity_score("def f(:\n", {"io_mode": "standard_input"})
        self.assertEqual(score, 0.0)

    def test_call_based_requires_expected_function(self):
        metadata = {"io_mode": "call_based", "fn_name": "solve"}
        self.assertEqual(static_validity_score("def other():\n    return 1\n", metadata), 0.0)
        self.assertEqual(static_validity_score("def solve():\n    return 1\n", metadata), 1.0)

    def test_all_wrong_format_is_strongly_gated(self):
        result = combine_scores(
            static_validity=1.0,
            pass_rate=0.0,
            contract=1.0,
            weights=self.weights,
        )
        self.assertAlmostEqual(result.code, 0.05)
        self.assertAlmostEqual(result.gated_contract, 0.25)
        self.assertAlmostEqual(result.total, 0.13)

    def test_partial_pass_keeps_dense_signal(self):
        low = combine_scores(
            static_validity=1.0,
            pass_rate=0.2,
            contract=0.8,
            weights=self.weights,
        )
        high = combine_scores(
            static_validity=1.0,
            pass_rate=0.8,
            contract=0.8,
            weights=self.weights,
        )
        self.assertGreater(high.total, low.total)
        self.assertGreater(high.contract_gate, low.contract_gate)

    def test_strict_pass_gets_full_bonus(self):
        result = combine_scores(
            static_validity=1.0,
            pass_rate=1.0,
            contract=1.0,
            weights=self.weights,
        )
        self.assertAlmostEqual(result.code, 1.0)
        self.assertAlmostEqual(result.total, 1.0)


if __name__ == "__main__":
    unittest.main()
