#!/usr/bin/env python3
"""最小版 GRPO 入口：复用现有训练器，只替换奖励函数。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reward.minimal_grpo_reward import make_reward_fn_with_cfg
from src.training import grpo_train


def main() -> None:
    # 只替换奖励函数，Curriculum、BestCheckpoint、日志和模型加载继续复用现有实现。
    grpo_train.make_reward_fn_with_cfg = make_reward_fn_with_cfg
    grpo_train.main()


if __name__ == "__main__":
    main()
