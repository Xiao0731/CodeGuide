#!/usr/bin/env python3
"""Minimal resumable Docker verification for the downloaded TACO-100 trajectory.

Only semantically relevant invariants are checked:
- frozen protocol content SHA256
- exactly 100 selected IDs
- exactly 35 complete generation files
- digest-pinned Docker image

Platform-dependent manifest fields such as path separators and local Git commit
are intentionally ignored during offline verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json