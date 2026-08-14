from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load_module(
    "generate_evalplus_code_capability",
    ROOT / "scripts" / "generate_evalplus_code_capability.py",
)
class EvalPlusCodeCapabilityTests(unittest.TestCase):
    def test_extract_python_solution_from_repeated_fence(self):
        raw = "```python\ndef add(a, b):\n    return a + b\n```\nExplanation"
        self.assertEqual(
            generator.extract_python_solution(raw),
            "def add(a, b):\n    return a + b\n",
        )

    def test_extract_python_solution_from_prefixed_assistant_body(self):
        raw = "def add(a, b):\n    return a + b\n```"
        self.assertEqual(
            generator.extract_python_solution(raw),
            "def add(a, b):\n    return a + b\n",
        )

    def test_syntax_error(self):
        self.assertIsNone(generator.syntax_error("def f():\n    return 1\n"))
        self.assertIsNotNone(generator.syntax_error("def f(:\n"))

if __name__ == "__main__":
    unittest.main()
