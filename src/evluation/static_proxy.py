"""Reproduce the external CodeGuide project's no-test AST proxy.

IMPORTANT:
- This is a structural/static proxy only.
- It is NOT correctness and must never be reported as Pass@1.
- The weights intentionally match the external implementation exactly.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StaticProxyResult:
    """Breakdown of the five-dimensional external static proxy."""

    score: float
    syntax_valid: bool
    has_function: bool
    has_comments_or_docstring: bool
    reasonable_line_count: bool
    no_pass_only_function: bool
    non_empty_lines: int
    comment_lines: int

    def to_dict(self) -> dict:
        return asdict(self)


def score_external_static_proxy(code: str) -> StaticProxyResult:
    """Score code using the external project's exact five dimensions.

    Weights:
      +0.30 AST parse succeeds
      +0.20 at least one function definition
      +0.20 at least two comment lines OR any docstring-like string expression
      +0.20 10-100 non-empty lines
      +0.10 no function whose sole body statement is ``pass``
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return StaticProxyResult(
            score=0.0,
            syntax_valid=False,
            has_function=False,
            has_comments_or_docstring=False,
            reasonable_line_count=False,
            no_pass_only_function=False,
            non_empty_lines=sum(1 for line in code.splitlines() if line.strip()),
            comment_lines=sum(
                1 for line in code.splitlines() if line.strip().startswith("#")
            ),
        )

    score = 0.30

    has_function = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(tree)
    )
    if has_function:
        score += 0.20

    lines = code.splitlines()
    comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
    has_docstring = any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for node in ast.walk(tree)
    )
    has_comments_or_docstring = comment_lines >= 2 or has_docstring
    if has_comments_or_docstring:
        score += 0.20

    non_empty_lines = sum(1 for line in lines if line.strip())
    reasonable_line_count = 10 <= non_empty_lines <= 100
    if reasonable_line_count:
        score += 0.20

    has_pass_only = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                has_pass_only = True
                break
    no_pass_only_function = not has_pass_only
    if no_pass_only_function:
        score += 0.10

    return StaticProxyResult(
        score=round(min(score, 1.0), 4),
        syntax_valid=True,
        has_function=has_function,
        has_comments_or_docstring=has_comments_or_docstring,
        reasonable_line_count=reasonable_line_count,
        no_pass_only_function=no_pass_only_function,
        non_empty_lines=non_empty_lines,
        comment_lines=comment_lines,
    )
