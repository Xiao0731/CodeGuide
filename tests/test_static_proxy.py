from src.evaluation.static_proxy import score_external_static_proxy


def test_syntax_error_scores_zero():
    result = score_external_static_proxy("def broken(:\n    pass")
    assert result.score == 0.0
    assert not result.syntax_valid


def test_short_script_scores_ast_plus_no_pass_only():
    result = score_external_static_proxy("x = 1\nprint(x)\n")
    assert result.score == 0.4
    assert result.syntax_valid
    assert not result.has_function
    assert result.no_pass_only_function


def test_full_proxy_score_matches_external_implementation():
    code = """\
# comment one
# comment two
def solve(values):
    total = 0
    for value in values:
        total += value
    if total > 0:
        return total
    return 0

answer = solve([1, 2, 3])
print(answer)
"""
    result = score_external_static_proxy(code)
    assert result.score == 1.0
    assert result.has_function
    assert result.has_comments_or_docstring
    assert result.reasonable_line_count
    assert result.no_pass_only_function


def test_pass_only_function_loses_last_dimension():
    result = score_external_static_proxy("def placeholder():\n    pass\n")
    assert result.score == 0.5
    assert not result.no_pass_only_function
