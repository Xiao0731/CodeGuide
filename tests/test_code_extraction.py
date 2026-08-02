from src.data.code_validator import extract_code


def test_extract_code_prefers_final_explicit_python_over_pseudocode():
    response = """Example:
```text
读取输入：然后输出
```

```python
print(input())
```
"""
    assert extract_code(response) == "print(input())"


def test_extract_code_uses_last_python_fence():
    response = """```python
x = 1
```
Explanation.
```python
x = 2
print(x)
```
"""
    assert extract_code(response) == "x = 2\nprint(x)"


def test_extract_code_falls_back_to_untagged_fence():
    assert extract_code("```\nprint(42)\n```") == "print(42)"


def test_call_based_prefers_function_over_trailing_example():
    response = """```python
def find(values):
    return len(values)
```

Example:
```python
print(find([]))
```"""
    assert extract_code(response, io_mode="call_based", fn_name="find").startswith("def find")


def test_standard_input_prefers_complete_program():
    response = """```python
print(solve_example())
```

```python
n = int(input())
answer = n * 2
print(answer)
```"""
    assert extract_code(response, io_mode="standard_input") == (
        "n = int(input())\nanswer = n * 2\nprint(answer)"
    )


def test_single_code_block_keeps_compatibility_with_context():
    code = "print(find([]))"
    assert extract_code(
        f"```python\n{code}\n```", io_mode="call_based", fn_name="find"
    ) == code
