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
