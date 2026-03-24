"""Tests for the YAML compactor."""

import textwrap
import tempfile
from pathlib import Path
import pytest

from src.compactor import (
    compact_code,
    compact_all_code,
    compact_full,
    CompactOptions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, source: str) -> None:
    (tmp_path / name).write_text(textwrap.dedent(source).strip(), encoding="utf-8")


# ---------------------------------------------------------------------------
# compact_code tests
# ---------------------------------------------------------------------------


# TEST: compact_code returns YAML with 'code:' section header
def test_compact_code_has_header(tmp_path):
    _write(tmp_path, "quiz.py", """
        def generate_quiz(words: list) -> dict:
            \"\"\"Generate a quiz.\"\"\"
            pass
    """)
    result = compact_code(tmp_path, "quiz", top_k=5)
    assert result.content.startswith("code:")


# TEST: compact_code includes relevant file in output
def test_compact_code_includes_relevant_file(tmp_path):
    _write(tmp_path, "quiz.py", """
        def generate_quiz(words: list) -> dict:
            \"\"\"Generate a quiz.\"\"\"
            pass
    """)
    _write(tmp_path, "unrelated.py", """
        def load_config(): pass
    """)
    result = compact_code(tmp_path, "quiz")
    assert "quiz.py" in result.content
    assert "generate_quiz" in result.content


# TEST: compact_code populates file_names with included relative paths
def test_compact_code_file_names(tmp_path):
    _write(tmp_path, "quiz.py", "def quiz(): pass\n")
    result = compact_code(tmp_path, "quiz")
    assert any("quiz.py" in fn for fn in result.file_names)


# TEST: compact_code returns fallback message when no files match
def test_compact_code_no_match(tmp_path):
    _write(tmp_path, "app.py", "def main(): pass\n")
    result = compact_code(tmp_path, "quantum entanglement")
    assert "No files relevant" in result.content


# TEST: compact_code strips function bodies from output
def test_compact_code_bodies_stripped(tmp_path):
    _write(tmp_path, "order.py", """
        def cancel_order(order_id: str) -> bool:
            \"\"\"Cancel an order.\"\"\"
            db.update(order_id, status='cancelled')
            return True
    """)
    result = compact_code(tmp_path, "order cancel")
    assert "db.update" not in result.content
    assert "cancel_order" in result.content


# TEST: compact_code preserves class structure
def test_compact_code_class_structure(tmp_path):
    _write(tmp_path, "service.py", """
        class OrderService:
            \"\"\"Manages orders.\"\"\"
            def cancel(self, order_id: str) -> bool:
                \"\"\"Cancel an order.\"\"\"
                pass
    """)
    result = compact_code(tmp_path, "order service")
    assert "class OrderService" in result.content
    assert "def cancel" in result.content


# TEST: compact_code includes body hints for non-trivial functions
def test_compact_code_body_hints(tmp_path):
    _write(tmp_path, "processor.py", """
        def process_order(order_id: str) -> dict:
            \"\"\"Process an order.\"\"\"
            order = Order(order_id)
            result = validate(order)
            return result
    """)
    result = compact_code(tmp_path, "process order")
    # Body hint should appear as # → ... comment
    assert "→" in result.content or "# →" in result.content


# ---------------------------------------------------------------------------
# compact_all_code tests
# ---------------------------------------------------------------------------


# TEST: compact_all_code includes all non-test Python files
def test_compact_all_code_includes_all(tmp_path):
    _write(tmp_path, "module_a.py", "def func_a(): pass\n")
    _write(tmp_path, "module_b.py", "def func_b(): pass\n")
    _write(tmp_path, "test_module_a.py", "def test_a(): pass\n")  # should be excluded

    result = compact_all_code(tmp_path)
    assert "module_a.py" in result.content
    assert "module_b.py" in result.content
    assert "test_module_a.py" not in result.content


# TEST: compact_all_code skips __pycache__ directories
def test_compact_all_code_skips_pycache(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.py").write_text("x = 1\n")
    _write(tmp_path, "real.py", "def real(): pass\n")

    result = compact_all_code(tmp_path)
    assert "__pycache__" not in result.content


# ---------------------------------------------------------------------------
# compact_full tests
# ---------------------------------------------------------------------------


# TEST: compact_full produces a bundle with header, code section, token report
def test_compact_full_structure(tmp_path):
    _write(tmp_path, "quiz.py", """
        def generate_quiz(words: list) -> dict:
            \"\"\"Generate a quiz.\"\"\"
            pass
    """)

    opts = CompactOptions(code_path=tmp_path, query="quiz")
    result = compact_full(opts)

    assert "# Context Bundle" in result.content
    assert "# Generated:" in result.content
    assert "# Query: quiz" in result.content
    assert "code:" in result.content
    assert "Token Report" in result.content


# TEST: compact_full includes query in header
def test_compact_full_query_in_header(tmp_path):
    _write(tmp_path, "auth.py", "def login(user: str): pass\n")
    opts = CompactOptions(code_path=tmp_path, query="authentication login")
    result = compact_full(opts)
    assert "authentication login" in result.content


# TEST: compact_full with all_code dumps everything (no search)
def test_compact_full_all_code(tmp_path):
    _write(tmp_path, "module_a.py", "def func_a(): pass\n")
    _write(tmp_path, "module_b.py", "def func_b(): pass\n")

    opts = CompactOptions(code_path=tmp_path, query="", all_code=True)
    result = compact_full(opts)

    assert "module_a.py" in result.content
    assert "module_b.py" in result.content


# TEST: compact_full token_report is populated and has total > 0
def test_compact_full_token_report(tmp_path):
    _write(tmp_path, "app.py", "def main(): pass\n")
    opts = CompactOptions(code_path=tmp_path, query="main")
    result = compact_full(opts)
    assert result.token_report is not None
    assert result.token_report.total > 0


# TEST: compact_full with budget includes utilization in output
def test_compact_full_budget(tmp_path):
    _write(tmp_path, "app.py", "def main(): pass\n")
    opts = CompactOptions(code_path=tmp_path, query="main", budget=10000)
    result = compact_full(opts)
    # Budget info should be in the appended token report comment
    assert "10,000" in result.content or "Budget" in result.content
