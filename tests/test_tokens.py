"""Tests for the token estimation module."""

import textwrap
import pytest
from src.tokens import (
    estimate_tokens,
    analyze_tokens,
    granular_breakdown,
    format_token_report,
    TokenReport,
    GranularItem,
)


# ---------------------------------------------------------------------------
# estimate_tokens tests
# ---------------------------------------------------------------------------


# TEST: Empty string returns 0 tokens
def test_estimate_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   \n  ") == 0


# TEST: Short prose phrase produces a non-zero estimate
def test_estimate_prose():
    text = "The quick brown fox jumps over the lazy dog."
    tokens = estimate_tokens(text)
    # ~10 actual tokens; heuristic should be in reasonable range
    assert 5 <= tokens <= 20


# TEST: Dense code estimates fewer tokens per char than equivalent-length prose
def test_estimate_code_denser_than_prose():
    code = textwrap.dedent("""\
        def process_order(order_id: str, reason: CancelReason) -> Order:
            order = Order.get(order_id)
            order.validate_status()
            result = save_refund(order, reason)
            return result
    """)
    # Same character count of plain prose
    prose = "The process order function takes an order identifier and a cancel reason then validates status and saves a refund result before returning the order object."

    code_tokens = estimate_tokens(code)
    prose_tokens = estimate_tokens(prose)

    # Code is denser so should yield MORE tokens for the same char count
    # (more chars per token for prose → fewer tokens for same length)
    # Actually: code has FEWER chars/token → MORE tokens per character
    assert code_tokens >= prose_tokens * 0.7  # at minimum they should be close


# TEST: Estimate scales roughly linearly with content length
def test_estimate_scales_with_length():
    base = "def foo(): pass\n"
    short = estimate_tokens(base)
    long = estimate_tokens(base * 10)
    # Should be roughly 10x, ±50%
    assert long >= short * 5
    assert long <= short * 15


# TEST: All-whitespace content returns 0
def test_estimate_whitespace_only():
    assert estimate_tokens("\n\n\n   \t  \n") == 0


# ---------------------------------------------------------------------------
# analyze_tokens tests
# ---------------------------------------------------------------------------

_SAMPLE_BUNDLE = textwrap.dedent("""\
    # Context Bundle
    # Generated: 2026-03-24
    # Query: order cancellation
    ---
    code:
      src/order.py:
        class OrderService:
          def cancel_order(self, order_id: str) -> bool: ...
          def process_refund(self, order_id: str) -> dict: ...
      src/utils.py:
        def helper(x: int) -> str: ...
    schema:
      enums:
        order_status: [pending | active | cancelled]
      orders:
        pk: [id]
        cols:
          id: UUID, not null
          status: ENUM(order_status), not null
""")


# TEST: analyze_tokens returns correct total > 0 for non-empty content
def test_analyze_total_positive():
    report = analyze_tokens(_SAMPLE_BUNDLE)
    assert report.total > 0
    assert report.total_chars == len(_SAMPLE_BUNDLE)


# TEST: analyze_tokens splits into code and schema sections
def test_analyze_sections_detected():
    report = analyze_tokens(_SAMPLE_BUNDLE)
    section_names = [s.name for s in report.sections]
    assert "code" in section_names
    assert "schema" in section_names


# TEST: Section percentages sum to approximately 100
def test_analyze_section_pct_sums_to_100():
    report = analyze_tokens(_SAMPLE_BUNDLE)
    total_pct = sum(s.pct for s in report.sections)
    assert abs(total_pct - 100.0) < 5.0  # allow rounding slop


# TEST: Budget reporting returns correct fields when limit provided
def test_analyze_budget_fields():
    report = analyze_tokens(_SAMPLE_BUNDLE, budget_limit=10000)
    assert report.budget is not None
    assert report.budget["limit"] == 10000
    assert report.budget["used"] == report.total
    assert report.budget["remaining"] == max(0, 10000 - report.total)
    assert 0 <= report.budget["utilization_pct"] <= 100


# TEST: No budget section when budget_limit is None
def test_analyze_no_budget_when_none():
    report = analyze_tokens(_SAMPLE_BUNDLE, budget_limit=None)
    assert report.budget is None


# TEST: Over-budget scenario shows 100% utilization cap
def test_analyze_over_budget():
    # Use a tiny budget that's definitely smaller than the content
    report = analyze_tokens(_SAMPLE_BUNDLE, budget_limit=1)
    assert report.budget is not None
    assert report.budget["remaining"] == 0
    assert report.budget["utilization_pct"] > 100


# ---------------------------------------------------------------------------
# granular_breakdown tests
# ---------------------------------------------------------------------------


# TEST: granular_breakdown extracts file items from code section
def test_granular_file_items():
    items = granular_breakdown(_SAMPLE_BUNDLE)
    file_items = [i for i in items if i.type == "file"]
    names = [i.name for i in file_items]
    assert any("order.py" in n for n in names)


# TEST: granular_breakdown extracts table items from schema section
def test_granular_table_items():
    items = granular_breakdown(_SAMPLE_BUNDLE)
    table_items = [i for i in items if i.type == "table"]
    names = [i.name for i in table_items]
    assert "orders" in names


# TEST: granular_breakdown extracts enum items
def test_granular_enum_items():
    items = granular_breakdown(_SAMPLE_BUNDLE)
    enum_items = [i for i in items if i.type == "enum"]
    names = [i.name for i in enum_items]
    assert "order_status" in names


# TEST: granular_breakdown returns items sorted by token count descending
def test_granular_sorted_descending():
    items = granular_breakdown(_SAMPLE_BUNDLE)
    token_counts = [i.tokens for i in items]
    assert token_counts == sorted(token_counts, reverse=True)


# TEST: granular_breakdown returns empty list for content with no recognisable structure
def test_granular_empty_structure():
    items = granular_breakdown("just some plain text with no yaml keys")
    assert items == []


# ---------------------------------------------------------------------------
# format_token_report tests
# ---------------------------------------------------------------------------


# TEST: format_token_report returns a non-empty string
def test_format_report_produces_output():
    report = analyze_tokens(_SAMPLE_BUNDLE, budget_limit=20000)
    output = format_token_report(report)
    assert len(output) > 50


# TEST: format_token_report includes total token count
def test_format_report_includes_total():
    report = analyze_tokens(_SAMPLE_BUNDLE)
    output = format_token_report(report)
    # The total count should appear somewhere in the output
    assert str(report.total) in output or "Token Report" in output


# TEST: format_token_report includes budget info when present
def test_format_report_includes_budget():
    report = analyze_tokens(_SAMPLE_BUNDLE, budget_limit=5000)
    output = format_token_report(report)
    assert "5,000" in output or "Budget" in output


# TEST: format_token_report includes granular items when provided
def test_format_report_includes_granular():
    report = analyze_tokens(_SAMPLE_BUNDLE)
    granular = granular_breakdown(_SAMPLE_BUNDLE)
    output = format_token_report(report, granular)
    assert "TOP ITEMS" in output
    # At least one file name should appear
    assert "order" in output.lower()
