"""Tests for the FK-aware schema search module."""

import textwrap
from pathlib import Path
import pytest

from src.schema.schema import SchemaState, TableSchema, ColumnDef, extract_schema
from src.schema.schema_search import (
    score_table,
    propagate_fk_scores,
    cross_reference_code_to_schema,
    search_schema,
    compact_schema_to_yaml,
    TableScore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_state() -> SchemaState:
    """Build a small SchemaState fixture for search tests."""
    state = SchemaState()

    # orders table with FK to users and an enum column
    orders = TableSchema(name="orders")
    orders.columns["id"] = ColumnDef("id", "UUID", nullable=False, is_primary_key=True)
    orders.columns["user_id"] = ColumnDef("user_id", "UUID", nullable=False, references="users(id)")
    orders.columns["status"] = ColumnDef(
        "status", "TEXT", nullable=False,
        enum_values=["pending", "active", "cancelled"]
    )
    orders.primary_key = ["id"]
    state.tables["orders"] = orders

    # users table
    users = TableSchema(name="users")
    users.columns["id"] = ColumnDef("id", "UUID", nullable=False, is_primary_key=True)
    users.columns["email"] = ColumnDef("email", "TEXT", nullable=False)
    users.primary_key = ["id"]
    state.tables["users"] = users

    # refunds table with FK to orders
    refunds = TableSchema(name="refunds")
    refunds.columns["id"] = ColumnDef("id", "UUID", nullable=False, is_primary_key=True)
    refunds.columns["order_id"] = ColumnDef("order_id", "UUID", nullable=False, references="orders(id)")
    refunds.columns["amount"] = ColumnDef("amount", "DECIMAL", nullable=False)
    refunds.primary_key = ["id"]
    state.tables["refunds"] = refunds

    state.enums["order_status"] = ["pending", "active", "cancelled"]
    return state


# ---------------------------------------------------------------------------
# score_table
# ---------------------------------------------------------------------------


# TEST: score_table gives highest score to table whose name exactly matches query
def test_score_table_name_match():
    state = _build_state()
    from src.ast_fetcher.search import tokenize_query
    terms = tokenize_query("orders")
    ts = score_table(state.tables["orders"], terms, state)
    ts_users = score_table(state.tables["users"], terms, state)
    assert ts.score > ts_users.score
    assert ts.score > 0


# TEST: score_table detects enum value matches in columns
def test_score_table_enum_value_match():
    state = _build_state()
    from src.ast_fetcher.search import tokenize_query
    terms = tokenize_query("cancelled")
    ts = score_table(state.tables["orders"], terms, state)
    assert ts.score > 0
    assert any("enum val" in r for r in ts.reasons)


# TEST: score_table detects FK target match
def test_score_table_fk_target_match():
    state = _build_state()
    from src.ast_fetcher.search import tokenize_query
    terms = tokenize_query("user")
    ts = score_table(state.tables["orders"], terms, state)
    # orders has user_id → users(id), so "user" in FK target boosts it
    assert ts.score > 0


# TEST: score_table returns zero score for empty query terms
def test_score_table_empty_terms():
    state = _build_state()
    ts = score_table(state.tables["orders"], [], state)
    assert ts.score == 0.0


# TEST: score_table detects partial table name match via _ segments
def test_score_table_partial_name():
    state = _build_state()
    # Add a table with underscore name
    t = TableSchema(name="user_sessions")
    t.columns["id"] = ColumnDef("id", "UUID")
    state.tables["user_sessions"] = t

    from src.ast_fetcher.search import tokenize_query
    terms = tokenize_query("session")
    ts = score_table(t, terms, state)
    assert ts.score > 0


# ---------------------------------------------------------------------------
# propagate_fk_scores
# ---------------------------------------------------------------------------


# TEST: propagate_fk_scores boosts FK target tables
def test_propagate_fk_boosts_target():
    state = _build_state()
    # Start with only orders having a score
    scores = {"orders": TableScore(name="orders", score=10.0)}
    propagate_fk_scores(scores, state)
    # orders → users(id), so users should get a boost
    assert "users" in scores
    assert scores["users"].score > 0


# TEST: propagate_fk_scores applies reverse boost too
def test_propagate_fk_reverse_boost():
    state = _build_state()
    # users has a score, orders FKs to users → orders should get a reverse boost
    scores = {"users": TableScore(name="users", score=10.0)}
    propagate_fk_scores(scores, state)
    # orders has FK to users → reverse boost to orders
    assert "orders" in scores
    assert scores["orders"].score > 0


# TEST: propagate_fk_scores does not create infinite feedback loops
def test_propagate_fk_no_feedback_loop():
    state = _build_state()
    scores = {
        "orders": TableScore(name="orders", score=5.0),
        "users": TableScore(name="users", score=3.0),
    }
    initial_total = sum(ts.score for ts in scores.values())
    propagate_fk_scores(scores, state)
    # Total should increase (boosts added) but not explode
    final_total = sum(ts.score for ts in scores.values())
    assert final_total > initial_total
    assert final_total < initial_total * 10  # Not more than 10x growth


# ---------------------------------------------------------------------------
# cross_reference_code_to_schema
# ---------------------------------------------------------------------------


# TEST: cross_reference boosts tables whose names appear in code file names
def test_cross_reference_boosts_matching_tables():
    state = _build_state()
    scores: dict[str, TableScore] = {}
    cross_reference_code_to_schema(
        ["src/order.service.py", "src/user.controller.py"],
        scores,
        state,
    )
    assert "orders" in scores
    assert "users" in scores
    assert scores["orders"].score > 0
    assert scores["users"].score > 0


# TEST: cross_reference does not boost unrelated tables
def test_cross_reference_no_boost_unrelated():
    state = _build_state()
    scores: dict[str, TableScore] = {}
    cross_reference_code_to_schema(
        ["src/auth.service.py"],  # "auth" doesn't match any table
        scores,
        state,
    )
    # refunds, orders, users not mentioned → shouldn't be boosted by cross-ref
    assert "refunds" not in scores


# ---------------------------------------------------------------------------
# search_schema integration
# ---------------------------------------------------------------------------


# TEST: search_schema returns relevant tables sorted by score
def test_search_schema_returns_relevant():
    state = _build_state()
    results = search_schema(state, "order cancellation")
    names = [r.table_schema.name for r in results]
    assert "orders" in names
    # orders should be first or near-top
    assert names.index("orders") < 2


# TEST: search_schema propagates FK boost to related tables
def test_search_schema_fk_propagation():
    state = _build_state()
    # "refund" should directly match refunds table; orders (FK target) should also appear
    results = search_schema(state, "refund")
    names = [r.table_schema.name for r in results]
    assert "refunds" in names
    # orders should appear because refunds has FK to orders (FK propagation)
    assert "orders" in names


# TEST: search_schema with code_file_names cross-references to boost tables
def test_search_schema_with_code_cross_ref():
    state = _build_state()
    results = search_schema(
        state, "status update",
        code_file_names=["src/order.repository.py", "src/refund.handler.py"]
    )
    names = [r.table_schema.name for r in results]
    # Both orders and refunds should appear because of code cross-ref
    assert "orders" in names
    assert "refunds" in names


# TEST: search_schema respects top_k limit
def test_search_schema_top_k():
    state = _build_state()
    results = search_schema(state, "id uuid", top_k=1)
    assert len(results) <= 1


# TEST: search_schema returns empty for query with no matches
def test_search_schema_no_match():
    state = _build_state()
    results = search_schema(state, "quantum physics simulation")
    assert results == []


# ---------------------------------------------------------------------------
# compact_schema_to_yaml
# ---------------------------------------------------------------------------


# TEST: compact_schema_to_yaml produces schema: header
def test_compact_schema_yaml_header():
    state = _build_state()
    yaml = compact_schema_to_yaml(state, ["orders"])
    assert yaml.startswith("schema:")


# TEST: compact_schema_to_yaml includes table columns
def test_compact_schema_yaml_columns():
    state = _build_state()
    yaml = compact_schema_to_yaml(state, ["orders"])
    assert "id:" in yaml or "id: UUID" in yaml
    assert "status:" in yaml or "status: TEXT" in yaml


# TEST: compact_schema_to_yaml includes enum values for referenced enums
def test_compact_schema_yaml_enums():
    state = _build_state()
    # Give orders.status a data_type that references the named enum
    state.tables["orders"].columns["status"].data_type = "order_status"
    yaml = compact_schema_to_yaml(state, ["orders"])
    assert "order_status" in yaml
    assert "pending" in yaml


# TEST: compact_schema_to_yaml includes FK references
def test_compact_schema_yaml_fk():
    state = _build_state()
    yaml = compact_schema_to_yaml(state, ["orders"])
    assert "users(id)" in yaml


# TEST: compact_schema_to_yaml includes primary key
def test_compact_schema_yaml_pk():
    state = _build_state()
    yaml = compact_schema_to_yaml(state, ["orders"])
    assert "pk:" in yaml


# TEST: compact_schema_to_yaml returns empty schema for unknown tables
def test_compact_schema_yaml_unknown_table():
    state = _build_state()
    yaml = compact_schema_to_yaml(state, ["nonexistent_table"])
    assert "schema:" in yaml
    assert "nonexistent_table" not in yaml
