"""Tests for the self-healing trigger — the diagnostic that eliminates stale docs as a variable."""

import pytest
from src.doc_curator.db import get_db, upsert_doc, get_stale_docs
from src.doc_curator.healer import analyze_error, trigger_heal


@pytest.fixture
def db(tmp_path):
    conn = get_db(tmp_path / "test_heal.db")
    upsert_doc(
        conn,
        url="https://supabase.com/docs/auth",
        category="Auth",
        framework="Supabase",
        content_markdown="# Auth\nsignInWithPassword...",
        extracted_signatures=[{"name": "signInWithPassword", "params": "email, password"}],
    )
    yield conn
    conn.close()


# TEST: AttributeError with matching doc entry returns Tier 1 with correct URL
def test_tier1_attribute_error(db):
    error = AttributeError("module 'supabase.auth' has no attribute 'signInWithPassword'")
    result = analyze_error(error, "", db)
    assert result["tier"] == 1
    assert result["match"] == "https://supabase.com/docs/auth"
    assert result["should_rescrape"] is True


# TEST: Unrelated error with no doc match returns Tier 2
def test_tier2_unrelated_error(db):
    error = TimeoutError("Connection timed out after 30 seconds")
    result = analyze_error(error, "", db)
    assert result["tier"] == 2
    assert result["match"] is None
    assert result["should_rescrape"] is False


# TEST: trigger_heal flags matched doc as stale in database
def test_trigger_heal_flags_stale(db):
    error = AttributeError("has no attribute 'signInWithPassword'")
    trigger_heal(error, "", db)
    stale = get_stale_docs(db)
    assert len(stale) == 1
    assert stale[0]["url"] == "https://supabase.com/docs/auth"


# TEST: Tier 2 errors do not flag any docs as stale
def test_tier2_no_stale_flag(db):
    error = RuntimeError("Something completely unrelated broke")
    trigger_heal(error, "", db)
    stale = get_stale_docs(db)
    assert len(stale) == 0
