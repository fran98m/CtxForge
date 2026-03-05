"""Tests for the documentation database — the verified ground truth store."""

import json
import sqlite3
import pytest
from src.doc_curator.db import (
    get_db,
    upsert_doc,
    mark_stale,
    mark_used,
    get_signatures_by_framework,
    get_stale_docs,
    get_pending_urls,
    search_docs,
    get_stats,
    upsert_code_context,
    get_code_context,
    clear_code_context,
)


@pytest.fixture
def db(tmp_path):
    """Create a fresh test database."""
    db_path = tmp_path / "test_docs.db"
    conn = get_db(db_path)
    yield conn
    conn.close()


# TEST: Inserting a doc and retrieving it returns correct signatures
def test_upsert_and_retrieve(db):
    upsert_doc(
        db,
        url="https://supabase.com/docs/auth",
        category="Auth",
        framework="Supabase",
        content_markdown="# Auth\nSign in with password...",
        extracted_signatures=[{"name": "signInWithPassword", "params": "email, password", "returns": "Session"}],
    )
    results = get_signatures_by_framework(db, "Supabase")
    assert len(results) == 1
    assert results[0]["signatures"][0]["name"] == "signInWithPassword"


# TEST: Marking a doc as stale excludes it from signature retrieval
def test_mark_stale_excludes_from_retrieval(db):
    upsert_doc(
        db,
        url="https://example.com/docs",
        category="API",
        framework="Example",
        content_markdown="# Example",
        extracted_signatures=[{"name": "doThing", "params": "", "returns": "void"}],
    )
    mark_stale(db, "https://example.com/docs")
    results = get_signatures_by_framework(db, "Example")
    assert len(results) == 0


# TEST: Re-upserting a stale doc resets status to valid
def test_reupsert_heals_stale(db):
    upsert_doc(db, "https://x.com/docs", "Auth", "X", "# X", [{"name": "a"}])
    mark_stale(db, "https://x.com/docs")
    assert len(get_stale_docs(db)) == 1

    upsert_doc(db, "https://x.com/docs", "Auth", "X", "# X updated", [{"name": "b"}])
    assert len(get_stale_docs(db)) == 0
    results = get_signatures_by_framework(db, "X")
    assert results[0]["signatures"][0]["name"] == "b"


# TEST: Pending URLs returns unscraped next_urls for recursive discovery
def test_pending_urls(db):
    upsert_doc(
        db,
        url="https://docs.example.com/intro",
        category="API",
        framework="Example",
        content_markdown="# Intro",
        extracted_signatures=[],
        next_urls=["https://docs.example.com/auth", "https://docs.example.com/db"],
    )
    pending = get_pending_urls(db)
    assert "https://docs.example.com/auth" in pending
    assert "https://docs.example.com/db" in pending


# TEST: Search finds docs by keyword in content and signatures
def test_search_docs(db):
    upsert_doc(
        db,
        url="https://redis.io/docs",
        category="Database",
        framework="Redis",
        content_markdown="# Redis\nUsed for caching and session storage",
        extracted_signatures=[{"name": "set", "params": "key, value"}],
    )
    results = search_docs(db, "caching")
    assert len(results) == 1
    assert results[0]["framework"] == "Redis"


# TEST: Search tokenizes multi-word queries and matches individual terms
def test_search_docs_multi_term(db):
    upsert_doc(db, "https://a.com", "Database", "Redis", "# Redis caching layer", [])
    upsert_doc(db, "https://b.com", "Auth", "Supabase", "# Supabase auth with sessions", [])
    # "caching session storage" should match both (caching -> Redis, session -> both)
    results = search_docs(db, "caching session storage")
    assert len(results) >= 1
    # Redis matches more terms (caching + session is not in its content, but caching is)
    # Supabase matches "session"
    urls = [r["url"] for r in results]
    assert "https://a.com" in urls


# TEST: Search with natural language query finds results via individual terms
def test_search_docs_natural_language(db):
    upsert_doc(
        db,
        url="https://llamaindex.ai/docs",
        category="API",
        framework="LlamaIndex",
        content_markdown="# LlamaIndex\nBuild knowledge graphs and RAG pipelines",
        extracted_signatures=[{"name": "VectorStoreIndex", "params": "nodes"}],
    )
    results = search_docs(db, "how could knowledge graphs benefit the project")
    assert len(results) == 1
    assert results[0]["framework"] == "LlamaIndex"


# TEST: Search with underscore compound terms splits and matches
def test_search_docs_underscore_terms(db):
    upsert_doc(
        db,
        url="https://example.com/graphs",
        category="API",
        framework="Example",
        content_markdown="# Knowledge Graph API\nBuild knowledge graphs",
        extracted_signatures=[],
    )
    results = search_docs(db, "knowledge_graphs")
    assert len(results) == 1


# TEST: Stats returns correct counts of valid and stale docs
def test_stats(db):
    upsert_doc(db, "https://a.com", "Auth", "A", "# A", [])
    upsert_doc(db, "https://b.com", "DB", "B", "# B", [])
    mark_stale(db, "https://b.com")

    stats = get_stats(db)
    assert stats["total"] == 2
    assert stats["valid"] == 1
    assert stats["stale"] == 1
    assert stats["code_context_files"] == 0
    assert stats["code_context_projects"] == []


# ── Code context table tests ─────────────────────────────────────────


# TEST: Upserting code context stores and retrieves file signatures for a project
def test_upsert_and_get_code_context(db):
    upsert_code_context(db, "/home/user/project", "src/app.py", "def main(): ...\ndef run(): ...")
    upsert_code_context(db, "/home/user/project", "src/utils.py", "def helper(): ...")

    results = get_code_context(db, "/home/user/project")
    assert len(results) == 2
    assert results[0]["file_path"] == "src/app.py"
    assert "def main" in results[0]["signatures"]
    assert results[1]["file_path"] == "src/utils.py"


# TEST: Upserting code context with same project+file replaces the old snapshot
def test_code_context_upsert_replaces(db):
    upsert_code_context(db, "/project", "main.py", "def old(): ...")
    upsert_code_context(db, "/project", "main.py", "def new(): ...")

    results = get_code_context(db, "/project")
    assert len(results) == 1
    assert "def new" in results[0]["signatures"]


# TEST: Code context for different projects stays isolated
def test_code_context_project_isolation(db):
    upsert_code_context(db, "/project-a", "app.py", "def a(): ...")
    upsert_code_context(db, "/project-b", "app.py", "def b(): ...")

    results_a = get_code_context(db, "/project-a")
    results_b = get_code_context(db, "/project-b")
    assert len(results_a) == 1
    assert "def a" in results_a[0]["signatures"]
    assert len(results_b) == 1
    assert "def b" in results_b[0]["signatures"]


# TEST: clear_code_context removes all files for a project and returns count
def test_clear_code_context(db):
    upsert_code_context(db, "/project", "a.py", "def a(): ...")
    upsert_code_context(db, "/project", "b.py", "def b(): ...")
    upsert_code_context(db, "/other", "c.py", "def c(): ...")

    deleted = clear_code_context(db, "/project")
    assert deleted == 2
    assert get_code_context(db, "/project") == []
    assert len(get_code_context(db, "/other")) == 1


# TEST: Code context with query stores and retrieves the query field
def test_code_context_stores_query(db):
    upsert_code_context(db, "/project", "app.py", "def main(): ...", query="authentication flow")
    results = get_code_context(db, "/project")
    assert results[0]["query"] == "authentication flow"


# TEST: Stats includes code context counts alongside library doc counts
def test_stats_includes_code_context(db):
    upsert_doc(db, "https://a.com", "Auth", "A", "# A", [])
    upsert_code_context(db, "/project-x", "main.py", "def main(): ...")
    upsert_code_context(db, "/project-x", "utils.py", "def util(): ...")

    stats = get_stats(db)
    assert stats["total"] == 1
    assert stats["code_context_files"] == 2
    assert "/project-x" in stats["code_context_projects"]
