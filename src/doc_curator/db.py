"""
Documentation database: SQLite schema and operations.

Two isolated schemas to prevent context poisoning:
- library_docs: scraped external documentation (Jina Reader → LLM curator → SQLite)
- code_context: AST-extracted codebase snapshots (fetcher output, cached)

The coding agent queries these separately and the context assembler
combines them with explicit provenance. Never mix library docs with
code context in the same table — that's how you get hallucinated APIs
attributed to your own codebase.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


DB_DEFAULT_PATH = Path.home() / ".context-framework" / "docs.db"

SCHEMA = """
-- External library documentation (scraped from the web)
CREATE TABLE IF NOT EXISTS library_docs (
    url TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    framework TEXT NOT NULL,
    content_markdown TEXT NOT NULL,
    extracted_signatures TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'valid',
    last_verified_at TEXT NOT NULL,
    last_used_at TEXT,
    scraped_at TEXT NOT NULL,
    next_urls TEXT
);

CREATE INDEX IF NOT EXISTS idx_libdocs_category ON library_docs(category);
CREATE INDEX IF NOT EXISTS idx_libdocs_framework ON library_docs(framework);
CREATE INDEX IF NOT EXISTS idx_libdocs_status ON library_docs(status);

-- Codebase context snapshots (AST fetcher output, cached per project)
CREATE TABLE IF NOT EXISTS code_context (
    project_path TEXT NOT NULL,
    file_path TEXT NOT NULL,
    signatures TEXT NOT NULL,
    query TEXT,
    snapshot_at TEXT NOT NULL,
    PRIMARY KEY (project_path, file_path)
);

CREATE INDEX IF NOT EXISTS idx_codectx_project ON code_context(project_path);
"""


def get_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open or create the documentation database."""
    path = db_path or DB_DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Run schema migrations. Idempotent — safe to call on every open."""
    # Migration 1: Drop legacy 'docs' table from pre-separation schema.
    # Data was already migrated manually or is stale — the old table just
    # wastes space and confuses direct SQL queries.
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "docs" in tables and "library_docs" in tables:
        conn.execute("DROP TABLE docs")
        conn.commit()


def upsert_doc(
    conn: sqlite3.Connection,
    url: str,
    category: str,
    framework: str,
    content_markdown: str,
    extracted_signatures: list[dict],
    next_urls: Optional[list[str]] = None,
) -> None:
    """Insert or update a documentation entry."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO library_docs (url, category, framework, content_markdown,
                         extracted_signatures, status, last_verified_at,
                         scraped_at, next_urls)
        VALUES (?, ?, ?, ?, ?, 'valid', ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            category = excluded.category,
            framework = excluded.framework,
            content_markdown = excluded.content_markdown,
            extracted_signatures = excluded.extracted_signatures,
            status = 'valid',
            last_verified_at = excluded.last_verified_at,
            scraped_at = excluded.scraped_at,
            next_urls = excluded.next_urls
        """,
        (
            url,
            category,
            framework,
            content_markdown,
            json.dumps(extracted_signatures),
            now,
            now,
            json.dumps(next_urls or []),
        ),
    )
    conn.commit()


def mark_stale(conn: sqlite3.Connection, url: str) -> None:
    """Flag a documentation entry as stale (self-healing trigger)."""
    conn.execute("UPDATE library_docs SET status = 'stale' WHERE url = ?", (url,))
    conn.commit()


def mark_used(conn: sqlite3.Connection, url: str) -> None:
    """Update last_used_at timestamp. Only heal docs you're actively using."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE library_docs SET last_used_at = ? WHERE url = ?", (now, url))
    conn.commit()


def get_signatures_by_framework(
    conn: sqlite3.Connection, framework: str
) -> list[dict]:
    """Pull extracted signatures for a framework. Used by the AST fetcher."""
    rows = conn.execute(
        "SELECT url, extracted_signatures FROM library_docs WHERE framework = ? AND status = 'valid'",
        (framework,),
    ).fetchall()
    results = []
    for row in rows:
        mark_used(conn, row["url"])
        sigs = json.loads(row["extracted_signatures"])
        results.append({"url": row["url"], "signatures": sigs})
    return results


def get_signatures_by_category(
    conn: sqlite3.Connection, category: str
) -> list[dict]:
    """Pull extracted signatures for a category (e.g., 'Auth', 'Database')."""
    rows = conn.execute(
        "SELECT url, extracted_signatures FROM library_docs WHERE category = ? AND status = 'valid'",
        (category,),
    ).fetchall()
    results = []
    for row in rows:
        mark_used(conn, row["url"])
        sigs = json.loads(row["extracted_signatures"])
        results.append({"url": row["url"], "signatures": sigs})
    return results


def get_stale_docs(conn: sqlite3.Connection) -> list[dict]:
    """Get all stale documentation entries for re-scraping."""
    rows = conn.execute(
        "SELECT url, framework, category FROM library_docs WHERE status = 'stale'"
    ).fetchall()
    return [dict(row) for row in rows]


def get_pending_urls(conn: sqlite3.Connection) -> list[str]:
    """Get next_urls that haven't been scraped yet (recursive discovery)."""
    rows = conn.execute("SELECT next_urls FROM library_docs WHERE next_urls != '[]'").fetchall()
    scraped = set(
        r["url"] for r in conn.execute("SELECT url FROM library_docs").fetchall()
    )
    pending = []
    for row in rows:
        for url in json.loads(row["next_urls"]):
            if url not in scraped and url not in pending:
                pending.append(url)
    return pending


def search_docs(conn: sqlite3.Connection, query: str) -> list[dict]:
    """
    Tokenized keyword search across documentation content and signatures.

    Splits the query into individual terms, matches any term (OR logic),
    and ranks results by the number of distinct terms found.
    """
    # Tokenize: lowercase, split on non-alpha, drop short/stop words
    raw_terms = _tokenize_search_query(query)
    if not raw_terms:
        return []

    # Fetch all valid docs and score them against query terms
    rows = conn.execute(
        "SELECT url, category, framework, content_markdown, extracted_signatures FROM library_docs WHERE status = 'valid'"
    ).fetchall()

    scored = []
    for row in rows:
        searchable = (row["content_markdown"] + " " + row["extracted_signatures"]).lower()
        hits = sum(1 for term in raw_terms if term in searchable)
        if hits > 0:
            scored.append((hits, row))

    # Sort by number of matching terms (descending)
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for _hits, row in scored:
        mark_used(conn, row["url"])
        results.append({
            "url": row["url"],
            "category": row["category"],
            "framework": row["framework"],
            "extracted_signatures": row["extracted_signatures"],
        })
    return results


def _tokenize_search_query(query: str) -> list[str]:
    """Split a search query into lowercase terms, dropping noise."""
    import re
    stop = {"the", "a", "an", "is", "in", "of", "to", "and", "or", "for",
            "how", "could", "can", "what", "does", "do", "it", "be", "with"}
    words = re.findall(r'[a-z0-9]+', query.lower())
    # Also split underscored/camelCase compound terms
    expanded = []
    for w in words:
        parts = w.split('_')
        for part in parts:
            # Split camelCase
            sub = re.findall(r'[a-z]+|[0-9]+', part.lower())
            expanded.extend(sub)
    return [w for w in expanded if len(w) >= 2 and w not in stop]


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get database statistics, separated by source type."""
    # Library documentation stats
    lib_total = conn.execute("SELECT COUNT(*) as c FROM library_docs").fetchone()["c"]
    lib_valid = conn.execute(
        "SELECT COUNT(*) as c FROM library_docs WHERE status = 'valid'"
    ).fetchone()["c"]
    lib_stale = conn.execute(
        "SELECT COUNT(*) as c FROM library_docs WHERE status = 'stale'"
    ).fetchone()["c"]
    frameworks = conn.execute(
        "SELECT DISTINCT framework FROM library_docs"
    ).fetchall()
    categories = conn.execute(
        "SELECT DISTINCT category FROM library_docs"
    ).fetchall()

    # Code context stats
    code_total = conn.execute("SELECT COUNT(*) as c FROM code_context").fetchone()["c"]
    code_projects = conn.execute(
        "SELECT DISTINCT project_path FROM code_context"
    ).fetchall()

    return {
        "total": lib_total,
        "valid": lib_valid,
        "stale": lib_stale,
        "frameworks": [r["framework"] for r in frameworks],
        "categories": [r["category"] for r in categories],
        "code_context_files": code_total,
        "code_context_projects": [r["project_path"] for r in code_projects],
    }


# ── Code context operations (separate from library docs) ──────────────


def upsert_code_context(
    conn: sqlite3.Connection,
    project_path: str,
    file_path: str,
    signatures: str,
    query: Optional[str] = None,
) -> None:
    """Store or update a code context snapshot for a file in a project."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO code_context (project_path, file_path, signatures, query, snapshot_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(project_path, file_path) DO UPDATE SET
            signatures = excluded.signatures,
            query = excluded.query,
            snapshot_at = excluded.snapshot_at
        """,
        (project_path, file_path, signatures, query, now),
    )
    conn.commit()


def get_code_context(
    conn: sqlite3.Connection, project_path: str
) -> list[dict]:
    """Get all cached code context for a project, ordered by file path."""
    rows = conn.execute(
        "SELECT file_path, signatures, query, snapshot_at FROM code_context WHERE project_path = ? ORDER BY file_path",
        (project_path,),
    ).fetchall()
    return [dict(row) for row in rows]


def clear_code_context(
    conn: sqlite3.Connection, project_path: str
) -> int:
    """Remove all code context for a project (e.g. before a fresh snapshot). Returns rows deleted."""
    cursor = conn.execute(
        "DELETE FROM code_context WHERE project_path = ?", (project_path,)
    )
    conn.commit()
    return cursor.rowcount
