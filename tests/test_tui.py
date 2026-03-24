"""Tests for the TUI — the orchestrator that wires the framework together."""

import json
import shlex
import pytest
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from src.tui import (
    dispatch,
    cmd_health,
    cmd_stats,
    cmd_context,
    cmd_scrape,
    cmd_curate,
    cmd_search,
    cmd_heal,
    cmd_help,
    cmd_clear,
    cmd_pipeline,
    COMMANDS,
)


# ---------------------------------------------------------------------------
# dispatch() tests
# ---------------------------------------------------------------------------


# TEST: Dispatch returns False for quit commands to exit the loop
def test_dispatch_quit():
    assert dispatch("quit", {}) is False
    assert dispatch("exit", {}) is False
    assert dispatch("q", {}) is False


# TEST: Dispatch returns True for empty lines and comments
def test_dispatch_empty_and_comments():
    assert dispatch("", {}) is True
    assert dispatch("   ", {}) is True
    assert dispatch("# this is a comment", {}) is True


# TEST: Unknown commands print error but don't crash
def test_dispatch_unknown_command(capsys):
    result = dispatch("foobar", {})
    assert result is True
    captured = capsys.readouterr()
    assert "Unknown command" in captured.err


# TEST: Bare URL input suggests curate command instead of unknown error
def test_dispatch_bare_url_suggests_curate(capsys):
    result = dispatch("https://example.com/docs", {})
    assert result is True
    captured = capsys.readouterr()
    assert "curate" in captured.err
    assert "https://example.com/docs" in captured.err


# TEST: All expected commands (old and new) are registered in COMMANDS dict
def test_all_commands_registered():
    expected = {
        "context", "scrape", "curate", "pipeline", "search", "heal",
        "stats", "health", "clear", "help",
        # New compact commands added in v0.2
        "full", "dump", "map", "tokens",
    }
    assert expected == set(COMMANDS.keys())


# ---------------------------------------------------------------------------
# cmd_health() tests
# ---------------------------------------------------------------------------


# TEST: Health command shows server status when server is up
def test_cmd_health_ok(capsys):
    mock_result = {"status": "ok", "model": "qwen-3.5", "message": "running"}
    mock_mod = MagicMock()
    mock_mod.health_check.return_value = mock_result
    with patch.dict("sys.modules", {"doc_curator": MagicMock(), "doc_curator.llama_client": mock_mod}):
        cmd_health([], {"llama_url": "http://localhost:8080"})

    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert "qwen-3.5" in captured.out


# TEST: Health command shows DOWN when server unreachable
def test_cmd_health_down(capsys):
    mock_result = {"status": "error", "model": None, "message": "Cannot connect"}
    with patch.dict("sys.modules"):
        mock_mod = MagicMock()
        mock_mod.health_check.return_value = mock_result
        with patch.dict("sys.modules", {"doc_curator.llama_client": mock_mod}):
            cmd_health([], {})

    captured = capsys.readouterr()
    assert "DOWN" in captured.out


# ---------------------------------------------------------------------------
# cmd_stats() tests
# ---------------------------------------------------------------------------


# TEST: Stats command displays database statistics
def test_cmd_stats(capsys, tmp_path):
    mock_stats = {
        "total": 5, "valid": 4, "stale": 1,
        "frameworks": ["Next.js", "Supabase"],
        "categories": ["Auth", "Database"],
        "code_context_files": 12,
        "code_context_projects": ["/home/user/my-project"],
    }
    mock_conn = MagicMock()

    with patch.dict("sys.modules"):
        mock_db = MagicMock()
        mock_db.get_db.return_value = mock_conn
        mock_db.get_stats.return_value = mock_stats
        with patch.dict("sys.modules", {"doc_curator.db": mock_db}):
            cmd_stats([], {"db_path": str(tmp_path / "test.db")})

    captured = capsys.readouterr()
    assert "5 total" in captured.out
    assert "4 valid" in captured.out
    assert "1 stale" in captured.out


# ---------------------------------------------------------------------------
# cmd_context() tests
# ---------------------------------------------------------------------------


# TEST: Context command extracts signatures from a directory
def test_cmd_context_directory(capsys, tmp_path):
    (tmp_path / "app.py").write_text('def hello(): pass\n')

    with patch.dict("sys.modules"):
        mock_fetcher = MagicMock()
        mock_fetcher.extract_from_directory.return_value = {"app.py": "def hello():\n    ..."}
        mock_fetcher.format_context.return_value = "# --- app.py ---\ndef hello():\n    ..."
        with patch.dict("sys.modules", {"ast_fetcher.fetcher": mock_fetcher}):
            cmd_context([str(tmp_path)], {})

    captured = capsys.readouterr()
    assert "app.py" in captured.out


# TEST: Context command with --query uses targeted extraction
def test_cmd_context_with_query(capsys, tmp_path):
    (tmp_path / "quiz.py").write_text('def quiz(): pass\n')

    with patch.dict("sys.modules"):
        mock_fetcher = MagicMock()
        mock_fetcher.extract_relevant.return_value = "# Query: quiz\n# --- quiz.py ---\ndef quiz():\n    ..."
        with patch.dict("sys.modules", {"ast_fetcher.fetcher": mock_fetcher}):
            cmd_context([str(tmp_path), "--query", "quiz"], {})

    captured = capsys.readouterr()
    assert "quiz" in captured.out


# TEST: Context command with nonexistent path shows error
def test_cmd_context_bad_path(capsys):
    cmd_context(["/nonexistent/path/foo"], {})
    captured = capsys.readouterr()
    assert "not found" in captured.err


# TEST: Context command with no args shows usage
def test_cmd_context_no_args(capsys):
    cmd_context([], {})
    captured = capsys.readouterr()
    assert "Usage" in captured.err


# ---------------------------------------------------------------------------
# cmd_scrape() tests
# ---------------------------------------------------------------------------


# TEST: Scrape command calls Jina Reader and outputs markdown preview
def test_cmd_scrape(capsys):
    with patch.dict("sys.modules"):
        mock_scraper = MagicMock()
        mock_scraper.scrape_url.return_value = "# FastAPI Docs\n\nSome content here..."
        with patch.dict("sys.modules", {"doc_curator.scraper": mock_scraper}):
            cmd_scrape(["https://fastapi.tiangolo.com/tutorial/"], {})

    captured = capsys.readouterr()
    assert "FastAPI" in captured.out


# TEST: Scrape command handles failed scrape gracefully
def test_cmd_scrape_failure(capsys):
    with patch.dict("sys.modules"):
        mock_scraper = MagicMock()
        mock_scraper.scrape_url.return_value = None
        with patch.dict("sys.modules", {"doc_curator.scraper": mock_scraper}):
            cmd_scrape(["https://bad-url.example.com"], {})

    captured = capsys.readouterr()
    assert "FAILED" in captured.err


# ---------------------------------------------------------------------------
# cmd_curate() tests
# ---------------------------------------------------------------------------


# TEST: Curate command runs full scrape→LLM→store pipeline
def test_cmd_curate_full(capsys, tmp_path):
    llm_response = json.dumps({
        "category": "API",
        "framework": "FastAPI",
        "signatures": [{"name": "get", "params": "path", "returns": "Response", "description": "GET route"}],
        "next_urls_to_scrape": [],
    })

    mock_scraper = MagicMock()
    mock_scraper.scrape_url.return_value = "# FastAPI\nContent..."

    mock_llama = MagicMock()
    mock_llama.curate_document.return_value = llm_response

    mock_curator = MagicMock()
    mock_curator.parse_curator_response.return_value = json.loads(llm_response)

    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.get_db.return_value = mock_conn

    with patch.dict("sys.modules", {
        "doc_curator": MagicMock(),
        "doc_curator.scraper": mock_scraper,
        "doc_curator.llama_client": mock_llama,
        "doc_curator.curator": mock_curator,
        "doc_curator.db": mock_db,
    }):
        cmd_curate(["https://fastapi.tiangolo.com"], {"db_path": str(tmp_path / "test.db")})

    captured = capsys.readouterr()
    assert "FastAPI" in captured.out
    assert "API" in captured.out
    mock_db.upsert_doc.assert_called_once()


# TEST: Curate command handles LLM failure gracefully
def test_cmd_curate_llm_failure(capsys):
    mock_scraper = MagicMock()
    mock_scraper.scrape_url.return_value = "# Some docs"

    mock_llama = MagicMock()
    mock_llama.curate_document.return_value = None

    mock_curator = MagicMock()
    mock_db = MagicMock()

    with patch.dict("sys.modules", {
        "doc_curator": MagicMock(),
        "doc_curator.scraper": mock_scraper,
        "doc_curator.llama_client": mock_llama,
        "doc_curator.curator": mock_curator,
        "doc_curator.db": mock_db,
    }):
        cmd_curate(["https://example.com"], {})

    captured = capsys.readouterr()
    assert "llama-server" in captured.err.lower() or "LLM" in captured.err


# ---------------------------------------------------------------------------
# cmd_search() tests
# ---------------------------------------------------------------------------


# TEST: Search command finds and displays matching documents
def test_cmd_search(capsys, tmp_path):
    mock_conn = MagicMock()
    mock_results = [
        {
            "category": "Auth",
            "framework": "Supabase",
            "url": "https://supabase.com/docs/auth",
            "extracted_signatures": json.dumps([
                {"name": "signIn", "params": "email, password"},
                {"name": "signOut", "params": ""},
            ]),
        }
    ]

    mock_db = MagicMock()
    mock_db.get_db.return_value = mock_conn
    mock_db.search_docs.return_value = mock_results

    with patch.dict("sys.modules", {"doc_curator.db": mock_db}):
        cmd_search(["signIn"], {"db_path": str(tmp_path / "test.db")})

    captured = capsys.readouterr()
    assert "Supabase" in captured.out
    assert "signIn" in captured.out


# TEST: Search with no results shows appropriate message
def test_cmd_search_no_results(capsys, tmp_path):
    mock_db = MagicMock()
    mock_db.get_db.return_value = MagicMock()
    mock_db.search_docs.return_value = []

    with patch.dict("sys.modules", {"doc_curator.db": mock_db}):
        cmd_search(["nonexistent"], {"db_path": str(tmp_path / "test.db")})

    captured = capsys.readouterr()
    assert "No matches" in captured.out


# ---------------------------------------------------------------------------
# cmd_heal() tests
# ---------------------------------------------------------------------------


# TEST: Heal command re-scrapes stale docs and stores updated versions
def test_cmd_heal_with_stale(capsys, tmp_path):
    stale_docs = [{"url": "https://old.com/docs", "framework": "OldLib", "category": "API"}]
    llm_response = json.dumps({
        "category": "API",
        "framework": "NewLib",
        "signatures": [],
        "next_urls_to_scrape": [],
    })

    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.get_db.return_value = mock_conn
    mock_db.get_stale_docs.return_value = stale_docs

    mock_scraper = MagicMock()
    mock_scraper.scrape_url.return_value = "# Updated docs"

    mock_llama = MagicMock()
    mock_llama.curate_document.return_value = llm_response

    mock_curator = MagicMock()
    mock_curator.parse_curator_response.return_value = json.loads(llm_response)

    with patch.dict("sys.modules", {
        "doc_curator": MagicMock(),
        "doc_curator.db": mock_db,
        "doc_curator.scraper": mock_scraper,
        "doc_curator.llama_client": mock_llama,
        "doc_curator.curator": mock_curator,
    }):
        cmd_heal([], {"db_path": str(tmp_path / "test.db")})

    captured = capsys.readouterr()
    assert "Healed" in captured.out
    mock_db.upsert_doc.assert_called_once()


# TEST: Heal command with no stale docs shows clean message
def test_cmd_heal_nothing_stale(capsys, tmp_path):
    mock_db = MagicMock()
    mock_db.get_db.return_value = MagicMock()
    mock_db.get_stale_docs.return_value = []

    with patch.dict("sys.modules", {
        "doc_curator": MagicMock(),
        "doc_curator.db": mock_db,
        "doc_curator.scraper": MagicMock(),
        "doc_curator.llama_client": MagicMock(),
        "doc_curator.curator": MagicMock(),
    }):
        cmd_heal([], {"db_path": str(tmp_path / "test.db")})

    captured = capsys.readouterr()
    assert "No stale" in captured.out or "0 stale" in captured.err


# ---------------------------------------------------------------------------
# cmd_help() tests
# ---------------------------------------------------------------------------


# TEST: Help command lists all available commands
def test_cmd_help(capsys):
    cmd_help([], {})
    captured = capsys.readouterr()
    assert "context" in captured.out
    assert "scrape" in captured.out
    assert "curate" in captured.out
    assert "heal" in captured.out
    assert "search" in captured.out
    assert "health" in captured.out
    assert "quit" in captured.out


# ---------------------------------------------------------------------------
# Integration: dispatch routes to correct handler
# ---------------------------------------------------------------------------


# TEST: Dispatch correctly routes 'help' command to cmd_help handler
def test_dispatch_routes_help(capsys):
    dispatch("help", {})
    captured = capsys.readouterr()
    assert "context" in captured.out  # help output includes command list


# TEST: Dispatch handles exceptions in commands without crashing
def test_dispatch_handles_exceptions(capsys):
    # context with a bad path will print error but not crash
    result = dispatch("context /this/path/does/not/exist/ever", {})
    assert result is True
    captured = capsys.readouterr()
    assert "not found" in captured.err or "Error" in captured.err
