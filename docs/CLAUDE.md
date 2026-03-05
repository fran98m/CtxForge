# Context Engineering Framework

## What This Is

Tooling for a context engineering pipeline that enables a solo developer on a single RTX 3090 to ship production software with AI as a collaborative partner.

Three components:
1. **AST Fetcher** — deterministic Python script that compresses codebases into token-efficient context (function signatures, test headers, no bodies/boilerplate)
2. **Doc Curator** — Jina Reader (HTML→Markdown) + local LLM pass to scrape, categorize, and structure library documentation into SQLite. Includes self-healing trigger: when code fails due to stale docs, the relevant doc entry is flagged, re-scraped, and the coding agent retries
3. **PR Generator** — auto-generates git commits and PR summaries from a spec, capturing architectural rationale (the "why") without requiring human discipline

## Architecture

The framework has two phases:
- **Phase 0 (free, web UI):** Human debates architecture with Claude + Gemini. Produces micro-manifest, doc URLs, spike conclusions. No code runs locally in this phase.
- **Phase 1 (local, RTX 3090):** Doc Curator ingests library docs → Human writes spec from Phase 0 → AST Fetcher assembles context → Coding agent (Qwen 3.5) writes code + tests → Auto-commit + PR summary

## Hard Constraints

- Must run on a single RTX 3090 (24GB VRAM)
- Target inference model: Qwen 3.5 27B or 35B-A3B at Q4 quantization
- No unnecessary dependencies — stdlib where possible
- SQLite for doc storage (no external DB)
- No raw HTML processing — Jina Reader handles HTML stripping
- AST fetcher is pure Python, zero model calls, fully deterministic
- Tests are mandatory for all components
- Every test must have a one-line header comment explaining what it validates

## Stack

- Python 3.11+
- SQLite3 (stdlib)
- ast module (stdlib) for AST fetcher
- Jina Reader API (free tier) for HTML→Markdown
- requests (HTTP calls to Jina)
- No frameworks — this is CLI tooling, not a web app

## Project Structure

```
context-framework/
├── CLAUDE.md              # This file — project manifest
├── architecture.md         # Template: per-project micro-manifest
├── src/
│   ├── ast_fetcher/       # Deterministic codebase compressor
│   │   ├── __init__.py
│   │   ├── fetcher.py     # Core AST stripping + context assembly
│   │   └── search.py      # Test header / PR summary keyword matching
│   ├── doc_curator/       # Documentation ingestion + self-healing
│   │   ├── __init__.py
│   │   ├── scraper.py     # Jina Reader integration
│   │   ├── curator.py     # LLM-powered categorization + extraction
│   │   ├── db.py          # SQLite schema + operations
│   │   └── healer.py      # Self-healing trigger logic
│   └── pr_generator/      # Auto-commit + PR summary
│       ├── __init__.py
│       └── generator.py   # Spec → commit message + PR summary
├── tests/                 # All tests — the behavioral contracts
├── spikes/                # Phase 0 spike code (quarantined from fetchers)
└── docs/                  # Framework documentation
```

## Code Style

- Type hints on all function signatures
- Docstrings on all public functions
- No classes unless genuinely needed — prefer functions and modules
- Error handling: fail loudly with clear messages, no silent swallowing
- Print to stderr for status, stdout for output (unix philosophy)

## Development Workflow

1. Build AST fetcher first (zero dependencies, immediately testable)
2. Build doc curator second (needs Jina API, SQLite)
3. Build PR generator third (needs git)
4. Integration: wire components together into CLI
