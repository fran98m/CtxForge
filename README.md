# CtxForge

A context engineering toolkit for solo developers shipping production software with AI on local hardware (single RTX 3090).

Built around one insight: **the context window is a workspace, not a conversation log.** Every inference call should construct optimal context from scratch.

## The Problem

Local LLMs (Qwen 3.5 27B/35B) are capable enough to write good code — *if* you give them the right context. Most AI coding tools fail not because the model is dumb, but because the context is bad: bloated with irrelevant code, stale documentation, and no architectural guidance.

## The Solution

Three components that assemble optimal context for every inference call:

### 1. AST Fetcher
Deterministic Python script (zero model calls) that compresses your codebase into token-efficient context. Strips function bodies, mock setups, and boilerplate — keeps only signatures, interfaces, and test contracts.

**Compression ratio:** 500-1,500 token test file → 20 token function signature.

```bash
python -m src.ast_fetcher.fetcher ./your-project
```

### 2. Doc Curator
Scrapes library documentation via Jina Reader (HTML → clean Markdown), then uses a local LLM to categorize and extract function signatures into SQLite. Includes a self-healing trigger: when code fails due to stale docs, the relevant entry is flagged and re-scraped automatically.

### 3. PR Generator
Auto-generates git commits and PR summaries from your spec, capturing architectural rationale without requiring discipline. The PR history becomes your project's architectural record — no separate documentation needed.

## Architecture

| Phase | What | Where | Model |
|-------|------|-------|-------|
| Phase 0 | Architecture debate, spike validation | Claude.ai + Gemini | Free frontier models |
| Phase 1 | Doc ingestion → coding → tests → PR | Local RTX 3090 | Qwen 3.5 |

**Key principle:** Use frontier models (free) for hard reasoning. Use local models for execution. Use deterministic scripts for everything else.

## Quick Start

```bash
git clone https://github.com/[you]/context-framework.git
cd context-framework
pip install -e ".[dev]"
pytest
```

## Status

Early development. The AST fetcher is functional. Doc curator and PR generator are scaffolded.

## License

MIT
