# Context Engineering Framework

A context engineering toolkit for solo developers shipping production software with AI on local hardware (single RTX 3090).

Built around one insight: **the context window is a workspace, not a conversation log.** Every inference call should construct optimal context from scratch.

## The Problem

Local LLMs (Qwen 3.5 27B/35B) are capable enough to write good code — *if* you give them the right context. Most AI coding tools fail not because the model is dumb, but because the context is bad: bloated with irrelevant code, stale documentation, and no architectural guidance.

## The Solution

Five components that assemble optimal context for every inference call:

| Component | What it does | Model calls |
|-----------|-------------|-------------|
| **AST Fetcher** | Compresses Python codebases into token-efficient signatures | Zero |
| **Schema Module** | Replays SQL migrations → live schema + FK-aware search | Zero |
| **Compactor** | Merges code + schema into a single YAML context bundle | Zero |
| **Token Module** | Estimates token usage and reports per-file/table budget | Zero |
| **Doc Curator** | Scrapes docs via Jina, categorises with local LLM, self-heals | Local LLM |

## Quick Start

```bash
git clone https://github.com/[you]/context-framework.git
cd context-framework
uv sync
uv run pytest
```

---

## Core workflow: `full`

The main command. Searches your codebase (and optionally your DB migrations),
extracts signatures, and writes a YAML bundle ready to paste into any LLM.

```bash
# Code only — search for relevant files
uv run python -m src.tui full ./src "order cancellation refund"

# Code + database schema
uv run python -m src.tui full ./src ./db/migrations "order cancellation refund"

# Tune how many files / tables to include
uv run python -m src.tui full ./src ./db/migrations "payment" --topk 8 --topk-tables 15

# Check token budget (useful if context window is tight)
uv run python -m src.tui full ./src ./db/migrations "auth" --budget 8000

# Full dump — no search, everything included
uv run python -m src.tui dump ./src ./db/migrations
```

Output files are written to `results/context_bundle_<slug>_<timestamp>.yml`.

### YAML output format

```yaml
# Context Bundle
# Generated: 2026-03-24T10:30:45Z
# Query: order cancellation refund
---

code:
  src/order/service.py:
    class OrderService:
      # Manages order lifecycle
      # Validates order status | → Order → validate_status → save
      async def cancel_order(self, order_id: str, reason: CancelReason) -> Order: ...
      async def process_refund(self, order_id: str) -> RefundResult: ...

  src/payment/refund.py:
    # Process a refund | → create_refund → notify
    async def process_refund(order: Order) -> RefundResult: ...

schema:
  enums:
    cancel_reason: [customer_request | duplicate | fraud | other]
    order_status: [pending | confirmed | cancelled | refunded]

  orders:
    pk: [id]
    indexes: [(user_id), (created_at)]
    cols:
      id: UUID, not null
      user_id: UUID, not null -> users(id)
      status: order_status, not null, default pending
      cancel_reason: cancel_reason
      cancelled_at: TIMESTAMP

# ╔════════════════════════════════════════════╗
# ║  Token Report                              ║
# ║  Total: 3,241 tokens (12,847 chars)        ║
# ║  Budget: 8,000 — 40% used (4,759 left)    ║
# ╠════════════════════════════════════════════╣
# ║  SECTION BREAKDOWN                         ║
# ║  code    1,823  56%  ███████░░░░░          ║
# ║  schema  1,418  44%  ██████░░░░░░          ║
# ╚════════════════════════════════════════════╝
```

---

## All CLI commands

```
full <code> [migrations] <query> [flags]   YAML bundle: code + schema (main command)
dump <code> [migrations]                   Full dump, no search filter
map  <code>                                Domain entity map (classes + functions per file)
tokens <file> [--budget N]                Token analysis of any existing file
context <path> [--query "..."]            Plain-text context (legacy, no YAML)
```

### Flags for `full` / `dump`

| Flag | Default | Description |
|------|---------|-------------|
| `--topk N` | 5 | Max code files from search |
| `--topk-tables N` | 10 | Max schema tables from search |
| `--tables t1,t2` | — | Include specific tables by name |
| `--budget N` | — | Show token utilisation against this limit |
| `--all` | — | Full dump: all code files + all tables |
| `--all-schema` | — | Full schema, search code only |
| `--all-code` | — | Full code dump, search schema only |

### Examples

```bash
# Map every class and function in a project (good for orientation)
uv run python -m src.tui map ./src

# Check how many tokens a bundle you already generated uses
uv run python -m src.tui tokens results/context_bundle_order_20260324.yml --budget 32000

# Include specific tables regardless of query
uv run python -m src.tui full ./src ./migrations "auth flow" --tables users,sessions,tokens

# Full schema dump + targeted code search (fast for projects with stable DBs)
uv run python -m src.tui full ./src ./migrations "checkout" --all-schema --topk 6
```

---

## Component reference

### AST Fetcher (`src/ast_fetcher/`)

Strips function bodies from Python files, keeping only:
- Module docstring
- Import statements
- Class declarations with annotated fields and method signatures
- Function/method signatures with type annotations
- Decorators (`@app.get`, `@dataclass`, etc.)
- Docstring first line
- **Body hints** — compact action chain extracted before stripping (`# → Order → validate → save`)
- Test header comments (`# TEST: ...`)

**Compression ratio:** 500–1,500 token file → ~20 token signature.

Search uses TF-IDF keyword matching across filenames, identifiers, docstrings, and comments, with **bidirectional import graph propagation**: high-scoring files boost their dependencies *and* their callers.

```python
from src.ast_fetcher import extract_relevant, search_files

# Search first, compress second
context = extract_relevant(Path("./src"), query="order cancellation", top_k=5)

# Or search only (returns ranked file list)
ranked = search_files(Path("./src"), "payment refund", top_k=10)
```

### Schema Module (`src/schema/`)

Replays SQL migration files chronologically to build a live `SchemaState`. Supports:
- **dbmate format** (`-- migrate:up` / `-- migrate:down`)
- **Plain `.sql` files** (no markers required)
- `CREATE TABLE`, `ALTER TABLE` (ADD/DROP/RENAME COLUMN, ALTER TYPE, SET DEFAULT, ADD CONSTRAINT)
- `CREATE TYPE ... AS ENUM` (Postgres named enums)
- `CHECK (col IN (...))` inline enum detection
- `DROP TABLE` / `DROP TYPE`

Schema search is FK-aware: relevant tables boost their FK targets and vice versa. Code file names from the AST search are cross-referenced to further boost matching tables.

```python
from src.schema import extract_schema, search_schema, compact_schema_to_yaml

state = extract_schema("./db/migrations")

# Search tables relevant to a query
results = search_schema(state, "order cancellation", top_k=5,
                        code_file_names=["order.service.py"])

# Render to YAML
yaml = compact_schema_to_yaml(state, [r.table_schema.name for r in results])
```

### Token Module (`src/tokens.py`)

Built-in heuristic token estimator — no `tiktoken` dependency. Detects code vs. prose ratio (code ≈ 3.5 chars/token, prose ≈ 4.2 chars/token) and blends accordingly. Benchmarks within ±5–8% of GPT-4/Claude tokenisers.

```python
from src.tokens import analyze_tokens, granular_breakdown, format_token_report

report = analyze_tokens(content, budget_limit=8000)
granular = granular_breakdown(content)   # per-file and per-table breakdown
print(format_token_report(report, granular))
```

### Compactor (`src/compactor.py`)

Orchestrates the full pipeline: search → AST extract → schema → YAML → token report.

```python
from src.compactor import compact_full, CompactOptions
from pathlib import Path

result = compact_full(CompactOptions(
    code_path=Path("./src"),
    query="order cancellation",
    top_k=5,
    migrations_dir=Path("./db/migrations"),
    top_k_tables=10,
    budget=8000,
))

print(result.content)          # full YAML bundle
print(result.token_report.total)  # token count
```

### Doc Curator (`src/doc_curator/`)

Interactive TUI commands for managing external documentation:

```bash
uv run python -m src.tui   # start interactive mode

ctx> curate https://docs.example.com/api   # scrape + LLM categorise + store
ctx> search "vector store index"           # search stored docs
ctx> heal                                  # re-scrape stale docs
ctx> stats                                 # show DB statistics
ctx> health                                # check llama-server
```

---

## Architecture

| Phase | What | Where | Model |
|-------|------|-------|-------|
| Phase 0 | Architecture debate, spike validation | Claude.ai + Gemini | Free frontier models |
| Phase 1 | Context bundle → coding → tests → PR | Local RTX 3090 | Qwen 3.5 |

**Key principle:** Use frontier models (free) for hard reasoning. Use local models for execution. Use deterministic scripts for everything else.

---

## Status

- **AST Fetcher** — Functional. Bidirectional import graph, body hints.
- **Schema Module** — Functional. Migration replay, FK-aware search.
- **Compactor** — Functional. YAML bundles with token reports.
- **Token Module** — Functional. No external deps.
- **Doc Curator** — Functional. Scraper, LLM categorisation, self-healing.
- **PR Generator** — Scaffolded.

239 tests, all passing.

## License

MIT
