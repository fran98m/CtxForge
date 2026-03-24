"""
Terminal UI: the orchestrator that wires the entire framework together.

No curses, no blessed, no rich — just clean stdin/stdout interaction.
This is CLI tooling for a solo developer, not a dashboard.

Commands:
    full <code> [migrations] <query> [flags]  — Code + schema YAML bundle
    dump <code> [migrations]                  — Full dump (no search)
    map  <code>                               — Domain entity map
    tokens <file> [--budget N]               — Token analysis of a file
    context <path> [--query "..."]           — Extract codebase context (plain text)
    scrape <url> [url2 ...]                  — Scrape documentation via Jina
    curate <url>                             — Scrape + LLM categorize + store
    heal                                     — Check and re-scrape stale docs
    search <query>                           — Search stored documentation
    stats                                    — Show database statistics
    health                                   — Check llama-server status
    pipeline <url> [url2 ...]               — Full pipeline: scrape → curate → store
    help                                     — Show commands
    quit                                     — Exit

Flags for full/dump:
    --topk N          Max code files to include (default 5)
    --topk-tables N   Max schema tables (default 10)
    --tables t1,t2    Explicit table list
    --budget N        Token budget for report
    --all             Full dump (code + schema)
    --all-schema      Full schema, search code
    --all-code        Full code, search schema

Unix philosophy: status to stderr, output to stdout.
"""

import json
import sys
import shlex
import traceback
from pathlib import Path
from typing import Optional


def _print_err(msg: str) -> None:
    """Print to stderr for status messages."""
    print(msg, file=sys.stderr)


def _print_out(msg: str) -> None:
    """Print to stdout for data output."""
    print(msg)


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _ensure_results_dir() -> Path:
    """Create and return the results/ directory relative to cwd."""
    results = Path("results")
    results.mkdir(exist_ok=True)
    return results


def _build_output_path(prefix: str, ext: str, query: str = "") -> Path:
    """
    Build a timestamped output path under results/.

    Example: results/context_bundle_order_cancel_20260324_103045.yml
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = ""
    if query:
        # Derive slug from query: lowercase, replace spaces with _, max 24 chars
        import re
        slug = "_" + re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:24]
    return _ensure_results_dir() / f"{prefix}{slug}_{ts}.{ext}"


def _parse_flags(args: list[str]) -> tuple[list[str], dict]:
    """
    Split a flat args list into positional args and flag values.

    Supported flags:
        --topk N, --topk-tables N, --tables t1,t2,
        --budget N, --all, --all-schema, --all-code
    """
    positional: list[str] = []
    flags: dict = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--topk" and i + 1 < len(args):
            flags["top_k"] = int(args[i + 1]); i += 2
        elif a == "--topk-tables" and i + 1 < len(args):
            flags["top_k_tables"] = int(args[i + 1]); i += 2
        elif a == "--tables" and i + 1 < len(args):
            flags["explicit_tables"] = [t.strip() for t in args[i + 1].split(",")]; i += 2
        elif a == "--budget" and i + 1 < len(args):
            flags["budget"] = int(args[i + 1]); i += 2
        elif a == "--all":
            flags["all_code"] = True; flags["all_schema"] = True; i += 1
        elif a == "--all-schema":
            flags["all_schema"] = True; i += 1
        elif a == "--all-code":
            flags["all_code"] = True; i += 1
        elif a.startswith("--"):
            _print_err(f"Unknown flag: {a}")
            i += 1
        else:
            positional.append(a); i += 1
    return positional, flags


# ---------------------------------------------------------------------------
# New compact commands (full / dump / map / tokens)
# ---------------------------------------------------------------------------

def cmd_full(args: list[str], config: dict) -> None:
    """
    Produce a YAML context bundle: code signatures + optional schema.

    Usage:
        full <code_path> [migrations_path] <query> [flags]
        full <code_path> [migrations_path] --all

    The second positional argument is treated as a migrations directory if it
    is an existing directory; otherwise it is the start of the query string.
    """
    if not args:
        _print_err(
            "Usage: full <code_path> [migrations_path] <query> [flags]\n"
            "       full <code_path> --all"
        )
        return

    positional, flags = _parse_flags(args)

    code_path = Path(positional[0]).expanduser().resolve() if positional else None
    if code_path is None or not code_path.exists():
        _print_err(f"Error: code path '{positional[0] if positional else ''}' not found")
        return

    migrations_dir = None
    query_parts_start = 1
    if len(positional) >= 2:
        maybe_migrations = Path(positional[1]).expanduser().resolve()
        if maybe_migrations.is_dir():
            migrations_dir = maybe_migrations
            query_parts_start = 2

    query = " ".join(positional[query_parts_start:]).strip()

    from src.compactor import compact_full, CompactOptions

    opts = CompactOptions(
        code_path=code_path,
        query=query,
        top_k=flags.get("top_k", 5),
        all_code=flags.get("all_code", False),
        migrations_dir=migrations_dir,
        all_schema=flags.get("all_schema", False),
        top_k_tables=flags.get("top_k_tables", 10),
        explicit_tables=flags.get("explicit_tables"),
        budget=flags.get("budget"),
    )

    _print_err("Building context bundle...")
    result = compact_full(opts)

    out_path = _build_output_path("context_bundle", "yml", query)
    out_path.write_text(result.content, encoding="utf-8")
    _print_err(f"Written: {out_path}")

    if result.token_report:
        r = result.token_report
        _print_err(f"Tokens: {r.total:,}")
        if r.budget:
            _print_err(f"Budget: {r.budget['utilization_pct']}% of {r.budget['limit']:,}")

    _print_out(result.content)


def cmd_dump(args: list[str], config: dict) -> None:
    """
    Full dump — code + schema with no search filter.

    Usage: dump <code_path> [migrations_path]
    """
    if not args:
        _print_err("Usage: dump <code_path> [migrations_path]")
        return

    code_path = Path(args[0]).expanduser().resolve()
    if not code_path.exists():
        _print_err(f"Error: {code_path} not found")
        return

    migrations_dir = None
    if len(args) >= 2:
        maybe = Path(args[1]).expanduser().resolve()
        if maybe.is_dir():
            migrations_dir = maybe

    from src.compactor import compact_full, CompactOptions

    opts = CompactOptions(
        code_path=code_path,
        query="",
        all_code=True,
        migrations_dir=migrations_dir,
        all_schema=True,
    )

    _print_err("Dumping full context...")
    result = compact_full(opts)

    out_path = _build_output_path("context_bundle_dump", "yml")
    out_path.write_text(result.content, encoding="utf-8")
    _print_err(f"Written: {out_path}")

    if result.token_report:
        _print_err(f"Tokens: {result.token_report.total:,}")

    _print_out(result.content)


def cmd_map(args: list[str], config: dict) -> None:
    """
    Domain entity map — list all classes and standalone functions per file.

    Usage: map <code_path>
    """
    if not args:
        _print_err("Usage: map <code_path>")
        return

    import ast as _ast

    code_path = Path(args[0]).expanduser().resolve()
    if not code_path.exists():
        _print_err(f"Error: {code_path} not found")
        return

    _SKIP = {"__pycache__", ".git", ".venv", "venv", "node_modules"}
    lines: list[str] = ["# Domain Entity Map", f"# Source: {code_path}", ""]

    for py_file in sorted(code_path.rglob("*.py")):
        if any(part in _SKIP for part in py_file.parts):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = _ast.parse(source)
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue

        entities: list[str] = []
        for node in _ast.iter_child_nodes(tree):
            if isinstance(node, _ast.ClassDef):
                methods = [
                    item.name
                    for item in node.body
                    if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                ]
                method_str = f"  [{', '.join(methods[:6])}{'...' if len(methods) > 6 else ''}]" if methods else ""
                entities.append(f"  class {node.name}{method_str}")
            elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                entities.append(f"  def {node.name}()")

        if entities:
            rel = str(py_file.relative_to(code_path))
            lines.append(f"{rel}:")
            lines.extend(entities)
            lines.append("")

    content = "\n".join(lines)
    out_path = _build_output_path("domain_map", "txt")
    out_path.write_text(content, encoding="utf-8")
    _print_err(f"Written: {out_path}")
    _print_out(content)


def cmd_tokens(args: list[str], config: dict) -> None:
    """
    Analyse token usage of a context bundle file.

    Usage: tokens <file> [--budget N]
    """
    if not args:
        _print_err("Usage: tokens <file> [--budget N]")
        return

    positional, flags = _parse_flags(args)
    if not positional:
        _print_err("Usage: tokens <file> [--budget N]")
        return

    file_path = Path(positional[0]).expanduser().resolve()
    if not file_path.exists():
        _print_err(f"Error: {file_path} not found")
        return

    content = file_path.read_text(encoding="utf-8")

    from src.tokens import analyze_tokens, granular_breakdown, format_token_report

    budget = flags.get("budget")
    report = analyze_tokens(content, budget_limit=budget)
    granular = granular_breakdown(content)
    _print_out(format_token_report(report, granular))


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_health(args: list[str], config: dict) -> None:
    """Check llama-server status."""
    from doc_curator.llama_client import health_check

    base_url = config.get("llama_url", "http://localhost:8080")
    result = health_check(base_url)

    if result["status"] == "ok":
        _print_out(f"  Server: OK")
        _print_out(f"  Model:  {result.get('model', 'unknown')}")
        _print_out(f"  URL:    {base_url}")
        # Show context window size if detectable
        from doc_curator.llama_client import get_server_ctx_size
        ctx = get_server_ctx_size(base_url)
        if ctx:
            _print_out(f"  Context: {ctx} tokens")
    else:
        _print_out(f"  Server: DOWN")
        _print_out(f"  Error:  {result['message']}")


def cmd_stats(args: list[str], config: dict) -> None:
    """Show database statistics."""
    from doc_curator.db import get_db, get_stats

    db_path = config.get("db_path")
    conn = get_db(Path(db_path) if db_path else None)
    try:
        stats = get_stats(conn)
        _print_out("Library docs:")
        _print_out(f"  Documents:  {stats['total']} total, {stats['valid']} valid, {stats['stale']} stale")
        _print_out(f"  Frameworks: {', '.join(stats['frameworks']) or 'none'}")
        _print_out(f"  Categories: {', '.join(stats['categories']) or 'none'}")
        _print_out("Code context:")
        _print_out(f"  Cached files: {stats['code_context_files']}")
        _print_out(f"  Projects:     {', '.join(stats['code_context_projects']) or 'none'}")
    finally:
        conn.close()


def cmd_context(args: list[str], config: dict) -> None:
    """Extract codebase context with optional query filter."""
    if not args:
        _print_err("Usage: context <path> [--query \"your query\"]")
        return

    target = Path(args[0]).expanduser().resolve()
    query = None
    if "--query" in args:
        idx = args.index("--query")
        if idx + 1 < len(args):
            query = args[idx + 1]

    if not target.exists():
        _print_err(f"Error: {target} not found")
        return

    from ast_fetcher.fetcher import (
        extract_from_file,
        extract_from_directory,
        extract_relevant,
        format_context,
    )

    if target.is_file():
        _print_out(extract_from_file(target))
    elif target.is_dir():
        if query:
            _print_err(f"Searching for: {query}")
            _print_out(extract_relevant(target, query))
        else:
            sigs = extract_from_directory(target)
            _print_out(format_context(sigs))


def cmd_scrape(args: list[str], config: dict) -> None:
    """Scrape documentation URLs via Jina Reader."""
    if not args:
        _print_err("Usage: scrape <url> [url2 ...]")
        return

    from doc_curator.scraper import scrape_url

    for url in args:
        _print_err(f"Scraping: {url}")
        markdown = scrape_url(url)
        if markdown:
            _print_out(f"# --- {url} ---")
            _print_out(markdown[:2000])  # Preview
            _print_out(f"\n# [{len(markdown)} chars total]")
        else:
            _print_err(f"  FAILED: {url}")


def cmd_curate(args: list[str], config: dict) -> None:
    """Scrape a URL, send to LLM for categorization, and store in DB."""
    if not args:
        _print_err("Usage: curate <url>")
        return

    url = args[0]
    base_url = config.get("llama_url", "http://localhost:8080")

    from doc_curator.scraper import scrape_url
    from doc_curator.llama_client import curate_document
    from doc_curator.curator import parse_curator_response
    from doc_curator.db import get_db, upsert_doc

    # Step 1: Scrape
    _print_err(f"[1/3] Scraping: {url}")
    markdown = scrape_url(url)
    if not markdown:
        _print_err("  Scrape failed. Aborting.")
        return
    _print_err(f"  Got {len(markdown)} chars of markdown")

    # Step 2: LLM categorization
    _print_err(f"[2/3] Sending to LLM for categorization...")
    ctx_size = config.get("ctx_size", 0)  # 0 = auto-detect from server
    raw_response = curate_document(markdown, base_url=base_url, ctx_size=ctx_size)
    if not raw_response:
        _print_err("  LLM call failed. Is llama-server running? Try: health")
        return

    parsed = parse_curator_response(raw_response)
    if not parsed:
        _print_err("  LLM response couldn't be parsed as JSON.")
        _print_err(f"  Raw response (first 500 chars): {raw_response[:500]}")
        return

    # Step 3: Store in DB
    _print_err(f"[3/3] Storing in database...")
    db_path = config.get("db_path")
    conn = get_db(Path(db_path) if db_path else None)
    try:
        upsert_doc(
            conn,
            url=url,
            category=parsed["category"],
            framework=parsed["framework"],
            content_markdown=markdown,
            extracted_signatures=parsed.get("signatures", []),
            next_urls=parsed.get("next_urls_to_scrape"),
        )
        _print_out(f"  Stored: {url}")
        _print_out(f"  Category:  {parsed['category']}")
        _print_out(f"  Framework: {parsed['framework']}")
        _print_out(f"  Signatures: {len(parsed.get('signatures', []))}")
        next_urls = parsed.get("next_urls_to_scrape", [])
        if next_urls:
            _print_out(f"  Discovered URLs: {len(next_urls)}")
    finally:
        conn.close()


def cmd_pipeline(args: list[str], config: dict) -> None:
    """Full pipeline: scrape multiple URLs → curate each → store all."""
    if not args:
        _print_err("Usage: pipeline <url> [url2 ...]")
        return

    succeeded = 0
    failed = 0
    for url in args:
        _print_err(f"\n{'='*60}")
        _print_err(f"Pipeline: {url}")
        _print_err(f"{'='*60}")
        try:
            cmd_curate([url], config)
            succeeded += 1
        except Exception as e:
            _print_err(f"  Pipeline failed for {url}: {e}")
            failed += 1

    _print_err(f"\nPipeline complete: {succeeded} succeeded, {failed} failed")


def cmd_search(args: list[str], config: dict) -> None:
    """Search stored documentation."""
    if not args:
        _print_err("Usage: search <query>")
        return

    query = " ".join(args)
    from doc_curator.db import get_db, search_docs

    db_path = config.get("db_path")
    conn = get_db(Path(db_path) if db_path else None)
    try:
        results = search_docs(conn, query)
        if not results:
            _print_out("No matches found.")
            return

        _print_out(f"Found {len(results)} match(es):\n")
        for doc in results:
            _print_out(f"  [{doc['category']}] {doc['framework']} — {doc['url']}")
            sigs = json.loads(doc.get("extracted_signatures", "[]"))
            for sig in sigs[:3]:
                _print_out(f"    {sig.get('name', '?')}({sig.get('params', '')})")
            if len(sigs) > 3:
                _print_out(f"    ... and {len(sigs) - 3} more signatures")
    finally:
        conn.close()


def cmd_heal(args: list[str], config: dict) -> None:
    """Check for stale docs and re-scrape them."""
    from doc_curator.db import get_db, get_stale_docs, upsert_doc
    from doc_curator.scraper import scrape_url
    from doc_curator.llama_client import curate_document
    from doc_curator.curator import parse_curator_response

    db_path = config.get("db_path")
    base_url = config.get("llama_url", "http://localhost:8080")
    conn = get_db(Path(db_path) if db_path else None)

    try:
        stale = get_stale_docs(conn)
        if not stale:
            _print_out("No stale documents. All docs are current.")
            return

        _print_err(f"Found {len(stale)} stale doc(s). Re-scraping...")

        ctx_size = config.get("ctx_size", 0)  # 0 = auto-detect from server
        healed = 0
        failed = 0
        for doc in stale:
            url = doc["url"]
            _print_err(f"\n  Healing: {url}")

            markdown = scrape_url(url)
            if not markdown:
                _print_err(f"    Scrape failed for {url}")
                failed += 1
                continue

            raw_response = curate_document(markdown, base_url=base_url, ctx_size=ctx_size)
            if not raw_response:
                _print_err(f"    LLM categorization failed for {url}")
                failed += 1
                continue

            parsed = parse_curator_response(raw_response)
            if not parsed:
                _print_err(f"    Parse failed for {url}")
                failed += 1
                continue

            upsert_doc(
                conn,
                url=url,
                category=parsed["category"],
                framework=parsed["framework"],
                content_markdown=markdown,
                extracted_signatures=parsed.get("signatures", []),
                next_urls=parsed.get("next_urls_to_scrape"),
            )
            _print_out(f"  Healed: {url} → {parsed['framework']} / {parsed['category']}")
            healed += 1

        _print_err(f"\nHealing complete: {healed} healed, {failed} failed")
    finally:
        conn.close()


def cmd_help(args: list[str], config: dict) -> None:
    """Show available commands."""
    _print_out("""
Context Framework — Terminal Interface

Context bundle commands (YAML output, token-efficient):
  full <code> [migrations] <query> [flags]   Code + schema YAML bundle
  dump <code> [migrations]                   Full dump, no search filter
  map  <code>                                Domain entity map per file
  tokens <file> [--budget N]                Token analysis of any file

Flags for full/dump:
  --topk N          Max code files (default 5)
  --topk-tables N   Max schema tables (default 10)
  --tables t1,t2    Explicit table list
  --budget N        Token budget for utilisation report
  --all             Full dump (code + schema)
  --all-schema      Full schema, search code
  --all-code        Full code, search schema

Doc curator commands:
  context <path> [--query "..."]   Extract codebase context (plain text)
  scrape <url> [url2 ...]         Scrape docs via Jina Reader
  curate <url>                    Scrape + LLM categorize + store in DB
  pipeline <url> [url2 ...]       Full pipeline for multiple URLs
  heal                            Re-scrape stale documentation
  search <query>                  Search stored documentation
  stats                           Database statistics
  health                          Check llama-server status
  clear                           Clear the screen
  help                            This message
  quit                            Exit
""".strip())


def cmd_clear(args: list[str], config: dict) -> None:
    """Clear the terminal screen."""
    import os
    os.system('clear' if os.name != 'nt' else 'cls')


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

COMMANDS = {
    "full": cmd_full,
    "dump": cmd_dump,
    "map": cmd_map,
    "tokens": cmd_tokens,
    "context": cmd_context,
    "scrape": cmd_scrape,
    "curate": cmd_curate,
    "pipeline": cmd_pipeline,
    "search": cmd_search,
    "heal": cmd_heal,
    "stats": cmd_stats,
    "health": cmd_health,
    "clear": cmd_clear,
    "help": cmd_help,
}


def dispatch(line: str, config: dict) -> bool:
    """
    Parse and dispatch a single command line.

    Returns False if the user wants to quit, True otherwise.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return True

    if line in ("quit", "exit", "q"):
        return False

    try:
        parts = shlex.split(line)
    except ValueError as e:
        _print_err(f"Parse error: {e}")
        return True

    cmd_name = parts[0].lower()
    cmd_args = parts[1:]

    # Handle bare URLs — suggest the right command
    if cmd_name.startswith(("http://", "https://")):
        _print_err(f"Looks like a URL. Did you mean: curate {cmd_name}")
        return True

    handler = COMMANDS.get(cmd_name)
    if not handler:
        _print_err(f"Unknown command: {cmd_name}. Type 'help' for commands.")
        return True

    try:
        handler(cmd_args, config)
    except Exception as e:
        _print_err(f"Error in {cmd_name}: {e}")
        if "--debug" in sys.argv:
            traceback.print_exc(file=sys.stderr)

    return True


def run_interactive(config: Optional[dict] = None) -> None:
    """
    Run the interactive TUI loop.

    Reads commands from stdin one at a time. Supports piped input
    (non-interactive mode) and terminal input (with prompt).
    """
    config = config or {}

    # Add src/ to path for imports
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    is_tty = sys.stdin.isatty()

    if is_tty:
        _print_err("Context Framework v0.1")
        _print_err("Type 'help' for commands, 'quit' to exit.\n")

    while True:
        try:
            if is_tty:
                line = input("ctx> ")
            else:
                line = input()
        except (EOFError, KeyboardInterrupt):
            if is_tty:
                _print_err("\nGoodbye.")
            break

        if not dispatch(line, config):
            if is_tty:
                _print_err("Goodbye.")
            break


def run_once(command: str, config: Optional[dict] = None) -> None:
    """
    Run a single command non-interactively.

    Useful for scripting: python tui.py context src/ --query "search"
    """
    config = config or {}

    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    dispatch(command, config)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Context Framework — Terminal Interface",
        usage="python tui.py [command args...] | python tui.py (interactive)",
    )
    parser.add_argument("command", nargs="*", help="Command to run (omit for interactive mode)")
    parser.add_argument("--llama-url", default="http://localhost:8080", help="llama-server URL")
    parser.add_argument("--db-path", default=None, help="SQLite database path")
    parser.add_argument("--ctx-size", type=int, default=8192,
                        help="Model context window size in tokens (0 = auto-detect from server)")
    parser.add_argument("--debug", action="store_true", help="Show full tracebacks")

    args = parser.parse_args()

    config = {
        "llama_url": args.llama_url,
        "db_path": args.db_path,
        "ctx_size": args.ctx_size,
    }

    if args.command:
        # Non-interactive: run the command from argv
        cmd_line = shlex.join(args.command)
        run_once(cmd_line, config)
    else:
        # Interactive mode
        run_interactive(config)
