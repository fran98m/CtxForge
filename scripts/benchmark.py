#!/usr/bin/env python3
"""
scripts/benchmark.py

Compares raw token cost vs compact YAML output.
Run: python scripts/benchmark.py <code_path> [migrations_path]

This is NOT a feature — it's a curiosity tool to see how much
compression the context builder actually achieves.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ast_fetcher.fetcher import extract_from_directory, extract_signatures


# ─── Raw Measurement ─────────────────────────────────────────────────────────


class RawStats:
    label: str
    file_count: int
    total_chars: int
    total_lines: int
    total_tokens: int
    breakdown: list[dict]

    def __init__(
        self,
        label: str,
        file_count: int,
        total_chars: int,
        total_lines: int,
        total_tokens: int,
        breakdown: list[dict],
    ):
        self.label = label
        self.file_count = file_count
        self.total_chars = total_chars
        self.total_lines = total_lines
        self.total_tokens = total_tokens
        self.breakdown = breakdown


class CompactStats:
    label: str
    total_chars: int
    total_lines: int
    total_tokens: int

    def __init__(
        self, label: str, total_chars: int, total_lines: int, total_tokens: int
    ):
        self.label = label
        self.total_chars = total_chars
        self.total_lines = total_lines
        self.total_tokens = total_tokens


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation: ~4 chars per token for code.
    This is a heuristic, not exact.
    """
    return max(1, len(text) // 4)


def measure_raw_code(code_path: str) -> RawStats:
    """
    Measures the raw token cost of reading every Python file as-is.
    This is what you'd pay if you just cat'd every file into the context.
    """
    code_dir = Path(code_path)
    breakdown = []
    total_chars = 0
    total_lines = 0
    total_tokens = 0
    file_count = 0

    for py_file in sorted(code_dir.rglob("*.py")):
        # Skip excluded directories
        if any(part in {"__pycache__", ".git", "node_modules", ".venv", "venv", "spikes"} for part in py_file.parts):
            continue
        # Skip test files
        if ".test." in py_file.name or ".spec." in py_file.name:
            continue
        if py_file.name.startswith("test_") or py_file.name.endswith("_test.py"):
            continue

        rel_path = str(py_file.relative_to(code_dir))
        text = py_file.read_text(encoding="utf-8")
        chars = len(text)
        lines = text.count("\n") + 1
        tokens = estimate_tokens(text)

        breakdown.append({"file": rel_path, "chars": chars, "tokens": tokens})
        total_chars += chars
        total_lines += lines
        total_tokens += tokens
        file_count += 1

    breakdown.sort(key=lambda x: x["tokens"], reverse=True)

    return RawStats(
        label="Raw Python Files",
        file_count=file_count,
        total_chars=total_chars,
        total_lines=total_lines,
        total_tokens=total_tokens,
        breakdown=breakdown,
    )


def measure_raw_migrations(migrations_dir: str) -> RawStats:
    """
    Measures the raw token cost of reading every SQL migration as-is.
    """
    migrations_path = Path(migrations_dir)
    files = sorted([f for f in migrations_path.iterdir() if f.suffix == ".sql"])

    breakdown = []
    total_chars = 0
    total_lines = 0
    total_tokens = 0

    for file in files:
        text = file.read_text(encoding="utf-8")
        chars = len(text)
        lines = text.count("\n") + 1
        tokens = estimate_tokens(text)

        breakdown.append({"file": file.name, "chars": chars, "tokens": tokens})
        total_chars += chars
        total_lines += lines
        total_tokens += tokens

    breakdown.sort(key=lambda x: x["tokens"], reverse=True)

    return RawStats(
        label="Raw SQL Migrations",
        file_count=len(files),
        total_chars=total_chars,
        total_lines=total_lines,
        total_tokens=total_tokens,
        breakdown=breakdown,
    )


# ─── Compact Measurement ─────────────────────────────────────────────────────


def measure_compact_code(code_path: str) -> CompactStats:
    """
    Measures the compact token cost using signature extraction.
    """
    signatures = extract_from_directory(
        Path(code_path),
        include_tests=False,
        exclude_patterns=["__pycache__", ".git", "node_modules", ".venv", "venv", "spikes"],
    )

    # Format as YAML-like output
    yaml_parts = []
    for filepath, sig in sorted(signatures.items()):
        yaml_parts.append(f"# --- {filepath} ---")
        yaml_parts.append(sig)

    yaml = "\n\n".join(yaml_parts)
    return CompactStats(
        label="Compact Code Signatures",
        total_chars=len(yaml),
        total_lines=yaml.count("\n") + 1,
        total_tokens=estimate_tokens(yaml),
    )


def measure_compact_schema(migrations_dir: str) -> CompactStats:
    """
    Measures the compact token cost for SQL schema (just table definitions).
    For simplicity, we extract CREATE TABLE statements only.
    """
    migrations_path = Path(migrations_dir)
    files = sorted([f for f in migrations_path.iterdir() if f.suffix == ".sql"])

    yaml_parts = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        # Extract CREATE TABLE statements (simplified)
        lines = text.split("\n")
        schema_lines = []
        in_create = False
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("CREATE TABLE"):
                in_create = True
            if in_create:
                schema_lines.append(line)
                if stripped.endswith(";"):
                    in_create = False

        if schema_lines:
            yaml_parts.append(f"-- {file.name}")
            yaml_parts.extend(schema_lines)

    yaml = "\n".join(yaml_parts)
    return CompactStats(
        label="Compact Schema (CREATE TABLE only)",
        total_chars=len(yaml),
        total_lines=yaml.count("\n") + 1,
        total_tokens=estimate_tokens(yaml),
    )


# ─── Display ─────────────────────────────────────────────────────────────────


def bar(ratio: float, width: int = 40) -> str:
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def format_num(n: int) -> str:
    return f"{n:,}"


def print_comparison(raw: RawStats, compact: CompactStats) -> None:
    ratio = compact.total_tokens / raw.total_tokens if raw.total_tokens > 0 else 0
    savings = 1 - ratio
    compression_x = raw.total_tokens / compact.total_tokens if compact.total_tokens > 0 else float("inf")

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print(f"│  {raw.label:<58}│")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│                                                             │")
    print("│  RAW (just cat every file into context):                    │")
    print(f"│    Files:    {raw.file_count:>8}                                    │")
    print(f"│    Chars:    {format_num(raw.total_chars):>8}                                    │")
    print(f"│    Lines:    {format_num(raw.total_lines):>8}                                    │")
    print(f"│    Tokens:   {format_num(raw.total_tokens):>8}                                    │")
    print("│                                                             │")
    print("│  COMPACT (signatures/schema only):                          │")
    print(f"│    Chars:    {format_num(compact.total_chars):>8}                                    │")
    print(f"│    Lines:    {format_num(compact.total_lines):>8}                                    │")
    print(f"│    Tokens:   {format_num(compact.total_tokens):>8}                                    │")
    print("│                                                             │")
    print("│  COMPRESSION:                                               │")
    print(f"│    Ratio:    {compression_x:>5.1f}x smaller                                │")
    print(f"│    Savings:  {savings * 100:>5.1f}%                                        │")
    print("│                                                             │")
    print(f"│    Raw:     {bar(1, 45)} │")
    print(f"│    Compact: {bar(ratio, 45)} │")
    print("│                                                             │")
    print("└─────────────────────────────────────────────────────────────┘")


def print_top_files(raw: RawStats, top_n: int = 10) -> None:
    print(f"\n  Top {top_n} most expensive files (raw):")
    print("  " + "─" * 56)
    for item in raw.breakdown[:top_n]:
        name = item["file"]
        if len(name) > 38:
            name = "..." + name[-35:]
        print(f"  {name:<40} {format_num(item['tokens']):>7} tk")


def print_summary(
    raw_code: RawStats,
    compact_code: CompactStats,
    raw_migrations: Optional[RawStats] = None,
    compact_schema: Optional[CompactStats] = None,
) -> None:
    total_raw = raw_code.total_tokens + (raw_migrations.total_tokens if raw_migrations else 0)
    total_compact = compact_code.total_tokens + (compact_schema.total_tokens if compact_schema else 0)
    overall_ratio = total_raw / total_compact if total_compact > 0 else float("inf")
    overall_savings = (1 - total_compact / total_raw) * 100 if total_raw > 0 else 0

    print("\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("┃                    OVERALL SUMMARY                          ┃")
    print("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
    print("┃                                                             ┃")
    print(f"┃  Raw total:       {format_num(total_raw):>10} tokens                      ┃")
    print(f"┃  Compact total:   {format_num(total_compact):>10} tokens                      ┃")
    print(f"┃  Compression:     {overall_ratio:>10.1f}x                            ┃")
    print(f"┃  Token savings:   {overall_savings:>9.1f}%                            ┃")
    print("┃                                                             ┃")

    if raw_migrations and compact_schema:
        code_ratio = raw_code.total_tokens / compact_code.total_tokens if compact_code.total_tokens > 0 else float("inf")
        schema_ratio = raw_migrations.total_tokens / compact_schema.total_tokens if compact_schema.total_tokens > 0 else float("inf")
        print("┃  By category:                                               ┃")
        print(f"┃    Code:   {format_num(raw_code.total_tokens):>8} → {format_num(compact_code.total_tokens):>8}  ({code_ratio:.1f}x)              ┃")
        print(f"┃    Schema: {format_num(raw_migrations.total_tokens):>8} → {format_num(compact_schema.total_tokens):>8}  ({schema_ratio:.1f}x)              ┃")
        print("┃                                                             ┃")

    # Context window fitting guide
    windows = [
        {"name": "Claude Sonnet (200k)", "tokens": 200000},
        {"name": "GPT-4o (128k)", "tokens": 128000},
        {"name": "Practical limit (~8k)", "tokens": 8000},
    ]

    print("┃  Would it fit?                                              ┃")
    for w in windows:
        raw_fits = "✅" if total_raw <= w["tokens"] else "❌"
        compact_fits = "✅" if total_compact <= w["tokens"] else "❌"
        print(f"┃    {w['name']:<24} Raw: {raw_fits}  Compact: {compact_fits}        ┃")

    print("┃                                                             ┃")
    print("┃  Even with compact signatures, you likely need search.      ┃")
    print("┃  Use: --query 'your query' --top-k N to stay within budget. ┃")
    print("┃                                                             ┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/benchmark.py <code_path> [migrations_path]")
        sys.exit(1)

    code_path = sys.argv[1]
    migrations_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print("🔬 Running benchmark...\n")
    print(f"Code path:       {code_path}")
    if migrations_dir:
        print(f"Migrations path: {migrations_dir}")

    # Measure raw
    print("\n⏱  Measuring raw file sizes...")
    raw_code = measure_raw_code(code_path)

    raw_migrations: Optional[RawStats] = None
    if migrations_dir:
        raw_migrations = measure_raw_migrations(migrations_dir)

    # Measure compact
    print("⏱  Running compactor...")
    comp_code = measure_compact_code(code_path)

    comp_schema: Optional[CompactStats] = None
    if migrations_dir:
        comp_schema = measure_compact_schema(migrations_dir)

    # Display results
    print_comparison(raw_code, comp_code)
    print_top_files(raw_code)

    if raw_migrations and comp_schema:
        print_comparison(raw_migrations, comp_schema)

    print_summary(raw_code, comp_code, raw_migrations, comp_schema)


if __name__ == "__main__":
    main()
