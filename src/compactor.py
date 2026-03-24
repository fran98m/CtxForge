"""
YAML compactor: converts AST-extracted signatures into a compact,
LLM-friendly YAML context bundle.

Orchestrates the full pipeline:
    1. Search relevant files (or dump all)
    2. Extract AST signatures (body-stripped, hints included)
    3. Optionally merge SQL schema section (if schema module available)
    4. Emit a structured YAML bundle with a header and token report

Output format is intentionally YAML-ish (not strict valid YAML) because
syntax like function signatures contains characters that would require
quoting. The format is optimised to be read by LLMs, not parsed by machines.

Zero model calls. Zero external dependencies.
"""

import ast
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Options / result types
# ---------------------------------------------------------------------------

@dataclass
class CompactOptions:
    code_path: Path
    query: str = ""
    top_k: int = 5
    all_code: bool = False
    migrations_dir: Optional[Path] = None
    all_schema: bool = False
    top_k_tables: int = 10
    explicit_tables: Optional[list[str]] = None
    budget: Optional[int] = None


@dataclass
class CompactResult:
    content: str
    file_names: list[str] = field(default_factory=list)
    token_report: Optional[object] = None  # TokenReport from tokens.py


# ---------------------------------------------------------------------------
# Internal: file → YAML formatter
# ---------------------------------------------------------------------------

def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


def _slug(name: str) -> str:
    """Convert an identifier name to lowercase, keeping underscores."""
    return re.sub(r"[^a-z0-9_]", "", name.lower())


def _file_to_yaml(source: str, filepath: str) -> str:
    """
    Convert a Python source file into an indented YAML-ish block.

    Produces output like::

        class MyClass(Base):
          # Docstring first line | → call1 → call2
          def method(self, x: int) -> str: ...
        def standalone(a: str) -> bool: ...

    Bodies are stripped; only signatures, decorators, annotated fields,
    docstrings (first line), and body hints are preserved.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return f"  # PARSE ERROR: {filepath}"

    lines_src = source.splitlines()
    parts: list[str] = []

    # Imports (collapsed to a single line for brevity)
    imports: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names = ", ".join(a.name for a in node.names)
                imports.append(f"{node.module}.{names}")

    if imports:
        parts.append(f"  # imports: {', '.join(imports[:6])}{'...' if len(imports) > 6 else ''}")

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            parts.append(_class_to_yaml(node, lines_src))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(_func_to_yaml(node, lines_src, indent=2))

    return "\n".join(parts)


def _func_to_yaml(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    indent: int = 4,
) -> str:
    """Emit one function as an indented YAML-ish signature line + comment."""
    from src.ast_fetcher.fetcher import (
        _build_signature,
        _get_header_comment,
        _extract_body_hint,
    )

    pad = " " * indent
    out: list[str] = []

    header = _get_header_comment(node, lines)
    if header:
        out.append(f"{pad}{header}")

    for dec in node.decorator_list:
        out.append(f"{pad}@{ast.unparse(dec)}")

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    sig = f"{prefix} {_build_signature(node)}"

    docstring = ast.get_docstring(node)
    hint = _extract_body_hint(node)

    comment_parts: list[str] = []
    if docstring:
        comment_parts.append(docstring.strip().split("\n")[0])
    if hint:
        comment_parts.append(f"→ {hint}")

    if comment_parts:
        out.append(f"{pad}# {' | '.join(comment_parts)}")

    out.append(f"{pad}{sig}: ...")
    return "\n".join(out)


def _class_to_yaml(node: ast.ClassDef, lines: list[str]) -> str:
    """Emit a class block with its members as indented YAML-ish entries."""
    from src.ast_fetcher.fetcher import _unparse_node

    out: list[str] = []
    pad2 = "  "
    pad4 = "    "

    for dec in node.decorator_list:
        out.append(f"{pad2}@{ast.unparse(dec)}")

    bases = ", ".join(_unparse_node(b) for b in node.bases)
    header = f"{pad2}class {node.name}({bases}):" if bases else f"{pad2}class {node.name}:"
    out.append(header)

    docstring = ast.get_docstring(node)
    if docstring:
        out.append(f"{pad4}# {docstring.strip().split(chr(10))[0]}")

    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            ann = _unparse_node(item.annotation)
            target = item.target.id
            if item.value is not None:
                out.append(f"{pad4}{target}: {ann} = {_unparse_node(item.value)}")
            else:
                out.append(f"{pad4}{target}: {ann}")
        elif isinstance(item, ast.Assign):
            for t in item.targets:
                if isinstance(t, ast.Name):
                    out.append(f"{pad4}{t.id} = {_unparse_node(item.value)}")
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(_func_to_yaml(item, lines, indent=4))

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Public: compact_code
# ---------------------------------------------------------------------------

def compact_code(
    code_path: Path,
    query: str,
    top_k: int = 5,
) -> CompactResult:
    """
    Search code_path for files relevant to query, extract YAML signatures.

    Returns a CompactResult with content (YAML string) and file_names (list
    of relative paths that were included — used for schema cross-referencing).
    """
    from src.ast_fetcher.search import search_files

    ranked = search_files(code_path, query, top_k=top_k)
    if not ranked:
        return CompactResult(content=f"  # No files relevant to: {query}", file_names=[])

    file_names: list[str] = []
    blocks: list[str] = []

    for rel_path, score in ranked:
        full_path = code_path / rel_path
        try:
            source = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        yaml_block = _file_to_yaml(source, rel_path)
        if yaml_block.strip():
            blocks.append(f"  {rel_path}:\n{yaml_block}")
            file_names.append(rel_path)

    content = "code:\n" + "\n\n".join(blocks) if blocks else "code: {}"
    return CompactResult(content=content, file_names=file_names)


def compact_all_code(code_path: Path) -> CompactResult:
    """
    Dump signatures for every Python file in code_path (no search, no filter).

    Skips test files, __pycache__, and non-Python files.
    """
    _SKIP = {"__pycache__", ".git", ".venv", "venv", "node_modules", "spikes"}
    file_names: list[str] = []
    blocks: list[str] = []

    for py_file in sorted(code_path.rglob("*.py")):
        if any(part in _SKIP for part in py_file.parts):
            continue
        if py_file.name.startswith("test_") or py_file.name.endswith("_test.py"):
            continue

        rel_path = str(py_file.relative_to(code_path))
        try:
            source = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        yaml_block = _file_to_yaml(source, rel_path)
        if yaml_block.strip():
            blocks.append(f"  {rel_path}:\n{yaml_block}")
            file_names.append(rel_path)

    content = "code:\n" + "\n\n".join(blocks) if blocks else "code: {}"
    return CompactResult(content=content, file_names=file_names)


# ---------------------------------------------------------------------------
# Public: compact_full
# ---------------------------------------------------------------------------

def compact_full(opts: CompactOptions) -> CompactResult:
    """
    Orchestrate code + optional schema sections into a single YAML bundle.

    Produces a header, code section, optional schema section, and a token
    report appended as a comment block at the end.
    """
    from src.tokens import analyze_tokens, granular_breakdown, format_token_report

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- Code section ---
    if opts.all_code or not opts.query:
        code_result = compact_all_code(opts.code_path)
    else:
        code_result = compact_code(opts.code_path, opts.query, top_k=opts.top_k)

    # --- Schema section (optional) ---
    schema_content = ""
    if opts.migrations_dir and opts.migrations_dir.exists():
        schema_content = _compact_schema_section(opts, code_result.file_names)

    # --- Assemble bundle ---
    header_lines = [
        "# Context Bundle",
        f"# Generated: {now}",
    ]
    if opts.query:
        header_lines.append(f"# Query: {opts.query}")
    header_lines.append(f"# Codebase: {opts.code_path}")
    if opts.migrations_dir:
        header_lines.append(f"# Migrations: {opts.migrations_dir}")
    header_lines.append("# Format: Compact YAML (optimised for LLM token efficiency)")
    header_lines.append("---")

    sections = ["\n".join(header_lines), code_result.content]
    if schema_content:
        sections.append(schema_content)

    full_content = "\n\n".join(sections)

    # --- Token report ---
    report = analyze_tokens(full_content, budget_limit=opts.budget)
    granular = granular_breakdown(full_content)
    report_str = format_token_report(report, granular)

    full_content += "\n\n# " + "\n# ".join(report_str.splitlines())

    return CompactResult(
        content=full_content,
        file_names=code_result.file_names,
        token_report=report,
    )


def _compact_schema_section(opts: CompactOptions, code_file_names: list[str]) -> str:
    """Build the schema YAML section by delegating to the schema module."""
    try:
        from src.schema.schema import extract_schema
        from src.schema.schema_search import search_schema, compact_schema_to_yaml
    except ImportError:
        return "# schema: (schema module not available)"

    state = extract_schema(str(opts.migrations_dir))

    if opts.explicit_tables:
        table_names = opts.explicit_tables
    elif opts.all_schema or not opts.query:
        table_names = list(state.tables.keys())
    else:
        results = search_schema(
            state,
            opts.query,
            top_k=opts.top_k_tables,
            code_file_names=code_file_names,
        )
        table_names = [r.table_schema.name for r in results]

    return compact_schema_to_yaml(state, table_names)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Compact a Python codebase into a YAML context bundle."
    )
    p.add_argument("code_path", type=Path, help="Root directory of the codebase")
    p.add_argument("query", nargs="?", default="", help="Natural language query")
    p.add_argument("--migrations", type=Path, default=None, dest="migrations_dir")
    p.add_argument("--topk", type=int, default=5, dest="top_k")
    p.add_argument("--topk-tables", type=int, default=10, dest="top_k_tables")
    p.add_argument("--tables", type=str, default=None)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--all", action="store_true", dest="all_code")
    p.add_argument("--all-schema", action="store_true", dest="all_schema")
    p.add_argument("--output", type=Path, default=None)

    args = p.parse_args()

    explicit = [t.strip() for t in args.tables.split(",")] if args.tables else None

    opts = CompactOptions(
        code_path=args.code_path,
        query=args.query,
        top_k=args.top_k,
        all_code=args.all_code,
        migrations_dir=args.migrations_dir,
        all_schema=args.all_schema,
        top_k_tables=args.top_k_tables,
        explicit_tables=explicit,
        budget=args.budget,
    )

    result = compact_full(opts)

    if args.output:
        args.output.write_text(result.content, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(result.content)
