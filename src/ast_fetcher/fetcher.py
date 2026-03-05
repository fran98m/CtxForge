"""
AST Fetcher: deterministic codebase compressor.

Strips function bodies, mock setups, and boilerplate from Python files,
leaving only module signatures, function signatures, class interfaces,
and test headers. Produces token-efficient context for the coding agent.

This is NOT a model call. It is a deterministic Python script using the
stdlib ast module. No hallucination possible.

Typical compression: 500-1500 token test file → 20 token function signature.
"""

import ast
import sys
from pathlib import Path
from typing import Optional


def extract_signatures(source: str, filepath: str = "<unknown>") -> str:
    """
    Extract function/class signatures and test headers from Python source.

    Returns a compressed representation:
    - Module-level docstring (if present)
    - Import statements
    - Class names with method signatures (no bodies)
    - Function signatures with return type annotations (no bodies)
    - Test function signatures with their header comments

    Bodies, mock setups, fixtures, and boilerplate are stripped.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"# PARSE ERROR in {filepath}: {e}"

    lines = source.splitlines()
    output_parts = []

    # Module docstring
    docstring = ast.get_docstring(tree)
    if docstring:
        first_line = docstring.strip().split("\n")[0]
        output_parts.append(f'"""{first_line}"""')

    # Imports
    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            names = ", ".join(a.name for a in node.names)
            module = node.module or ""
            imports.append(f"from {module} import {names}")
    if imports:
        output_parts.append("\n".join(imports))

    # Classes and functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            output_parts.append(_extract_class(node, lines))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            output_parts.append(_extract_function(node, lines))

    return "\n\n".join(output_parts)


def _extract_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]
) -> str:
    """Extract a function signature with its header comment, decorators, and docstring."""
    parts = []

    # Header comment (line immediately above the first decorator or def)
    header_comment = _get_header_comment(node, lines)
    if header_comment:
        parts.append(header_comment)

    # Decorators (e.g. @app.get("/users"), @requires_auth)
    for decorator in node.decorator_list:
        parts.append(f"@{_unparse_node(decorator)}")

    # Function signature
    sig = _build_signature(node)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    parts.append(f"{prefix} {sig}:")

    # First line of docstring only
    docstring = ast.get_docstring(node)
    if docstring:
        first_line = docstring.strip().split("\n")[0]
        parts.append(f'    """{first_line}"""')

    parts.append("    ...")

    return "\n".join(parts)


def _extract_class(node: ast.ClassDef, lines: list[str]) -> str:
    """Extract a class definition with decorators, variables, and method signatures."""
    parts = []

    # Class-level decorators (e.g. @dataclass, @app.route)
    class_header: list[str] = []
    for decorator in node.decorator_list:
        class_header.append(f"@{_unparse_node(decorator)}")

    # Class declaration
    bases = ", ".join(_unparse_node(b) for b in node.bases)
    if bases:
        class_header.append(f"class {node.name}({bases}):")
    else:
        class_header.append(f"class {node.name}:")
    parts.append("\n".join(class_header))

    # Class docstring
    docstring = ast.get_docstring(node)
    if docstring:
        first_line = docstring.strip().split("\n")[0]
        parts.append(f'    """{first_line}"""')

    # Class body: annotated vars, plain assignments, and methods (in source order)
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            # Annotated assignment: name: Type  or  name: Type = value
            ann = _unparse_node(item.annotation)
            target = item.target.id
            if item.value is not None:
                parts.append(f"    {target}: {ann} = {_unparse_node(item.value)}")
            else:
                parts.append(f"    {target}: {ann}")
        elif isinstance(item, ast.Assign):
            # Plain assignment: __tablename__ = "users", etc.
            for target in item.targets:
                if isinstance(target, ast.Name):
                    parts.append(f"    {target.id} = {_unparse_node(item.value)}")
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_parts = []

            header_comment = _get_header_comment(item, lines)
            if header_comment:
                method_parts.append(f"    {header_comment}")

            # Method decorators (e.g. @property, @router.get("/path"))
            for decorator in item.decorator_list:
                method_parts.append(f"    @{_unparse_node(decorator)}")

            sig = _build_signature(item)
            prefix = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
            method_parts.append(f"    {prefix} {sig}:")

            method_docstring = ast.get_docstring(item)
            if method_docstring:
                first_line = method_docstring.strip().split("\n")[0]
                method_parts.append(f'        """{first_line}"""')

            method_parts.append("        ...")
            parts.append("\n".join(method_parts))

    return "\n\n".join(parts)


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a function signature string with type annotations."""
    args = []
    all_args = node.args

    # Regular arguments
    num_defaults = len(all_args.defaults)
    num_args = len(all_args.args)
    for i, arg in enumerate(all_args.args):
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {_unparse_node(arg.annotation)}"

        # Check if this arg has a default
        default_index = i - (num_args - num_defaults)
        if default_index >= 0:
            default = all_args.defaults[default_index]
            arg_str += f" = {_unparse_node(default)}"

        args.append(arg_str)

    # *args
    if all_args.vararg:
        vararg_str = f"*{all_args.vararg.arg}"
        if all_args.vararg.annotation:
            vararg_str += f": {_unparse_node(all_args.vararg.annotation)}"
        args.append(vararg_str)

    # keyword-only args
    for i, arg in enumerate(all_args.kwonlyargs):
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {_unparse_node(arg.annotation)}"
        if i < len(all_args.kw_defaults) and all_args.kw_defaults[i] is not None:
            arg_str += f" = {_unparse_node(all_args.kw_defaults[i])}"
        args.append(arg_str)

    # **kwargs
    if all_args.kwarg:
        kwarg_str = f"**{all_args.kwarg.arg}"
        if all_args.kwarg.annotation:
            kwarg_str += f": {_unparse_node(all_args.kwarg.annotation)}"
        args.append(kwarg_str)

    sig = f"{node.name}({', '.join(args)})"

    # Return annotation
    if node.returns:
        sig += f" -> {_unparse_node(node.returns)}"

    return sig


def _get_header_comment(
    node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]
) -> Optional[str]:
    """Get the comment line immediately preceding the first decorator or def keyword."""
    # Anchor to the first decorator's line if present; decorators appear before `def`
    # and node.lineno may point to either depending on Python version.
    anchor = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    line_idx = anchor - 2  # 1-indexed → 0-indexed, then one line above
    if 0 <= line_idx < len(lines):
        line = lines[line_idx].strip()
        if line.startswith("#"):
            return line
    return None


def _unparse_node(node: ast.AST) -> str:
    """Convert an AST node back to source. Uses ast.unparse on 3.9+."""
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def extract_from_file(filepath: Path) -> str:
    """Extract signatures from a single Python file."""
    source = filepath.read_text(encoding="utf-8")
    return extract_signatures(source, str(filepath))


def extract_from_directory(
    directory: Path,
    include_tests: bool = True,
    exclude_patterns: Optional[list[str]] = None,
) -> dict[str, str]:
    """
    Extract signatures from all Python files in a directory.

    Returns a dict mapping relative filepath to compressed signatures.
    """
    exclude = set(exclude_patterns or [])
    exclude.update({"__pycache__", ".git", "node_modules", ".venv", "venv", "spikes"})

    results = {}
    for py_file in sorted(directory.rglob("*.py")):
        # Skip excluded directories
        if any(part in exclude for part in py_file.parts):
            continue

        # Skip test files if not requested
        if not include_tests and (
            py_file.name.startswith("test_") or py_file.name.endswith("_test.py")
        ):
            continue

        rel_path = str(py_file.relative_to(directory))
        signatures = extract_from_file(py_file)
        if signatures.strip():
            results[rel_path] = signatures

    return results


def format_context(signatures: dict[str, str], max_tokens_approx: int = 4000) -> str:
    """
    Format extracted signatures into a context string for the coding agent.

    Estimates ~4 chars per token and truncates if needed,
    prioritizing test files and shorter entries.
    """
    char_budget = max_tokens_approx * 4

    parts = []
    total_chars = 0

    # Tests first (they're the behavioral contracts)
    test_entries = {k: v for k, v in signatures.items() if "test" in k.lower()}
    other_entries = {k: v for k, v in signatures.items() if "test" not in k.lower()}

    for filepath, sigs in {**test_entries, **other_entries}.items():
        entry = f"# --- {filepath} ---\n{sigs}"
        entry_chars = len(entry)

        if total_chars + entry_chars > char_budget:
            parts.append(f"\n# [TRUNCATED — {len(signatures) - len(parts)} files omitted]")
            break

        parts.append(entry)
        total_chars += entry_chars

    return "\n\n".join(parts)


def extract_relevant(
    directory: Path,
    query: str,
    max_tokens_approx: int = 4000,
    top_k: int = 20,
    include_tests: bool = True,
    exclude_patterns: Optional[list[str]] = None,
) -> str:
    """
    Search first, compress second. The targeted context extraction pipeline.

    Instead of compressing every file and truncating, this:
    1. Scores all files for relevance to the query (keyword matching)
    2. Extracts signatures only from the top-k relevant files
    3. Formats within the token budget — with much less truncation

    This is the main entry point for query-driven context assembly.
    """
    try:
        from .search import search_files, tokenize_query
    except ImportError:
        from search import search_files, tokenize_query

    # Phase 1: Search — find relevant files
    ranked = search_files(
        directory,
        query,
        top_k=top_k,
        include_tests=include_tests,
        exclude_patterns=exclude_patterns,
    )

    if not ranked:
        return f"# No files relevant to: {query}"

    # Phase 2: Compress — extract signatures only from relevant files
    signatures = {}
    for rel_path, score in ranked:
        full_path = directory / rel_path
        try:
            sigs = extract_from_file(full_path)
            if sigs.strip():
                signatures[rel_path] = sigs
        except (UnicodeDecodeError, PermissionError, FileNotFoundError):
            continue

    # Phase 3: Format — assemble within token budget
    terms = tokenize_query(query)
    header = f"# Query: {query}\n# Terms: {terms}\n# Relevant files: {len(signatures)}/{len(ranked)} scored\n"
    context = format_context(signatures, max_tokens_approx=max_tokens_approx)

    return header + "\n" + context


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python fetcher.py <path> [--no-tests] [--query 'your query']",
            file=sys.stderr,
        )
        sys.exit(1)

    target = Path(sys.argv[1])
    include_tests = "--no-tests" not in sys.argv

    # Extract --query value if present
    query = None
    if "--query" in sys.argv:
        idx = sys.argv.index("--query")
        if idx + 1 < len(sys.argv):
            query = sys.argv[idx + 1]

    if target.is_file():
        print(extract_from_file(target))
    elif target.is_dir():
        if query:
            # Search first, compress second
            print(extract_relevant(target, query, include_tests=include_tests))
        else:
            # Original behavior: compress everything
            sigs = extract_from_directory(target, include_tests=include_tests)
            print(format_context(sigs))
    else:
        print(f"Error: {target} not found", file=sys.stderr)
        sys.exit(1)
