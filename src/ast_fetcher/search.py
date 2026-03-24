"""
Relevance search: keyword-based file scoring for targeted context extraction.

The fetcher's compression is proven. But for real projects (50+ files),
compressing everything and truncating wastes the context window on irrelevant
code. This module solves that: search first, compress second.

Strategy: score each Python file against a query using weighted keyword
matching across multiple surfaces (filenames, identifiers, docstrings,
comments, test headers). Propagate relevance through the import graph
so dependencies of relevant files get included too.

Zero model calls. Pure Python. Fully deterministic.
"""

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Query tokenization
# ---------------------------------------------------------------------------

# Common English stop words that add noise to keyword matching
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "but", "and", "or", "if",
    "while", "about", "up", "that", "this", "these", "those", "i", "it",
    "its", "my", "we", "you", "he", "she", "they", "what", "which", "who",
    "add", "want", "feature", "implement", "build", "create", "make",
})

# Simple suffix rules for lightweight stemming (no nltk needed)
_SUFFIX_RULES = [
    ("ation", ""),   # normalization → normaliz
    ("tion", ""),    # detection → detec
    ("sion", ""),    # progression → progres
    ("ing", ""),     # scoring → scor
    ("ment", ""),    # management → manage — wait, this cuts too much
    ("ness", ""),    # darkness → dark
    ("ies", "y"),    # queries → query
    ("es", ""),      # classes → class — careful
    ("s", ""),       # tests → test
    ("ed", ""),      # processed → process
    ("er", ""),      # loader → load
    ("ly", ""),      # quickly → quick
]


def tokenize_query(query: str) -> list[str]:
    """
    Break a natural language query into searchable keyword stems.

    Splits on whitespace and common delimiters, lowercases, removes
    stop words, and applies lightweight suffix stripping.

    >>> tokenize_query("I want to add a quiz feature for scoring")
    ['quiz', 'scor']
    """
    # Split on non-alphanumeric characters
    raw_tokens = re.split(r"[^a-zA-Z0-9_]+", query.lower())
    raw_tokens = [t for t in raw_tokens if t and t not in _STOP_WORDS and len(t) > 1]

    # Also split snake_case and camelCase tokens into parts
    expanded = []
    for token in raw_tokens:
        # snake_case
        parts = token.split("_")
        if len(parts) > 1:
            expanded.extend(p for p in parts if p and len(p) > 1)
        # camelCase
        camel_parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", token).lower().split()
        if len(camel_parts) > 1:
            expanded.extend(p for p in camel_parts if p and len(p) > 1)
        expanded.append(token)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in expanded:
        if t not in seen and t not in _STOP_WORDS:
            seen.add(t)
            unique.append(t)

    # Stem each token
    stemmed = []
    for token in unique:
        stemmed.append(_stem(token))
        # Keep original too if different from stem
        if _stem(token) != token:
            stemmed.append(token)

    # Deduplicate again
    seen = set()
    result = []
    for t in stemmed:
        if t not in seen and len(t) > 1:
            seen.add(t)
            result.append(t)

    return result


def _stem(word: str) -> str:
    """Lightweight suffix stripping. Not perfect, but zero dependencies."""
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
            return word[: -len(suffix)] + replacement
    return word


# ---------------------------------------------------------------------------
# File surface extraction (what we search against)
# ---------------------------------------------------------------------------

def extract_searchable_surface(source: str, filepath: str) -> dict[str, list[str]]:
    """
    Extract all searchable text surfaces from a Python file.

    Returns a dict with these keys:
    - 'filename': terms from the file path
    - 'identifiers': function names, class names, variable names
    - 'docstrings': docstring text
    - 'comments': comment lines (including test headers)
    - 'imports': imported module/package names
    """
    surfaces: dict[str, list[str]] = {
        "filename": [],
        "identifiers": [],
        "docstrings": [],
        "comments": [],
        "imports": [],
    }

    # --- Filename terms ---
    path_stem = Path(filepath).stem  # e.g. "test_quiz_integration"
    path_parts = Path(filepath).parts  # e.g. ("tests", "test_quiz.py")
    for part in path_parts:
        # Strip .py extension and split on underscores
        clean = part.replace(".py", "")
        surfaces["filename"].extend(
            t for t in re.split(r"[^a-zA-Z0-9]+", clean) if t and len(t) > 1
        )

    # --- AST-based extraction ---
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to regex-based extraction for unparseable files
        surfaces["comments"] = re.findall(r"#\s*(.+)", source)
        return surfaces

    # Walk the entire AST
    for node in ast.walk(tree):
        # Function and class names
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            surfaces["identifiers"].append(node.name)
            # Also split the name into parts (test_quiz_scoring → [test, quiz, scoring])
            surfaces["identifiers"].extend(
                p for p in node.name.split("_") if p and len(p) > 1
            )
            # Docstring
            docstring = ast.get_docstring(node)
            if docstring:
                surfaces["docstrings"].append(docstring)

        elif isinstance(node, ast.ClassDef):
            surfaces["identifiers"].append(node.name)
            # Split CamelCase
            camel_parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", node.name).lower().split()
            surfaces["identifiers"].extend(
                p for p in camel_parts if p and len(p) > 1
            )
            docstring = ast.get_docstring(node)
            if docstring:
                surfaces["docstrings"].append(docstring)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                surfaces["imports"].append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                surfaces["imports"].append(node.module)
                # Split dotted imports: modules.data_loader → [modules, data, loader]
                surfaces["imports"].extend(
                    p for p in re.split(r"[._]+", node.module) if p and len(p) > 1
                )

    # --- Comment extraction (AST doesn't parse comments) ---
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            comment_text = stripped.lstrip("#").strip()
            if comment_text:
                surfaces["comments"].append(comment_text)

    return surfaces


# ---------------------------------------------------------------------------
# Import graph extraction
# ---------------------------------------------------------------------------

def extract_import_graph(
    directory: Path, exclude_patterns: Optional[set[str]] = None
) -> dict[str, set[str]]:
    """
    Build a simple import graph: file → set of files it imports.

    Only resolves local imports (within the directory). External packages
    are ignored since they don't help with relevance propagation.
    """
    exclude = exclude_patterns or {"__pycache__", ".git", "node_modules", ".venv", "venv", "spikes"}

    # Map module names to file paths for local resolution
    module_to_file: dict[str, str] = {}
    py_files: list[tuple[str, Path]] = []

    for py_file in sorted(directory.rglob("*.py")):
        if any(part in exclude for part in py_file.parts):
            continue
        rel = str(py_file.relative_to(directory))
        py_files.append((rel, py_file))

        # Register both dotted module path and bare filename
        # e.g. "modules/data_loader.py" → "modules.data_loader" and "data_loader"
        module_path = rel.replace("/", ".").replace(".py", "")
        if module_path.endswith(".__init__"):
            module_path = module_path[: -len(".__init__")]
        module_to_file[module_path] = rel
        # Also the bare stem
        stem = py_file.stem
        if stem != "__init__":
            module_to_file[stem] = rel

    # Now build the graph
    graph: dict[str, set[str]] = defaultdict(set)

    for rel, py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            imported_module = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_module = alias.name
            elif isinstance(node, ast.ImportFrom):
                imported_module = node.module

            if imported_module:
                # Try to resolve to a local file
                # Check full dotted path first, then progressively shorter prefixes
                parts = imported_module.split(".")
                for i in range(len(parts), 0, -1):
                    candidate = ".".join(parts[:i])
                    if candidate in module_to_file:
                        target = module_to_file[candidate]
                        if target != rel:  # No self-edges
                            graph[rel].add(target)
                        break

    return dict(graph)


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

# Weight multipliers for each surface type
_WEIGHTS = {
    "filename": 5.0,     # File named quiz.py when searching "quiz" → very strong
    "identifiers": 3.0,  # Function named test_quiz_scoring → strong
    "docstrings": 2.0,   # Docstring mentions "quiz generation" → medium
    "comments": 2.5,     # Test header "# TEST: quiz produces valid scores" → medium-high
    "imports": 1.0,      # Imports quiz module → weak direct signal
}


def score_file(
    filepath: str,
    source: str,
    query_terms: list[str],
) -> float:
    """
    Score a single file's relevance to query terms.

    Returns a float ≥ 0.0. Higher = more relevant.
    Not normalized to 0-1 because we only care about relative ordering.
    """
    if not query_terms:
        return 0.0

    surfaces = extract_searchable_surface(source, filepath)
    total_score = 0.0

    for surface_name, texts in surfaces.items():
        weight = _WEIGHTS.get(surface_name, 1.0)
        # Join all texts into one searchable blob
        blob = " ".join(texts).lower()
        # Also stem the blob tokens for fuzzy matching
        blob_tokens = set(re.split(r"[^a-zA-Z0-9_]+", blob))
        blob_stems = {_stem(t) for t in blob_tokens if len(t) > 1}

        hits = 0
        for term in query_terms:
            # Exact substring match in the blob
            if term in blob:
                hits += 1.0
            # Stem match (fuzzy)
            elif term in blob_stems:
                hits += 0.7
            # Partial match (term is a prefix of some token)
            elif any(t.startswith(term) for t in blob_tokens if len(t) > len(term)):
                hits += 0.4

        if hits > 0:
            # Normalize by number of query terms so score reflects coverage
            coverage = hits / len(query_terms)
            total_score += weight * coverage

    return total_score


def search_files(
    directory: Path,
    query: str,
    top_k: int = 20,
    include_tests: bool = True,
    exclude_patterns: Optional[list[str]] = None,
    boost_down: float = 0.2,
    boost_up: float = 0.4,
) -> list[tuple[str, float]]:
    """
    Search a directory for Python files relevant to a natural language query.

    Returns a sorted list of (relative_path, score) tuples, highest first.
    Only files with score > 0 are returned. Bidirectional import graph
    propagation boosts both dependencies AND consumers of high-scoring files.

    Args:
        directory: Root directory to search.
        query: Natural language query (e.g. "quiz scoring progression").
        top_k: Maximum number of files to return.
        include_tests: Whether to include test files.
        exclude_patterns: Directory names to skip.
        boost_down: Fraction of a file's score propagated to its imports
                    (consumer → dependency). Default 0.2.
        boost_up:   Fraction of an import's score propagated to its consumers
                    (dependency → consumer). Default 0.4 — intentionally higher
                    because a highly-relevant dependency means callers probably
                    need context too.
    """
    exclude = set(exclude_patterns or [])
    exclude.update({"__pycache__", ".git", "node_modules", ".venv", "venv", "spikes"})

    query_terms = tokenize_query(query)
    if not query_terms:
        return []

    # Phase 1: Score every file independently
    file_scores: dict[str, float] = {}
    file_sources: dict[str, str] = {}

    for py_file in sorted(directory.rglob("*.py")):
        if any(part in exclude for part in py_file.parts):
            continue
        if not include_tests and (
            py_file.name.startswith("test_") or py_file.name.endswith("_test.py")
        ):
            continue

        rel = str(py_file.relative_to(directory))
        try:
            source = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        file_sources[rel] = source
        score = score_file(rel, source, query_terms)
        if score > 0:
            file_scores[rel] = score

    # Phase 2: Bidirectional import graph propagation
    # boost_down: high-scoring file pushes score to its imports (dependencies likely relevant)
    # boost_up:   high-scoring import pushes score back to consumers (callers likely relevant)
    if (boost_down > 0 or boost_up > 0) and file_scores:
        import_graph = extract_import_graph(directory, exclude)

        # Build reverse graph: imported_file → set of files that import it
        consumers: dict[str, set[str]] = defaultdict(set)
        for importer, imported_set in import_graph.items():
            for imported in imported_set:
                consumers[imported].add(importer)

        propagated: dict[str, float] = defaultdict(float)

        # boost_down: consumer scores flow to its dependencies
        if boost_down > 0:
            for filepath, score in file_scores.items():
                for imported_file in import_graph.get(filepath, set()):
                    propagated[imported_file] += score * boost_down

        # boost_up: dependency scores flow to their consumers
        if boost_up > 0:
            for filepath, score in file_scores.items():
                for consumer_file in consumers.get(filepath, set()):
                    propagated[consumer_file] += score * boost_up

        for filepath, boost in propagated.items():
            if filepath in file_sources:  # Only boost files we actually found
                file_scores[filepath] = file_scores.get(filepath, 0.0) + boost

    # Phase 3: Sort and return top-k
    ranked = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python search.py <directory> <query> [--top-k N]",
            file=sys.stderr,
        )
        print(
            'Example: python search.py ~/project "quiz scoring progression"',
            file=sys.stderr,
        )
        sys.exit(1)

    target = Path(sys.argv[1])
    query = sys.argv[2]

    top_k = 20
    if "--top-k" in sys.argv:
        idx = sys.argv.index("--top-k")
        if idx + 1 < len(sys.argv):
            top_k = int(sys.argv[idx + 1])

    if not target.is_dir():
        print(f"Error: {target} is not a directory", file=sys.stderr)
        sys.exit(1)

    results = search_files(target, query, top_k=top_k)

    if not results:
        print("No relevant files found.", file=sys.stderr)
        sys.exit(0)

    # Print ranked results
    print(f"# Query: {query}")
    print(f"# Terms: {tokenize_query(query)}")
    print(f"# Found {len(results)} relevant files\n")

    for filepath, score in results:
        print(f"  {score:6.2f}  {filepath}")
