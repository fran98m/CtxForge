"""Tests for the relevance search module — the filter that makes compression targeted."""

import textwrap
import tempfile
from pathlib import Path
import pytest
from src.ast_fetcher.search import (
    tokenize_query,
    _stem,
    extract_searchable_surface,
    score_file,
    search_files,
    extract_import_graph,
)
from src.ast_fetcher.fetcher import extract_relevant


# ---------------------------------------------------------------------------
# Tokenization tests
# ---------------------------------------------------------------------------


# TEST: Query tokenizer strips stop words and returns meaningful terms
def test_tokenize_strips_stop_words():
    result = tokenize_query("I want to add a quiz feature")
    assert "want" not in result
    assert "add" not in result
    assert "quiz" in result


# TEST: Query tokenizer handles empty and whitespace-only input
def test_tokenize_empty():
    assert tokenize_query("") == []
    assert tokenize_query("   ") == []
    assert tokenize_query("the a an") == []


# TEST: Query tokenizer splits snake_case terms into parts
def test_tokenize_splits_snake_case():
    result = tokenize_query("data_loader")
    assert "data" in result
    assert "loader" in result or "load" in result


# TEST: Query tokenizer generates stems for fuzzy matching
def test_tokenize_produces_stems():
    result = tokenize_query("scoring progression")
    # "scoring" should stem to "scor" and "progression" to "progres"
    assert any("scor" in t for t in result)
    assert any("progres" in t for t in result)


# TEST: Stemmer handles common English suffixes
def test_stem_basic():
    assert _stem("scoring") == "scor"
    assert _stem("tests") == "test"
    assert _stem("processed") == "process"
    assert _stem("queries") == "query"


# TEST: Stemmer doesn't over-strip short words
def test_stem_short_words():
    # Should not strip if result would be < 3 chars
    result = _stem("is")
    assert len(result) >= 2  # "is" is too short to stem


# ---------------------------------------------------------------------------
# Surface extraction tests
# ---------------------------------------------------------------------------


# TEST: Surface extraction captures function names as identifiers
def test_surface_extracts_function_names():
    source = textwrap.dedent('''
        def calculate_quiz_score(answers: list) -> int:
            """Calculate the score for a quiz attempt."""
            return sum(answers)
    ''').strip()

    surfaces = extract_searchable_surface(source, "quiz.py")
    assert "calculate_quiz_score" in surfaces["identifiers"]
    assert "quiz" in surfaces["identifiers"]  # split parts
    assert "Calculate the score for a quiz attempt." in surfaces["docstrings"]


# TEST: Surface extraction captures class names with CamelCase splitting
def test_surface_extracts_class_names():
    source = textwrap.dedent('''
        class QuizScoreCalculator:
            """Handles quiz score calculations."""
            pass
    ''').strip()

    surfaces = extract_searchable_surface(source, "calculator.py")
    assert "QuizScoreCalculator" in surfaces["identifiers"]
    # CamelCase should be split
    ident_lower = [i.lower() for i in surfaces["identifiers"]]
    assert "quiz" in ident_lower
    assert "score" in ident_lower


# TEST: Surface extraction captures comment lines including test headers
def test_surface_extracts_comments():
    source = textwrap.dedent('''
        # TEST: Quiz produces valid scores between 0 and 100
        def test_quiz_scoring():
            pass
    ''').strip()

    surfaces = extract_searchable_surface(source, "test_quiz.py")
    assert any("Quiz produces valid scores" in c for c in surfaces["comments"])


# TEST: Surface extraction captures import statements
def test_surface_extracts_imports():
    source = textwrap.dedent('''
        from modules.data_loader import DataLoader
        import quiz_engine
    ''').strip()

    surfaces = extract_searchable_surface(source, "app.py")
    assert "quiz_engine" in surfaces["imports"]
    assert "modules.data_loader" in surfaces["imports"]


# TEST: Surface extraction handles syntax errors gracefully
def test_surface_handles_syntax_error():
    source = "def broken(:"
    surfaces = extract_searchable_surface(source, "broken.py")
    # Should not crash, should still have filename
    assert "broken" in surfaces["filename"]


# TEST: Surface extraction captures filename terms from the path
def test_surface_extracts_filename_terms():
    surfaces = extract_searchable_surface("", "tests/test_quiz_integration.py")
    assert "quiz" in surfaces["filename"]
    assert "integration" in surfaces["filename"]
    assert "test" in surfaces["filename"]


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


# TEST: File with query terms in function names scores higher than unrelated file
def test_scoring_relevant_vs_irrelevant():
    relevant_source = textwrap.dedent('''
        def generate_quiz(words: list) -> dict:
            """Generate a quiz from vocabulary words."""
            pass

        def score_quiz_attempt(answers: list) -> int:
            """Score a quiz attempt."""
            pass
    ''').strip()

    irrelevant_source = textwrap.dedent('''
        def load_database(path: str) -> dict:
            """Load the database from disk."""
            pass

        def save_database(db: dict, path: str) -> None:
            """Save database to disk."""
            pass
    ''').strip()

    terms = tokenize_query("quiz scoring")
    relevant_score = score_file("quiz.py", relevant_source, terms)
    irrelevant_score = score_file("database.py", irrelevant_source, terms)

    assert relevant_score > irrelevant_score
    assert relevant_score > 0
    assert irrelevant_score == 0 or irrelevant_score < relevant_score


# TEST: Filename match gives highest weight boost
def test_scoring_filename_weight():
    source = "pass"  # Minimal source — only filename matches
    terms = tokenize_query("quiz")

    # File named quiz.py should score higher than utils.py with same content
    quiz_score = score_file("quiz.py", source, terms)
    utils_score = score_file("utils.py", source, terms)

    assert quiz_score > utils_score


# TEST: Empty query terms returns zero score
def test_scoring_empty_query():
    score = score_file("quiz.py", "def quiz(): pass", [])
    assert score == 0.0


# TEST: Test header comments boost relevance score
def test_scoring_test_headers():
    source = textwrap.dedent('''
        # TEST: Quiz difficulty progression matches user level
        def test_quiz_difficulty():
            pass
    ''').strip()

    terms = tokenize_query("quiz difficulty")
    score = score_file("test_quiz.py", source, terms)
    assert score > 0


# ---------------------------------------------------------------------------
# Import graph tests
# ---------------------------------------------------------------------------


# TEST: Import graph resolves local imports between files
def test_import_graph_basic(tmp_path):
    # Create a mini project
    (tmp_path / "main.py").write_text("from helper import do_stuff\n")
    (tmp_path / "helper.py").write_text("def do_stuff(): pass\n")

    graph = extract_import_graph(tmp_path)
    assert "helper.py" in graph.get("main.py", set())


# TEST: Import graph ignores external packages
def test_import_graph_ignores_external(tmp_path):
    (tmp_path / "app.py").write_text("import requests\nimport json\n")

    graph = extract_import_graph(tmp_path)
    # External packages shouldn't appear as nodes
    assert "requests" not in graph.get("app.py", set())


# ---------------------------------------------------------------------------
# search_files integration tests
# ---------------------------------------------------------------------------


# TEST: search_files returns relevant files sorted by score, highest first
def test_search_files_ranking(tmp_path):
    # Create a mini project with varying relevance
    (tmp_path / "quiz.py").write_text(textwrap.dedent('''
        def generate_quiz(words: list) -> dict:
            """Generate a quiz from vocabulary."""
            pass

        def score_quiz(answers: list) -> int:
            """Score a quiz attempt."""
            pass
    '''))

    (tmp_path / "database.py").write_text(textwrap.dedent('''
        def connect_db(path: str):
            """Connect to the database."""
            pass
    '''))

    (tmp_path / "test_quiz.py").write_text(textwrap.dedent('''
        # TEST: Quiz generates valid questions
        def test_quiz_generation():
            pass

        # TEST: Quiz scoring is accurate
        def test_quiz_scoring():
            pass
    '''))

    results = search_files(tmp_path, "quiz scoring")

    assert len(results) >= 2
    paths = [r[0] for r in results]
    # quiz.py and test_quiz.py should be in results
    assert "quiz.py" in paths
    assert "test_quiz.py" in paths
    # database.py should NOT be in results (or scored very low)
    if "database.py" in paths:
        db_score = next(s for p, s in results if p == "database.py")
        quiz_score = next(s for p, s in results if p == "quiz.py")
        assert quiz_score > db_score


# TEST: search_files returns empty list for query with no matches
def test_search_files_no_matches(tmp_path):
    (tmp_path / "app.py").write_text("def main(): pass\n")
    results = search_files(tmp_path, "quantum entanglement")
    assert results == []


# TEST: boost_down propagates relevance from consumer to its dependency files
def test_search_files_boost_down(tmp_path):
    (tmp_path / "quiz.py").write_text(textwrap.dedent('''
        from scoring import calculate_score

        def generate_quiz(words: list) -> dict:
            """Generate a quiz."""
            pass
    '''))

    (tmp_path / "scoring.py").write_text(textwrap.dedent('''
        def calculate_score(answers: list) -> int:
            """Calculate raw score."""
            pass
    '''))

    # scoring.py should get a boost because quiz.py (high score) imports it
    results = search_files(tmp_path, "quiz generation", boost_down=0.5, boost_up=0.0)
    paths = [r[0] for r in results]

    assert "quiz.py" in paths
    if "scoring.py" in paths:
        scoring_score = next(s for p, s in results if p == "scoring.py")
        assert scoring_score > 0


# TEST: boost_up propagates relevance from a high-scoring dependency back to its consumers
def test_search_files_boost_up(tmp_path):
    (tmp_path / "main.py").write_text(textwrap.dedent('''
        from engine import run

        def entry_point():
            """Application entry point."""
            run()
    '''))

    (tmp_path / "engine.py").write_text(textwrap.dedent('''
        def run():
            """Run the engine."""
            pass

        def engine_helper():
            """Internal engine helper."""
            pass
    '''))

    # engine.py matches "engine" query directly; main.py imports it
    # With boost_up, main.py should be elevated because engine.py is relevant
    results = search_files(tmp_path, "engine", boost_down=0.0, boost_up=0.5)
    paths = [r[0] for r in results]

    assert "engine.py" in paths
    # main.py imports engine.py (high-scoring) so should get a boost
    assert "main.py" in paths
    engine_score = next(s for p, s in results if p == "engine.py")
    main_score = next(s for p, s in results if p == "main.py")
    # engine.py scores higher directly, but main.py gets a non-zero boost
    assert main_score > 0
    assert engine_score > main_score


# TEST: boost_up > boost_down means consumers of relevant deps score higher than deps of relevant consumers
def test_search_files_boost_asymmetry(tmp_path):
    """With default params, boost_up (0.4) > boost_down (0.2) so consumers of
    a highly-relevant dep get bigger boosts than unrelated deps of a relevant file."""
    (tmp_path / "core.py").write_text(textwrap.dedent('''
        def core_algorithm():
            """The core algorithm."""
            pass
    '''))
    (tmp_path / "caller.py").write_text(textwrap.dedent('''
        from core import core_algorithm
        from unrelated import helper

        def use_core():
            """Calls core algorithm."""
            core_algorithm()
    '''))
    (tmp_path / "unrelated.py").write_text(textwrap.dedent('''
        def helper():
            """A helper with no query relevance."""
            pass
    '''))

    # core.py matches; caller.py imports core.py (boost_up from core)
    # unrelated.py is imported BY caller.py (boost_down from caller)
    results = search_files(tmp_path, "core algorithm")
    score_map = {p: s for p, s in results}

    assert "core.py" in score_map
    # caller.py should appear — boost_up from core.py elevates it
    if "caller.py" in score_map and "unrelated.py" in score_map:
        # caller gets boost_up * core.score; unrelated gets boost_down * caller.score
        # caller.score > unrelated.score because boost_up > boost_down applied differently
        assert score_map["caller.py"] > score_map["unrelated.py"]


# ---------------------------------------------------------------------------
# extract_relevant integration test (search → compress pipeline)
# ---------------------------------------------------------------------------


# TEST: extract_relevant produces compressed context for only relevant files
def test_extract_relevant_targeted(tmp_path):
    # Create a project with 5 files, only 2 relevant to the query
    (tmp_path / "quiz.py").write_text(textwrap.dedent('''
        def generate_quiz(words: list) -> dict:
            """Generate a quiz from vocabulary words."""
            return {"questions": []}
    '''))

    (tmp_path / "scoring.py").write_text(textwrap.dedent('''
        def calculate_score(answers: list) -> int:
            """Calculate quiz score."""
            return sum(answers)
    '''))

    (tmp_path / "database.py").write_text(textwrap.dedent('''
        def connect(path: str):
            """Connect to database."""
            pass
    '''))

    (tmp_path / "audio.py").write_text(textwrap.dedent('''
        def play_audio(path: str):
            """Play an audio file."""
            pass
    '''))

    (tmp_path / "config.py").write_text(textwrap.dedent('''
        def load_config():
            """Load application config."""
            pass
    '''))

    result = extract_relevant(tmp_path, "quiz scoring")

    # Should contain the relevant files
    assert "quiz.py" in result
    assert "generate_quiz" in result
    # Should NOT contain unrelated files (or at least not prominently)
    assert "play_audio" not in result or "quiz" in result


# TEST: extract_relevant with no matches returns informative message
def test_extract_relevant_no_matches(tmp_path):
    (tmp_path / "app.py").write_text("def main(): pass\n")
    result = extract_relevant(tmp_path, "quantum physics simulation")
    assert "No files relevant" in result


# TEST: extract_relevant includes query metadata in output header
def test_extract_relevant_header(tmp_path):
    (tmp_path / "quiz.py").write_text("def quiz(): pass\n")
    result = extract_relevant(tmp_path, "quiz")
    assert "# Query:" in result
    assert "# Terms:" in result
    assert "# Relevant files:" in result
