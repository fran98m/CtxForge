"""Tests for the documentation curator — LLM prompt building and response parsing."""

import json
import pytest
from src.doc_curator.curator import build_curator_prompt, parse_curator_response


# TEST: Short markdown is included verbatim without truncation
def test_build_prompt_short_markdown():
    md = "# Small Doc\ndef foo(): pass"
    prompt = build_curator_prompt(md)
    assert "# Small Doc" in prompt
    assert "def foo(): pass" in prompt
    assert "truncated" not in prompt


# TEST: Large markdown triggers smart truncation preserving signature-dense sections
def test_build_prompt_smart_truncation():
    # Build a 50K char document: prose intro + signature-dense section in the middle
    intro = "# Introduction\n" + ("This is prose about the library. " * 200) + "\n"
    sig_section = ("def function_name(param1, param2) -> ReturnType:\n    '''Docstring.'''\n" * 100)
    outro = ("More general content without any code. " * 500) + "\n"
    big_md = intro + sig_section + outro
    assert len(big_md) > 30000

    prompt = build_curator_prompt(big_md, max_chars=10000)
    # Smart truncation should keep the intro (first third)
    assert "Introduction" in prompt
    # And it should prefer the signature-dense section over prose outro
    assert "def function_name" in prompt
    assert "truncated" in prompt


# TEST: Density scoring prefers API content over navigation link blocks
def test_smart_truncation_penalizes_nav_links():
    # First third will be the intro
    intro = "# Overview\n" + ("Overview prose. " * 400) + "\n"
    # Then nav-heavy block (lots of markdown links)
    nav_block = ""
    for i in range(200):
        nav_block += f"    *   [Section {i}](https://example.com/section{i}/)\n"
    # Then actual API content
    api_block = ""
    for i in range(50):
        api_block += f"def api_func_{i}(param) -> Result:\n    '''Does thing {i}.'''\n"
    # Then filler
    filler = ("Just some regular text without links or code. " * 300) + "\n"

    big_md = intro + nav_block + api_block + filler
    assert len(big_md) > 30000

    prompt = build_curator_prompt(big_md, max_chars=12000)
    # API funcs should be preferred over nav links
    assert "api_func_" in prompt
    # Nav links should mostly NOT be in the selected chunks
    nav_count = prompt.count("](https://example.com/section")
    api_count = prompt.count("api_func_")
    assert api_count > nav_count, f"API ({api_count}) should outnumber nav ({nav_count})"


# TEST: Truncation respects max_chars parameter
def test_build_prompt_max_chars_limit():
    big_md = "x" * 50000
    prompt = build_curator_prompt(big_md, max_chars=5000)
    # The prompt template adds overhead, but the markdown portion should be limited
    assert "truncated" in prompt


# TEST: Valid JSON response is parsed correctly
def test_parse_valid_response():
    response = json.dumps({
        "category": "Database",
        "framework": "Redis",
        "signatures": [{"name": "set", "params": "key, value", "returns": "bool"}],
        "next_urls_to_scrape": ["https://redis.io/commands"],
    })
    parsed = parse_curator_response(response)
    assert parsed is not None
    assert parsed["category"] == "Database"
    assert parsed["framework"] == "Redis"
    assert len(parsed["signatures"]) == 1


# TEST: JSON wrapped in markdown code fences is still parsed
def test_parse_response_with_code_fences():
    inner = json.dumps({
        "category": "Auth",
        "framework": "Supabase",
        "signatures": [],
    })
    response = f"```json\n{inner}\n```"
    parsed = parse_curator_response(response)
    assert parsed is not None
    assert parsed["framework"] == "Supabase"


# TEST: Response missing required fields returns None
def test_parse_missing_fields():
    response = json.dumps({"category": "Auth"})
    parsed = parse_curator_response(response)
    assert parsed is None


# TEST: Completely invalid JSON returns None
def test_parse_invalid_json():
    parsed = parse_curator_response("this is not json at all")
    assert parsed is None


# TEST: Truncated JSON (missing closing brace) gives actionable error
def test_parse_truncated_json(capsys):
    truncated = '{"category": "Auth", "framework": "Supabase", "signatures": [{"name": "signIn"'
    parsed = parse_curator_response(truncated)
    assert parsed is None
    captured = capsys.readouterr()
    assert "truncated" in captured.err.lower()
    assert "context window" in captured.err.lower()


# TEST: Non-truncated invalid JSON gets standard parse error
def test_parse_garbage_json(capsys):
    parsed = parse_curator_response("{malformed: true}")
    assert parsed is None
    captured = capsys.readouterr()
    # Should get "Failed to parse" not "truncated"
    assert "failed to parse" in captured.err.lower()
    assert "truncated" not in captured.err.lower()
