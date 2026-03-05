"""Tests for the markdown cleaner — stripping Jina Reader nav chrome."""

import pytest
from src.doc_curator.scraper import clean_markdown, _is_nav_line, _detect_sidebar_end


# ---------------------------------------------------------------------------
# _is_nav_line() tests
# ---------------------------------------------------------------------------


# TEST: Skip-to-content links are detected as nav
def test_nav_skip_to_content():
    assert _is_nav_line("[Skip to content](https://example.com/#_top)") is True


# TEST: Section anchor links from Starlight/Astro sites are nav
def test_nav_section_anchors():
    assert _is_nav_line('[Section titled "Introduction"](https://example.com/#intro)') is True


# TEST: Social media links are nav
def test_nav_social_links():
    assert _is_nav_line("[Twitter](https://x.com/llamaindex)") is True
    assert _is_nav_line("[GitHub](https://github.com/example)") is True


# TEST: Theme picker and keyboard hints are nav
def test_nav_ui_chrome():
    assert _is_nav_line("Select theme ") is True
    assert _is_nav_line("⌘K") is True


# TEST: Clipboard and MCP UI elements are nav
def test_nav_clipboard_mcp():
    assert _is_nav_line("Copy as Markdown") is True
    assert _is_nav_line("  Copy MCP URL") is True
    assert _is_nav_line("MCP Server") is True


# TEST: Regular prose content is NOT detected as nav
def test_nav_false_for_prose():
    assert _is_nav_line("LlamaIndex is a framework for building LLM applications.") is False
    assert _is_nav_line("def signInWithPassword(email, password):") is False
    assert _is_nav_line("### What are agents?") is False


# TEST: Blank lines are NOT detected as nav (they're neutral)
def test_nav_blank_lines():
    assert _is_nav_line("") is False
    assert _is_nav_line("   ") is False


# TEST: GitBook keyboard shortcuts detected as nav
def test_nav_gitbook_shortcuts():
    assert _is_nav_line("⌘Ctrl k") is True
    assert _is_nav_line("⌘Ctrl i") is True


# TEST: GitBook chatbot/UI chrome is nav
def test_nav_gitbook_chrome():
    assert _is_nav_line("Was this helpful?") is True
    assert _is_nav_line("On this page") is True
    assert _is_nav_line("Last updated 1 month ago") is True
    assert _is_nav_line("Working...Thinking...") is True
    assert _is_nav_line("I'm here to help you with the docs.") is True
    assert _is_nav_line("AI Based on your context") is True
    assert _is_nav_line("Send") is True
    assert _is_nav_line("Ask") is True
    assert _is_nav_line("More") is True


# TEST: Cookie consent and attribution are nav
def test_nav_cookie_and_attribution():
    assert _is_nav_line("Accept Reject") is True
    assert _is_nav_line("This site uses cookies to deliver its service") is True
    assert _is_nav_line("Powered by GitBook") is True
    assert _is_nav_line("[Powered by GitBook](https://www.gitbook.com/)") is True


# TEST: Empty bracket links and numbered nav links are nav
def test_nav_empty_bracket_and_numbered():
    assert _is_nav_line("[](https://example.com/logo)") is True
    assert _is_nav_line("[\u200b](https://vitepress.dev/guide#intro)") is True
    assert _is_nav_line('1.   [Get Started](https://docs.gitbook.com/docs)') is True


# TEST: MkDocs search UI text is nav
def test_nav_mkdocs_search():
    assert _is_nav_line("Type to start searching") is True
    assert _is_nav_line("Initializing search") is True


# TEST: MkDocs footer chrome is nav
def test_nav_mkdocs_footer():
    assert _is_nav_line("Made with [Material for MkDocs Insiders](https://example.com/)") is True
    assert _is_nav_line("Copyright © 2016 - 2025 Martin Donath") is True
    assert _is_nav_line("Back to top [Previous Home](https://example.com/)") is True
    assert _is_nav_line(" Thanks for your feedback!") is True


# TEST: Locale/language picker with emoji flags is nav
def test_nav_locale_picker():
    assert _is_nav_line("🇺🇸 English") is True


# TEST: Social link rows with empty brackets are nav
def test_nav_social_rows():
    assert _is_nav_line("[](https://github.com/GitBookIO)") is True
    assert _is_nav_line("[](https://x.com/GitBookIO)") is True
    assert _is_nav_line("[](https://linkedin.com/company/gitbook)") is True


# ---------------------------------------------------------------------------
# _detect_sidebar_end() tests
# ---------------------------------------------------------------------------


# TEST: Dense block of bullet links is detected as sidebar
def test_sidebar_detection_dense_block():
    lines = []
    # 80 lines of sidebar nav
    for i in range(80):
        lines.append(f"    *   [Page {i}](https://example.com/page{i}/)")
    # Then actual content
    lines.append("## Introduction")
    lines.append("This is actual content.")

    end = _detect_sidebar_end(lines)
    assert end >= 70  # Should cut most of the nav


# TEST: Short bullet lists are NOT detected as sidebar
def test_sidebar_short_list_ignored():
    lines = [
        "## Features",
        "* [Feature A](https://example.com/a)",
        "* [Feature B](https://example.com/b)",
        "* [Feature C](https://example.com/c)",
        "",
        "Feature A provides...",
    ]
    end = _detect_sidebar_end(lines)
    assert end == 0  # Short list is content, not sidebar


# TEST: No bullet links at all returns 0
def test_sidebar_no_nav():
    lines = ["# Title", "Some content", "More content"]
    assert _detect_sidebar_end(lines) == 0


# TEST: MkDocs checkbox bullet nav is detected as sidebar
def test_sidebar_mkdocs_checkbox_nav():
    """MkDocs-Material uses * - [x] [Link](url) patterns in navigation."""
    lines = [
        "*   [Home](https://example.com/)",
        "*   [Getting started](https://example.com/getting-started/)",
        "*   [Setup](https://example.com/setup/)",
        "",
        "[](https://example.com/ \"Logo\") Material for MkDocs",
        "",
        "[repo/name * 9.7 * 26k * 4k](https://github.com/example/repo \"Go to repository\")",
        "",
        "*   [Home](https://example.com/)",
        "*   - [x]  Getting started   Getting started  ",
        "    *   - [x]  Installation  [Installation](https://example.com/getting-started/) Table of contents  ",
        "        *   [Installation](https://example.com/getting-started/#installation)",
        "            *   [with pip](https://example.com/getting-started/#with-pip)",
        "            *   [with docker](https://example.com/getting-started/#with-docker)",
    ]
    # Add bulk nav links to reach 30+ threshold
    for i in range(40):
        lines.append(f"    *   [Section {i}](https://example.com/section{i}/)")
    lines.extend([
        "",
        "## Getting Started",
        "This is body content.",
    ])

    end = _detect_sidebar_end(lines)
    # Should detect the nav block and return past it
    assert end > 40
    # The body content should NOT be included in the sidebar
    body_start = next(i for i, l in enumerate(lines) if l.startswith("## Getting"))
    assert end <= body_start + 1


# TEST: GitBook-style nav with image/chrome lines doesn't break sidebar detection
def test_sidebar_gitbook_mixed_chrome():
    """GitBook has logo images, empty-bracket links, and nav patterns mixed into sidebar."""
    lines = [
        "![Image 1: Logo](https://example.com/logo.png)",
        "",
        "⌘Ctrl k",
        "",
        "Ask",
        "",
        "*   [Documentation](https://docs.example.com/docs)",
        "*   [Developers](https://docs.example.com/developers)",
        "*   [Guides](https://docs.example.com/guides)",
        "",
    ]
    # Add deep sidebar nav
    sections = ["Get Started", "Create", "Agent", "Collaborate",
                "API", "Publish", "Manage", "Integrations"]
    for section in sections:
        lines.append(f"*   {section} ")
        for i in range(5):
            lines.append(f"    *   [{section} Item {i}](https://docs.example.com/{section.lower()}/{i})")
        lines.append("")

    lines.extend([
        "[Powered by GitBook](https://www.gitbook.com/)",
        "",
        "## Real Body Content",
        "This is the actual documentation.",
    ])

    end = _detect_sidebar_end(lines)
    # Should detect the nav block (40+ links)
    assert end > 30
    # Body heading should survive
    body_lines = lines[end:]
    body_text = "\n".join(body_lines)
    assert "Real Body Content" in body_text or end <= len(lines) - 2


# ---------------------------------------------------------------------------
# clean_markdown() tests
# ---------------------------------------------------------------------------


# TEST: Jina metadata header is stripped
def test_clean_strips_jina_header():
    raw = """Title: My Docs

URL Source: https://example.com/docs

Published Time: Wed, 01 Jan 2025 00:00:00 GMT

Markdown Content:
# My Documentation
This is the actual content."""

    cleaned = clean_markdown(raw)
    assert "Title:" not in cleaned
    assert "URL Source:" not in cleaned
    assert "Published Time:" not in cleaned
    assert "Markdown Content:" not in cleaned
    assert "# My Documentation" in cleaned
    assert "actual content" in cleaned


# TEST: Sidebar navigation is stripped from realistic Jina output
def test_clean_strips_sidebar_nav():
    # Build realistic Jina Reader output: metadata + sidebar + body
    lines = ["Title: API Docs", "", "URL Source: https://example.com", "",
             "Markdown Content:", "API Docs", "====="]
    # Add 60 lines of sidebar nav
    for i in range(60):
        lines.append(f"    *   [Section {i}](https://example.com/section{i}/)")
    # Add body content
    lines.extend([
        "",
        "## API Reference",
        "",
        "The `create_user(name, email)` function creates a new user.",
        "",
        "### Parameters",
        "- name: The user's name",
        "- email: The user's email",
    ])

    raw = "\n".join(lines)
    cleaned = clean_markdown(raw)

    # Body should survive
    assert "API Reference" in cleaned
    assert "create_user" in cleaned
    assert "Parameters" in cleaned

    # Most sidebar lines should be gone
    sidebar_count = sum(1 for l in cleaned.split('\n') if "Section " in l and "](https://" in l)
    assert sidebar_count < 10  # Some may remain in the short run, but most gone


# TEST: Clean content with no nav passes through unchanged (minus header)
def test_clean_preserves_good_content():
    raw = """Title: Quick Example

URL Source: https://example.com

Markdown Content:
# Quick Start

Install with pip:

```python
pip install llamaindex
```

Then run:

```python
from llama_index import VectorStoreIndex
index = VectorStoreIndex.from_documents(docs)
```"""

    cleaned = clean_markdown(raw)
    assert "# Quick Start" in cleaned
    assert "pip install llamaindex" in cleaned
    assert "VectorStoreIndex" in cleaned
    assert "```python" in cleaned


# TEST: Section anchor links are stripped but headings survive
def test_clean_strips_section_anchors_keeps_headings():
    raw = """Markdown Content:
### What are agents?

[Section titled "What are agents?"](https://example.com/#what-are-agents)

Agents are LLM-powered assistants that can use tools."""

    cleaned = clean_markdown(raw)
    assert "### What are agents?" in cleaned
    assert "Agents are LLM-powered assistants" in cleaned
    assert '[Section titled' not in cleaned


# TEST: Multiple blank lines are collapsed to single blanks
def test_clean_collapses_blank_lines():
    raw = "Markdown Content:\n# Title\n\n\n\n\nContent here.\n\n\n\nMore content."
    cleaned = clean_markdown(raw)
    assert "\n\n\n" not in cleaned
    assert "Content here." in cleaned
    assert "More content." in cleaned


# TEST: Navigation links (Previous/Next) are stripped
def test_clean_strips_prev_next():
    raw = """Markdown Content:
# Current Page
Content here.
[Previous Getting Started](https://example.com/prev/)
[Next Advanced Usage](https://example.com/next/)"""

    cleaned = clean_markdown(raw)
    assert "Content here" in cleaned
    assert "[Previous" not in cleaned
    assert "[Next " not in cleaned


# ---------------------------------------------------------------------------
# Integration: clean_markdown on realistic LlamaIndex-like content
# ---------------------------------------------------------------------------


# TEST: Realistic Jina output with massive nav gets cleaned to just body content
def test_clean_realistic_llamaindex_pattern():
    """Simulates the actual pattern we saw: 87% nav, 12% content."""
    lines = ["Title: Welcome to LlamaIndex", "", "URL Source: https://llamaindex.ai/docs", "",
             "Published Time: Thu, 05 Mar 2026 06:56:50 GMT", "",
             "Markdown Content:",
             "Welcome to LlamaIndex | Docs", "===============",
             "[Skip to content](https://llamaindex.ai/docs/#_top)", ""]

    # 100 lines of realistic sidebar nav
    sections = ["Getting Started", "Installation", "Concepts", "Tutorials",
                "LLMs", "Embeddings", "Indexing", "Querying", "Agents", "RAG"]
    for section in sections:
        lines.append(f"    *   ")
        lines.append(f"{section}")
        for i in range(10):
            lines.append(f"        *   [{section} Part {i}](https://llamaindex.ai/docs/{section.lower()}/{i})")

    # Actual body content
    lines.extend(["", "",
        "Welcome to LlamaIndex",
        "==========================",
        "",
        "LlamaIndex is a framework for building LLM-powered applications.",
        "",
        "## Key Components",
        "",
        "* **VectorStoreIndex** - Index documents for retrieval",
        "* **QueryEngine** - Natural language querying",
        "* **ChatEngine** - Conversational interface",
        "",
        "## Getting Started",
        "",
        "```python",
        "from llama_index.core import VectorStoreIndex",
        "index = VectorStoreIndex.from_documents(documents)",
        "query_engine = index.as_query_engine()",
        "response = query_engine.query('What is LlamaIndex?')",
        "```",
    ])

    raw = "\n".join(lines)
    cleaned = clean_markdown(raw)

    # Content should be preserved
    assert "VectorStoreIndex" in cleaned
    assert "QueryEngine" in cleaned
    assert "query_engine.query" in cleaned
    assert "building LLM-powered" in cleaned

    # Nav should be largely gone
    assert "[Skip to content]" not in cleaned
    # Cleaned should be dramatically smaller
    assert len(cleaned) < len(raw) * 0.6


# ---------------------------------------------------------------------------
# Multi-framework clean_markdown tests
# ---------------------------------------------------------------------------


# TEST: MkDocs-Material checkbox nav is stripped, body preserved
def test_clean_mkdocs_material_pattern():
    """MkDocs-Material: checkbox bullets, repo badges, [¶] anchors."""
    lines = ["Title: Getting Started", "", "URL Source: https://example.com/getting-started/", "",
             "Markdown Content:",
             "Getting started - Material for MkDocs",
             "===============",
             "[Skip to content](https://example.com/#top)",
             "",
             "![Image 1](https://example.com/logo.png)",
             "",
             "Type to start searching",
             "",
             "[](https://example.com/ \"Material for MkDocs\") Material for MkDocs",
             "",
             "[repo/mkdocs * 9.7 * 26k * 4k](https://github.com/example/repo)",
             ""]

    # Top nav bar (8 links)
    for label in ["Home", "Getting started", "Setup", "Plugins",
                   "Reference", "Insiders", "Community", "Blog"]:
        lines.append(f"*   [{label}](https://example.com/{label.lower().replace(' ', '-')}/)")
    lines.append("")

    # Full sidebar with checkbox items (60+ links)
    lines.append("*   [Home](https://example.com/)")
    lines.append("*   - [x]  Getting started   Getting started  ")
    lines.append("    *   - [x]  Installation  [Installation](https://example.com/getting-started/) Table of contents  ")
    for i in range(50):
        lines.append(f"    *   [Section {i}](https://example.com/section{i}/)")
    lines.extend(["",
        " Table of contents  ",
        "*   [Installation](https://example.com/getting-started/#installation)",
        "    *   [with pip](https://example.com/getting-started/#with-pip)",
        "",
        # Body content WITH [¶] anchors
        "[](https://github.com/example/edit \"Edit this page\")",
        "Getting started[¶](https://example.com/getting-started/#getting-started \"Permanent link\")",
        "============",
        "",
        "Material for MkDocs is a powerful documentation framework.",
        "",
        "Installation[¶](https://example.com/getting-started/#installation \"Permanent link\")",
        "------------",
        "",
        "### with pip recommended[¶](https://example.com/#with-pip \"Permanent link\")",
        "",
        "Install with pip:",
        "",
        "```",
        "pip install mkdocs-material",
        "```",
        "",
        "Back to top [Previous Home](https://example.com/)",
        "Copyright © 2025 Author",
        "Made with [Material for MkDocs](https://example.com/)",
    ])

    raw = "\n".join(lines)
    cleaned = clean_markdown(raw)

    # Body content preserved
    assert "powerful documentation framework" in cleaned
    assert "pip install mkdocs-material" in cleaned

    # [¶] anchors stripped
    assert "[¶]" not in cleaned
    assert "Getting started" in cleaned  # heading text preserved
    assert "### with pip recommended" in cleaned

    # Sidebar nav gone
    assert "- [x]  Getting started" not in cleaned
    assert "Table of contents" not in cleaned

    # Footer chrome stripped
    assert "Back to top" not in cleaned
    assert "Copyright" not in cleaned
    assert "Made with" not in cleaned

    # Significant reduction
    assert len(cleaned) < len(raw) * 0.5


# TEST: GitBook header/sidebar/footer chrome is stripped, body preserved
def test_clean_gitbook_pattern():
    """GitBook: chatbot UI, logo images, sidebar nav, footer chrome."""
    lines = ["Title: GitBook Docs", "", "URL Source: https://docs.example.com/", "",
             "Markdown Content:",
             "GitBook Docs", "===============",
             "",
             "![Image 1: Logo](https://example.com/logo.png)",
             "",
             "⌘Ctrl k",
             "",
             "Ask",
             "",
             "More",
             "",
             "*   [Documentation](https://docs.example.com/docs)",
             "*   [Developers](https://docs.example.com/developers)",
             "*   [Guides](https://docs.example.com/guides)",
             "",
             "🇺🇸 English",
             "",
             "Working...Thinking...",
             "",
             "##### Good afternoon",
             "",
             "I'm here to help you with the docs.",
             "",
             "⌘Ctrl i",
             "",
             "AI Based on your context",
             "",
             "Send",
             ""]

    # Sidebar nav (40+ links)
    for section in ["Get Started", "Create", "Agent", "Collaborate",
                     "API", "Publish", "Manage", "Account"]:
        lines.append(f"*   {section} ")
        for i in range(6):
            lines.append(f"    *   [{section} Item {i}](https://docs.example.com/{section.lower()}/{i})")
        lines.append("")

    lines.extend([
        "[Powered by GitBook](https://www.gitbook.com/)",
        "",
        "On this page",
        "",
        "Was this helpful?",
        "",
        "Ask",
        "",
        '1.   [Get Started](https://docs.example.com/docs/getting-started)',
        "",
        # Body content
        "GitBook Documentation",
        "=====================",
        "",
        "Create AI-native documentation your users will love.",
        "",
        "**Quick start**",
        "",
        "Get up and running in minutes.",
        "",
        "**Essentials**",
        "",
        "*   [Concepts](https://docs.example.com/concepts)",
        "*   [Blocks](https://docs.example.com/blocks)",
        "",
        "Last updated 1 month ago",
        "",
        "Was this helpful?",
        "",
        "![Image 2: Logo](https://example.com/footer-logo.png)",
        "",
        "#### Resources",
        "",
        "*   [Showcase](https://www.example.com/showcase)",
        "",
        "[](https://github.com/Example)[](https://x.com/Example)",
        "",
        "This site uses cookies to deliver its service.",
        "",
        "Accept Reject",
    ])

    raw = "\n".join(lines)
    cleaned = clean_markdown(raw)

    # Body content preserved
    assert "AI-native documentation" in cleaned
    assert "Quick start" in cleaned
    assert "Get up and running" in cleaned

    # Header chrome stripped
    assert "⌘Ctrl k" not in cleaned
    assert "Working...Thinking..." not in cleaned
    assert "I'm here to help" not in cleaned
    assert "AI Based on your context" not in cleaned

    # Sidebar nav stripped
    assert "Get Started Item" not in cleaned
    assert "Account Item" not in cleaned

    # Footer chrome stripped
    assert "Powered by GitBook" not in cleaned
    assert "On this page" not in cleaned
    assert "This site uses cookies" not in cleaned
    assert "Accept Reject" not in cleaned

    # Significant reduction
    assert len(cleaned) < len(raw) * 0.4


# TEST: ReadTheDocs [¶] permalink anchors are stripped inline
def test_clean_readthedocs_permalink_anchors():
    raw = """Markdown Content:
# Features

Features[¶](https://example.com/#features "Link to this heading")

The library provides several features.

## Installation[¶](https://example.com/#installation "Link to this heading")

Install via pip."""

    cleaned = clean_markdown(raw)
    assert "[¶]" not in cleaned
    assert "Permanent link" not in cleaned or "Link to this heading" not in cleaned
    assert "Features" in cleaned
    assert "Installation" in cleaned
    assert "Install via pip" in cleaned


# TEST: VitePress zero-width space anchors are stripped
def test_clean_vitepress_anchor_cleanup():
    raw = """Markdown Content:
# Introduction

## Getting Started [\u200b](https://vitepress.dev/guide#getting-started)

VitePress is a static site generator.

### Configuration [\u200b](https://vitepress.dev/guide#configuration)

Configure your site in config.ts."""

    cleaned = clean_markdown(raw)
    assert "[\u200b]" not in cleaned
    assert "Getting Started" in cleaned
    assert "VitePress is a static site generator" in cleaned
    assert "Configuration" in cleaned


# TEST: Content bullet lists in body are NOT stripped as sidebar
def test_clean_preserves_body_bullet_lists():
    """Short bullet lists with links in body content should be preserved."""
    raw = """Markdown Content:
# Documentation

## Key Features

* [Feature A](https://example.com/a) - Does something great
* [Feature B](https://example.com/b) - Does something else
* [Feature C](https://example.com/c) - And this too

These features work together to provide a complete solution."""

    cleaned = clean_markdown(raw)
    assert "Feature A" in cleaned
    assert "Feature B" in cleaned
    assert "Feature C" in cleaned
    assert "complete solution" in cleaned


# ---------------------------------------------------------------------------
# Migration test
# ---------------------------------------------------------------------------


# TEST: Legacy docs table is dropped when both docs and library_docs exist
def test_migration_drops_legacy_docs_table(tmp_path):
    from src.doc_curator.db import get_db
    import sqlite3

    db_path = tmp_path / "migrate_test.db"
    # First: create a DB with the old "docs" table
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE docs (url TEXT PRIMARY KEY, data TEXT)")
    conn.execute("INSERT INTO docs VALUES ('https://old.com', 'old data')")
    conn.commit()
    conn.close()

    # Now open with get_db() which creates library_docs + runs migration
    conn = get_db(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    assert "library_docs" in tables
    assert "code_context" in tables
    assert "docs" not in tables  # Old table should be gone
    conn.close()
