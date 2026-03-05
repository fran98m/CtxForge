"""
Documentation scraper: Jina Reader integration + markdown cleaning.

Converts documentation URLs to clean Markdown by routing through
Jina Reader's API. No BeautifulSoup, no raw HTML parsing.
Modern doc sites are 90% SVGs, Webpack chunks, and CSS —
Jina strips all of that and returns pure content.

Jina's output still contains navigation chrome (sidebars, breadcrumbs,
footer links) which can be 80-90% of the total text. clean_markdown()
strips this noise so the LLM sees actual documentation content.
"""

import re
import sys
from typing import Optional

import requests


JINA_READER_BASE = "https://r.jina.ai"
REQUEST_TIMEOUT = 30


def scrape_url(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[str]:
    """
    Scrape a documentation URL via Jina Reader, returning clean Markdown.

    Returns None if the scrape fails.
    """
    jina_url = f"{JINA_READER_BASE}/{url}"
    try:
        response = requests.get(
            jina_url,
            headers={"Accept": "text/markdown"},
            timeout=timeout,
        )
        response.raise_for_status()
        return clean_markdown(response.text)
    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}", file=sys.stderr)
        return None


def scrape_urls(urls: list[str]) -> dict[str, Optional[str]]:
    """
    Scrape multiple documentation URLs. Returns dict of url → markdown.

    Failed scrapes have None as the value.
    """
    results = {}
    for url in urls:
        print(f"Scraping: {url}", file=sys.stderr)
        results[url] = scrape_url(url)
    return results


# ---------------------------------------------------------------------------
# Markdown cleaning — strip Jina Reader chrome
# ---------------------------------------------------------------------------

# Lines that are purely navigation noise
_NAV_PATTERNS = [
    # --- Skip / anchor links ---
    re.compile(r'^\[Skip to content\]'),                    # Skip-to links
    re.compile(r'^\[Section titled "'),                     # Starlight/Astro section anchors
    re.compile(r'^\[Previous\b.*\]\('),                     # Prev/Next nav links
    re.compile(r'^\[Next\b.*\]\('),

    # --- Image placeholders ---
    re.compile(r'^!\[Image \d+'),                           # Jina image placeholders

    # --- Empty / anchor links ---
    re.compile(r'^\[\s*\]\(http'),                          # Empty bracket links: [](url)
    re.compile(r'^\[\u200b\]\('),                           # Zero-width space anchors (VitePress)
    re.compile(r'^\d+\.\s+\[.*\]\(http'),                  # Numbered list nav: 1. [Link](url)

    # --- UI chrome (theme, keyboard, clipboard) ---
    re.compile(r'^Select theme'),                           # Theme pickers
    re.compile(r'^⌘'),                                     # Keyboard shortcut hints (⌘K, ⌘Ctrl k)
    re.compile(r'^Copy as Markdown'),                       # Clipboard UI
    re.compile(r'^\s*View raw Markdown'),
    re.compile(r'^MCP Server$'),                            # MCP install UI
    re.compile(r'^\s*Copy MCP URL'),
    re.compile(r'cursor://'),                               # IDE deep links

    # --- Social links ---
    re.compile(r'^\[Twitter\]|^\[LinkedIn\]|^\[GitHub\]'),  # Named social links
    re.compile(r'^\[\]\(https://(github|x|twitter|linkedin|youtube)\.com'),  # Empty-bracket socials

    # --- Search UI ---
    re.compile(r'^Type to start searching'),                # MkDocs search prompt
    re.compile(r'^Initializing search$'),                   # MkDocs search init

    # --- GitBook / general platform chrome ---
    re.compile(r'^\[?Powered by\b'),                        # Footer: Powered by GitBook/MkDocs
    re.compile(r'^Was this helpful'),                        # Feedback prompt
    re.compile(r'^On this page$'),                          # TOC heading (standalone)
    re.compile(r'^Last updated\b'),                         # Timestamp footer
    re.compile(r'^Accept Reject$'),                         # Cookie consent buttons
    re.compile(r'^This site uses cookies'),                 # Cookie banner
    re.compile(r'^Working\.\.\.'),                          # Chatbot loading states
    re.compile(r"^I'm here to help"),                       # Chatbot greeting
    re.compile(r'^AI Based on your context'),               # Chatbot label
    re.compile(r'^Send$'),                                  # Chatbot send button (standalone)
    re.compile(r'^More$'),                                  # Menu toggle (standalone)
    re.compile(r'^Ask$'),                                   # Chatbot trigger (standalone)

    # --- Locale / language pickers ---
    re.compile(r'^[\U0001F1E0-\U0001F1FF]{2}\s+\w'),       # Emoji flag pickers: 🇺🇸 English

    # --- MkDocs footer chrome ---
    re.compile(r'^Made with\s+\['),                         # Made with [Material for MkDocs]
    re.compile(r'^Copyright\b'),                            # Copyright © ...
    re.compile(r'^Back to top\b'),                          # Back to top link
    re.compile(r'^\s*Thanks for your feedback'),            # MkDocs feedback widget
]


def _is_nav_line(line: str) -> bool:
    """Return True if the line is navigation chrome, not documentation content."""
    stripped = line.strip()
    if not stripped:
        return False  # blank lines are neutral, not nav
    for pat in _NAV_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def _detect_sidebar_end(lines: list[str]) -> int:
    """
    Detect where the sidebar navigation ends and body content begins.

    Jina Reader dumps the full sidebar nav before the page body. The sidebar
    is a dense block of "* [Link text](url)" lines interspersed with bare
    section labels (e.g., "Getting Started") and blank lines. We look for
    where this dense nav block ends.

    Handles multiple doc framework patterns:
    - Starlight/Astro: indented bullet links with section labels
    - MkDocs-Material: checkbox bullets (* - [x] [Link](url)), repo badges
    - GitBook: section-header bullets (* Get Started), nested links
    - ReadTheDocs/Sphinx: dense bullet-link sidebars
    """
    link_bullet = re.compile(r'^\s*\*\s+\[')
    bare_bullet = re.compile(r'^\s*\*\s*$')
    checkbox_bullet = re.compile(r'^\s*\*\s+- \[[ x]\]')

    def _is_nav_context(line: str) -> bool:
        """Return True if this line is part of sidebar nav context."""
        stripped = line.strip()
        if not stripped:
            return True  # blank lines between nav groups
        if link_bullet.match(stripped):
            return True  # bullet link: *   [Link](url)
        if bare_bullet.match(stripped):
            return True  # bare bullet: *   (section opener)
        # MkDocs checkbox nav: * - [x] [Link](url) text
        if checkbox_bullet.match(stripped):
            return True
        # Any bullet line containing a link (MkDocs expanded items)
        if stripped.startswith('*') and '](http' in stripped:
            return True
        # Image placeholders: ![Image N: ...
        if re.match(r'^!\[Image\s+\d+', stripped):
            return True
        # Empty-bracket links: [](url) — logos, anchors
        if re.match(r'^\[\s*\]\(', stripped) or stripped.startswith('[\u200b]('):
            return True
        # Repository info badges: [name * ver * stars * forks](github_url)
        if re.match(r'^\[.{0,60}\]\(https://github\.com/', stripped):
            return True
        # Product/pricing nav crammed together: "Product[Link](url)"
        if re.match(r'^[A-Z]\w+\[', stripped) and '](http' in stripped:
            return True
        # Lines matching known nav patterns
        if _is_nav_line(stripped):
            return True
        # Short bare text = section label (e.g., "Getting Started")
        # Heading markers (#) or long prose are NOT nav labels
        if (len(stripped) < 60 and not stripped.startswith('#')
                and not stripped.startswith('```') and '](http' not in stripped
                and '.' not in stripped):
            return True
        return False

    def _is_nav_link(line: str) -> bool:
        """Return True if this line is an actual navigation link."""
        stripped = line.strip()
        if link_bullet.match(stripped):
            return True
        # MkDocs checkbox items with links
        if checkbox_bullet.match(stripped) and '](http' in stripped:
            return True
        # Any bullet containing a link
        if stripped.startswith('*') and '](http' in stripped:
            return True
        return False

    # Track the longest dense nav block
    best_run_end = 0
    run_start = 0
    link_count = 0  # actual links in current run

    for i, line in enumerate(lines):
        if _is_nav_context(line):
            if _is_nav_link(line):
                link_count += 1
        else:
            # Run broken — was it a real sidebar? (needs 30+ actual links)
            if link_count >= 30:
                best_run_end = i
            run_start = i + 1
            link_count = 0

    # Check if we ended inside a run
    if link_count >= 30:
        best_run_end = len(lines)

    return best_run_end


def clean_markdown(raw: str) -> str:
    """
    Strip navigation chrome from Jina Reader markdown output.

    Removes:
    - Jina metadata header (Title:, URL Source:, Published Time:)
    - Sidebar navigation (dense blocks of bullet-point links)
    - Section anchor links, social links, theme pickers, image placeholders
    - Duplicated heading patterns from Starlight/Astro doc sites
    - Previous/Next page navigation

    Keeps:
    - Page title and headings
    - Prose paragraphs
    - Code blocks and inline code
    - Content bullet lists (shorter, mixed with prose)
    - Relevant internal links within prose
    """
    lines = raw.split('\n')

    # Phase 1: Strip Jina metadata header
    content_start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'Markdown Content:':
            content_start = i + 1
            break
        # Also detect "Title:" / "URL Source:" / "Published Time:" block
        if i < 10 and (line.startswith('Title:') or line.startswith('URL Source:')
                       or line.startswith('Published Time:')):
            content_start = max(content_start, i + 1)

    lines = lines[content_start:]

    # Phase 2: Detect and skip sidebar navigation block
    sidebar_end = _detect_sidebar_end(lines)
    if sidebar_end > 0:
        # Keep the page title (first heading) if it's before the sidebar
        title_line = None
        for line in lines[:min(5, sidebar_end)]:
            if line.strip() and not line.strip().startswith('[') and not line.strip().startswith('*'):
                title_line = line
                break
        lines = lines[sidebar_end:]
        if title_line and lines and not any(title_line.strip() in l for l in lines[:5]):
            lines = [title_line, ''] + lines

    # Phase 3: Remove individual nav-pattern lines
    cleaned = []
    prev_blank = False
    for line in lines:
        if _is_nav_line(line):
            continue

        # Collapse multiple blank lines
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        prev_blank = is_blank
        cleaned.append(line)

    result = '\n'.join(cleaned).strip()

    # Phase 4: Strip inline anchor noise
    # ReadTheDocs [¶](url "tooltip") permalink anchors
    result = re.sub(r'\[¶\]\([^)]*\)', '', result)
    # VitePress [​](url) zero-width space anchors
    result = re.sub(r'\[\u200b\]\([^)]*\)', '', result)

    return result
