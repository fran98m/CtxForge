"""
Documentation curator: LLM-powered categorization and extraction.

Takes clean Markdown from Jina Reader and uses a local LLM (Qwen 3.5)
to categorize the documentation and extract function signatures.

This is the only component that requires a running local model.
All other components are deterministic.
"""

import json
import re
import sys
from typing import Optional


# Prompt template for the curator LLM
CURATOR_PROMPT = """You are a documentation curator. Analyze this library documentation and extract structured information.

IMPORTANT extraction rules:
- Extract ALL functions, methods, classes, and constructors mentioned in the text
- Include functions described in prose, not just code blocks (e.g. "the signIn() method" → extract signIn)
- For overview pages that list components/modules, extract each named component as a signature
- If a page references classes like VectorStoreIndex, QueryEngine, etc., extract those too
- For next_urls: include only documentation page URLs, skip social media, GitHub repos, and external sites

Respond ONLY with valid JSON, no preamble, no markdown backticks.

Required JSON format:
{{
    "category": "one of: Auth, Database, UI, Routing, API, Framework, LLM, RAG, Agent, Indexing, Embedding, Storage, Testing, Utility, Config, Overview, Other",
    "framework": "the library/framework name (e.g., Next.js, Supabase, LlamaIndex)",
    "signatures": [
        {{
            "name": "function, method, or class name",
            "params": "parameter signature if known, else empty string",
            "returns": "return type if known, else empty string",
            "description": "one line description"
        }}
    ],
    "next_urls_to_scrape": ["documentation URLs referenced that should also be scraped"]
}}

Documentation to analyze:
{markdown}
"""


def build_curator_prompt(markdown: str, max_chars: int = 24000) -> str:
    """
    Build the prompt for the curator LLM.

    If the markdown exceeds max_chars, uses a smart truncation strategy:
    takes the first section (likely overview/intro) plus the densest
    middle sections where function signatures tend to live.
    """
    if len(markdown) <= max_chars:
        return CURATOR_PROMPT.format(markdown=markdown)

    # Smart truncation: first chunk (intro) + most content-rich sections
    header_budget = max_chars // 3
    body_budget = max_chars - header_budget - 200  # leave room for truncation notice

    header = markdown[:header_budget]

    # Score remaining chunks by density of code-like content
    remaining = markdown[header_budget:]
    chunk_size = 2000
    chunks = [remaining[i:i+chunk_size] for i in range(0, len(remaining), chunk_size)]

    def _signature_density(text: str) -> int:
        """Score a chunk by how much API content it has vs navigation noise."""
        score = 0
        # Positive: code/API indicators
        for marker in ('def ', 'function ', 'class ', '->', '=>',
                       'param', 'return', 'args', 'import ', '```',
                       '**', 'method', 'constructor', 'instance'):
            score += text.count(marker) * 2
        # Mild positive: parentheses (could be function calls or link syntax)
        score += text.count('(') + text.count(')')
        # Penalty: markdown links are nav noise, not API content
        score -= text.count('](http') * 3
        score -= text.count('](https') * 3
        # Penalty: bullet-only lines (sidebar nav pattern)
        nav_bullets = len(re.findall(r'^\s*\*\s+\[', text, re.MULTILINE))
        score -= nav_bullets * 5
        return max(score, 0)

    scored_chunks = [(i, _signature_density(c), c) for i, c in enumerate(chunks)]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Take the densest chunks, preserving document order
    selected = []
    budget_left = body_budget
    picks = []
    for idx, _score, chunk in scored_chunks:
        if budget_left <= 0:
            break
        picks.append((idx, chunk))
        budget_left -= len(chunk)
    picks.sort(key=lambda x: x[0])  # restore document order

    body = "\n[...truncated...]\n".join(chunk for _, chunk in picks)

    truncated = f"{header}\n\n[...{len(markdown) - max_chars} chars truncated — showing densest sections...]\n\n{body}"
    return CURATOR_PROMPT.format(markdown=truncated)


def parse_curator_response(response: str) -> Optional[dict]:
    """
    Parse the LLM's JSON response. Handles common formatting issues.

    Returns None if parsing fails.
    """
    # Strip markdown code fences if present
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.startswith("json"):
        cleaned = cleaned[4:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Detect truncated JSON — the LLM ran out of context tokens mid-output
        if cleaned.startswith('{') and not cleaned.rstrip().endswith('}'):
            print(
                f"Curator response is truncated JSON (cut off mid-generation). "
                f"The model's context window is too small for this document. "
                f"Try a shorter URL, or increase server context (llama-server --ctx-size).",
                file=sys.stderr,
            )
        else:
            print(f"Failed to parse curator response: {e}", file=sys.stderr)
        return None

    # Validate required fields
    required = {"category", "framework", "signatures"}
    if not required.issubset(data.keys()):
        missing = required - data.keys()
        print(f"Curator response missing fields: {missing}", file=sys.stderr)
        return None

    return data
