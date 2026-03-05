"""
Self-healing trigger: eliminates stale documentation as a failure variable.

This is NOT a foolproof auto-fix. It is NOT an autonomous repair loop.
It rules out one variable: "is the failure caused by stale docs?"

Tier 1 (deterministic): error trace names a specific function from a specific
library → flag doc stale → re-scrape → retry.

Tier 2 (ambiguous): error doesn't map cleanly to a doc entry → flag and
surface to human. Human decides whether to re-scrape or debug manually.
"""

import re
import sqlite3
import sys
import traceback
from typing import Optional

from .db import get_stale_docs, mark_stale, search_docs


def analyze_error(
    error: Exception,
    error_traceback: str,
    conn: sqlite3.Connection,
) -> dict:
    """
    Analyze a test/runtime failure and determine if stale docs might be the cause.

    Returns:
        {
            "tier": 1 or 2,
            "match": "url of matched doc" or None,
            "suggestion": "human-readable explanation",
            "should_rescrape": True/False,
        }
    """
    error_msg = str(error)
    error_type = type(error).__name__

    # Tier 1: try to match specific function/attribute names in the error
    # Common patterns: "X has no attribute Y", "X is not a function",
    # "cannot import name Y from X", "No module named X"
    patterns = [
        # AttributeError: 'module' object has no attribute 'signIn'
        r"has no attribute '(\w+)'",
        # TypeError: X.signIn() is not a function
        r"(\w+)\(\) is not a function",
        # ImportError: cannot import name 'X' from 'Y'
        r"cannot import name '(\w+)' from '([\w.]+)'",
        # ModuleNotFoundError
        r"No module named '([\w.]+)'",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_msg)
        if match:
            search_term = match.group(1)
            doc_matches = search_docs(conn, search_term)
            if doc_matches:
                return {
                    "tier": 1,
                    "match": doc_matches[0]["url"],
                    "suggestion": (
                        f"Tier 1 match: '{search_term}' found in docs for "
                        f"{doc_matches[0]['framework']}. Likely stale — "
                        f"re-scraping recommended."
                    ),
                    "should_rescrape": True,
                }

    # Tier 2: ambiguous failure — surface to human
    return {
        "tier": 2,
        "match": None,
        "suggestion": (
            f"Tier 2: {error_type} — could not map to a specific doc entry. "
            f"Error: {error_msg[:200]}. Manual investigation recommended."
        ),
        "should_rescrape": False,
    }


def trigger_heal(
    error: Exception,
    error_traceback: str,
    conn: sqlite3.Connection,
) -> dict:
    """
    Run the self-healing analysis and flag stale docs if Tier 1 match found.

    Returns the analysis result. Does NOT auto-rescrape — that's the
    orchestrator's job. This only flags and reports.
    """
    analysis = analyze_error(error, error_traceback, conn)

    if analysis["tier"] == 1 and analysis["match"]:
        mark_stale(conn, analysis["match"])
        print(
            f"[HEAL] Flagged as stale: {analysis['match']}",
            file=sys.stderr,
        )
    elif analysis["tier"] == 2:
        print(
            f"[HEAL] Ambiguous failure — surfacing to human: "
            f"{analysis['suggestion']}",
            file=sys.stderr,
        )

    return analysis
