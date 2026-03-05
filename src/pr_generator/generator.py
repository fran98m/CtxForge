"""
PR Generator: auto-commit and PR summary from spec.

Generates meaningful git commit messages and PR summaries from the
spec that drove the coding agent. Captures the "why" — which concerns
were raised during Phase 0 and how they were resolved.

This is where architectural rationale gets preserved automatically,
without requiring human discipline for git hygiene.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional


def git_add_and_commit(
    repo_path: Path,
    message: str,
    add_all: bool = True,
) -> bool:
    """Stage and commit changes with an auto-generated message."""
    try:
        if add_all:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )

        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_path,
            capture_output=True,
        )

        if result.returncode == 0:
            print("Nothing to commit — working tree clean", file=sys.stderr)
            return False

        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        print(f"Committed: {message[:80]}", file=sys.stderr)
        return True

    except subprocess.CalledProcessError as e:
        print(f"Git error: {e.stderr.decode()}", file=sys.stderr)
        return False


def build_commit_message(
    feature: str,
    da_concerns: Optional[list[str]] = None,
    resolutions: Optional[list[str]] = None,
    constraints_maintained: Optional[list[str]] = None,
) -> str:
    """
    Build a structured commit message from spec components.

    Format: feat: <feature> (DA concern: <concern> → resolved via <resolution>)
    """
    msg = f"feat: {feature}"

    details = []
    if da_concerns and resolutions:
        for concern, resolution in zip(da_concerns, resolutions):
            details.append(f"DA concern: {concern} -> resolved: {resolution}")

    if constraints_maintained:
        details.append(
            f"Constraints maintained: {', '.join(constraints_maintained)}"
        )

    if details:
        msg += "\n\n" + "\n".join(details)

    return msg


def build_pr_summary(
    feature: str,
    spec_source: str,
    da_concerns: Optional[list[str]] = None,
    resolutions: Optional[list[str]] = None,
    constraints_maintained: Optional[list[str]] = None,
) -> str:
    """
    Build a PR summary (~80 tokens) that captures the architectural rationale.

    This is the primary architectural record after the micro-manifest
    is superseded by PR history.
    """
    lines = [
        f"Feature: {feature}",
        f"Spec source: {spec_source}",
    ]

    if da_concerns and resolutions:
        concern_lines = []
        for concern, resolution in zip(da_concerns, resolutions):
            concern_lines.append(f"{concern} -> {resolution}")
        lines.append(f"Concerns addressed: {'; '.join(concern_lines)}")

    if constraints_maintained:
        lines.append(
            f"Constraints maintained: {', '.join(constraints_maintained)}"
        )

    return "\n".join(lines)
