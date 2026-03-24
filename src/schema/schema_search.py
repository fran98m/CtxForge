"""
Schema search: FK-aware table scoring and code-to-schema cross-referencing.

Mirrors the logic of schema-search.ts:
- Score tables against a query using column names, enum values, FK targets
- Bidirectional FK graph propagation (related tables get boosted)
- Cross-reference code file names to further boost matching tables

Zero model calls. Pure Python.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.schema.schema import SchemaState, TableSchema
from src.ast_fetcher.search import tokenize_query, _stem


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TableScore:
    name: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class SchemaSearchResult:
    table_schema: TableSchema
    score: float
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

_TABLE_NAME_EXACT = 5.0
_TABLE_NAME_PARTIAL = 3.5    # term matches a _-split segment
_COL_NAME_EXACT = 2.0
_COL_NAME_PARTIAL = 1.0
_ENUM_VALUE = 3.0
_FK_TARGET = 2.5
_COL_COMMENT = 1.5


def score_table(
    table: TableSchema,
    query_terms: list[str],
    state: SchemaState,
) -> TableScore:
    """
    Score a single table's relevance to the given (stemmed) query terms.

    Returns a TableScore with a float score and structured match reasons.
    """
    ts = TableScore(name=table.name)
    if not query_terms:
        return ts

    # Table name segments (split on underscore)
    table_segments = [s.lower() for s in table.name.split("_") if s]

    for term in query_terms:
        stemmed_term = _stem(term)

        # Table name — exact match
        if term == table.name.lower() or stemmed_term == _stem(table.name.lower()):
            ts.score += _TABLE_NAME_EXACT
            ts.reasons.append(f"table name: {table.name}")

        # Table name — partial match (term matches a _-split segment)
        # Also compare original term against segment stems for symmetry
        # e.g. query "session" vs segment "sessions" → _stem("sessions")=="session" ✓
        elif any(
            term == seg or stemmed_term == _stem(seg) or term == _stem(seg)
            for seg in table_segments
        ):
            ts.score += _TABLE_NAME_PARTIAL
            ts.reasons.append(f"table segment: {table.name} ∋ {term}")

        # Column names
        for col_name, col_def in table.columns.items():
            col_segs = [s.lower() for s in col_name.split("_") if s]

            if term == col_name.lower() or stemmed_term == _stem(col_name.lower()):
                ts.score += _COL_NAME_EXACT
                ts.reasons.append(f"col exact: {table.name}.{col_name}")
            elif any(
                term == seg or stemmed_term == _stem(seg) or term == _stem(seg)
                for seg in col_segs
            ):
                ts.score += _COL_NAME_PARTIAL
                ts.reasons.append(f"col partial: {table.name}.{col_name}")

            # Enum values in column
            if col_def.enum_values:
                for val in col_def.enum_values:
                    if term in val.lower() or stemmed_term in _stem(val.lower()):
                        ts.score += _ENUM_VALUE
                        ts.reasons.append(f"enum val: {col_name}={val}")
                        break  # one match per column is enough

            # FK target
            if col_def.references:
                ref_table = col_def.references.split("(")[0].lower()
                if term in ref_table or stemmed_term in _stem(ref_table):
                    ts.score += _FK_TARGET
                    ts.reasons.append(f"fk: {table.name}.{col_name} → {col_def.references}")

            # Column comment
            if col_def.comment:
                if term in col_def.comment.lower():
                    ts.score += _COL_COMMENT
                    ts.reasons.append(f"comment: {col_name}")

        # Named enums in state that match
        for enum_name, values in state.enums.items():
            if term in enum_name or stemmed_term in _stem(enum_name):
                if any(_col.data_type.lower().rstrip("()") in enum_name
                       for _col in table.columns.values()):
                    ts.score += _ENUM_VALUE * 0.5
                    ts.reasons.append(f"enum type: {enum_name}")

    return ts


# ---------------------------------------------------------------------------
# FK graph propagation
# ---------------------------------------------------------------------------

def propagate_fk_scores(
    scores: dict[str, TableScore],
    state: SchemaState,
    boost: float = 0.35,
) -> None:
    """
    Bidirectional FK score propagation — mutates the scores dict in-place.

    Forward  (A → B): A has FK to B, A has score → B gets boosted.
    Backward (B → A): B is a FK target, B has score → tables that FK to B
                      get boosted (weaker, 0.5x).

    Uses a snapshot of scores before propagation to avoid feedback loops.
    """
    snapshot = {name: ts.score for name, ts in scores.items()}

    # Build reverse FK map: target_table → set of tables that FK to it
    fk_sources: dict[str, set[str]] = {}
    for table_name, table in state.tables.items():
        for col in table.columns.values():
            if col.references:
                target = col.references.split("(")[0].lower()
                fk_sources.setdefault(target, set()).add(table_name)

    for table_name, table in state.tables.items():
        source_score = snapshot.get(table_name, 0.0)
        if source_score <= 0:
            continue

        for col in table.columns.values():
            if not col.references:
                continue
            target_table = col.references.split("(")[0].lower()
            if target_table not in state.tables:
                continue

            # Forward boost: FK source → FK target
            if target_table not in scores:
                scores[target_table] = TableScore(name=target_table)
            scores[target_table].score += boost * source_score
            scores[target_table].reasons.append(
                f"fk-boost from {table_name} (score={source_score:.1f})"
            )

        # Backward boost: if this table is a FK target and it has a score,
        # boost all tables that FK to it
        for source_of_this in fk_sources.get(table_name, set()):
            if source_of_this not in scores:
                scores[source_of_this] = TableScore(name=source_of_this)
            scores[source_of_this].score += boost * 0.5 * source_score
            scores[source_of_this].reasons.append(
                f"rev-fk-boost: {source_of_this} → {table_name}"
            )


# ---------------------------------------------------------------------------
# Code-to-schema cross-reference
# ---------------------------------------------------------------------------

def cross_reference_code_to_schema(
    code_file_names: list[str],
    scores: dict[str, TableScore],
    state: SchemaState,
    boost: float = 2.0,
) -> None:
    """
    Boost tables whose names appear in the code file names — mutates scores.

    Derives terms from file names (e.g. "order.service.py" → ["order", "service"])
    and boosts any table whose name (or a segment) matches those terms.
    """
    # Extract terms from all file names
    code_terms: set[str] = set()
    for fname in code_file_names:
        stem = re.sub(r"\.(py|ts|js)$", "", fname.split("/")[-1])
        parts = re.split(r"[._\-]", stem)
        code_terms.update(p.lower() for p in parts if len(p) > 2)

    if not code_terms:
        return

    for table_name, table in state.tables.items():
        table_segs = [s.lower() for s in table_name.split("_") if s]
        for term in code_terms:
            if term in table_name or any(term in seg for seg in table_segs):
                if table_name not in scores:
                    scores[table_name] = TableScore(name=table_name)
                scores[table_name].score += boost
                scores[table_name].reasons.append(f"code-ref: {term}")
                break  # one match per table/term combo


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

def search_schema(
    state: SchemaState,
    query: str,
    top_k: int = 10,
    code_file_names: Optional[list[str]] = None,
) -> list[SchemaSearchResult]:
    """
    Search tables in the schema state for relevance to a natural language query.

    Applies FK-graph propagation and optional code-file cross-referencing.

    Returns up to top_k results sorted by score descending.
    """
    query_terms = tokenize_query(query)

    scores: dict[str, TableScore] = {}
    for table_name, table in state.tables.items():
        ts = score_table(table, query_terms, state)
        if ts.score > 0:
            scores[table_name] = ts

    if scores:
        propagate_fk_scores(scores, state)

    if code_file_names:
        cross_reference_code_to_schema(code_file_names, scores, state)

    ranked = sorted(scores.values(), key=lambda ts: ts.score, reverse=True)

    results: list[SchemaSearchResult] = []
    for ts in ranked[:top_k]:
        if ts.name in state.tables:
            results.append(
                SchemaSearchResult(
                    table_schema=state.tables[ts.name],
                    score=ts.score,
                    reasons=ts.reasons,
                )
            )

    return results


# ---------------------------------------------------------------------------
# YAML serialiser (used by compactor.py)
# ---------------------------------------------------------------------------

def compact_schema_to_yaml(
    state: SchemaState,
    table_names: list[str],
) -> str:
    """
    Serialise the requested tables (and their referenced enums) to YAML.

    Output format::

        schema:
          enums:
            order_status: [pending | active | cancelled]
          orders:
            pk: [id]
            indexes: [(user_id), (created_at)]
            cols:
              id: UUID, not null
              status: ENUM(order_status), not null, default pending [pending | active | cancelled]
              user_id: UUID, not null -> users(id)
    """
    if not table_names:
        return "schema: {}"

    # Collect enums referenced by the included tables
    referenced_enums: dict[str, list[str]] = {}
    for tname in table_names:
        if tname not in state.tables:
            continue
        table = state.tables[tname]
        for col in table.columns.values():
            for enum_name, enum_vals in state.enums.items():
                if enum_name in col.data_type.lower():
                    referenced_enums[enum_name] = enum_vals

    lines: list[str] = ["schema:"]

    # Enums block
    if referenced_enums:
        lines.append("  enums:")
        for ename, vals in sorted(referenced_enums.items()):
            lines.append(f"    {ename}: [{' | '.join(vals)}]")

    # Tables
    for tname in table_names:
        if tname not in state.tables:
            continue
        table = state.tables[tname]
        lines.append(f"  {tname}:")

        if table.primary_key:
            lines.append(f"    pk: [{', '.join(table.primary_key)}]")

        if table.indexes:
            lines.append(f"    indexes: [{', '.join(table.indexes)}]")

        if table.partition_by:
            lines.append(f"    partition_by: {table.partition_by}")

        if table.columns:
            lines.append("    cols:")
            for col_name, col in table.columns.items():
                parts: list[str] = [col.data_type]
                if not col.nullable:
                    parts.append("not null")
                if col.default_value:
                    parts.append(f"default {col.default_value}")
                if col.is_unique and not col.is_primary_key:
                    parts.append("unique")
                if col.references:
                    parts.append(f"-> {col.references}")
                col_str = ", ".join(parts)
                if col.enum_values:
                    col_str += f" [{' | '.join(col.enum_values)}]"
                lines.append(f"      {col_name}: {col_str}")

    return "\n".join(lines)
