"""
SQL Schema Module: parses dbmate-format migration files to build a SchemaState.

Replays migrations chronologically (lexicographic sort = dbmate's timestamp
convention) and builds a live view of the current database schema from:
  - CREATE TABLE statements (columns, types, constraints, FKs)
  - ALTER TABLE statements (ADD/DROP/RENAME/ALTER COLUMN)
  - CREATE TYPE ... AS ENUM (Postgres named enums)
  - CHECK constraints (inline enum detection for non-Postgres DBs)
  - DROP TABLE / DROP TYPE

Also works with plain .sql files (no migrate:up/down markers).

Zero external dependencies. Pure Python. Regex-based SQL parsing.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ColumnDef:
    name: str
    data_type: str
    nullable: bool = True
    default_value: Optional[str] = None
    is_primary_key: bool = False
    is_unique: bool = False
    enum_values: Optional[list[str]] = None   # Detected via CHECK or ENUM type
    references: Optional[str] = None          # "table(col)" FK target
    comment: Optional[str] = None


@dataclass
class TableSchema:
    name: str
    columns: dict[str, ColumnDef] = field(default_factory=dict)
    primary_key: list[str] = field(default_factory=list)
    indexes: list[str] = field(default_factory=list)
    partition_by: Optional[str] = None
    enum_types: dict[str, list[str]] = field(default_factory=dict)  # local ENUM cols


@dataclass
class SchemaState:
    tables: dict[str, TableSchema] = field(default_factory=dict)
    enums: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Migration discovery
# ---------------------------------------------------------------------------

def discover_migrations(migrations_dir: str) -> list[str]:
    """
    Return sorted absolute paths to all .sql files in migrations_dir.

    Lexicographic sort matches dbmate's timestamp naming convention
    (e.g. 20230101120000_create_users.sql < 20230102_add_orders.sql).
    """
    d = Path(migrations_dir)
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.rglob("*.sql"))


# ---------------------------------------------------------------------------
# SQL text helpers
# ---------------------------------------------------------------------------

def extract_up_section(filepath: str) -> str:
    """
    Extract the -- migrate:up section from a dbmate migration file.

    Falls back to the whole file content if no markers are present,
    allowing plain .sql files to work too.
    """
    text = Path(filepath).read_text(encoding="utf-8")

    up_match = re.search(r"--\s*migrate:up\b", text, re.IGNORECASE)
    down_match = re.search(r"--\s*migrate:down\b", text, re.IGNORECASE)

    if up_match is None:
        return text  # plain .sql file, use everything

    start = up_match.end()
    end = down_match.start() if down_match and down_match.start() > start else len(text)
    return text[start:end]


def split_statements(sql: str) -> list[str]:
    """
    Split SQL text into individual statements respecting string literals.

    Splits on semicolons that are NOT inside single-quoted strings.
    """
    statements: list[str] = []
    current: list[str] = []
    in_string = False
    i = 0

    while i < len(sql):
        ch = sql[i]
        if ch == "'" and not in_string:
            in_string = True
            current.append(ch)
        elif ch == "'" and in_string:
            # Handle escaped quotes: ''
            if i + 1 < len(sql) and sql[i + 1] == "'":
                current.append("''")
                i += 2
                continue
            in_string = False
            current.append(ch)
        elif ch == ";" and not in_string:
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(ch)
        i += 1

    # Trailing statement without semicolon
    stmt = "".join(current).strip()
    if stmt:
        statements.append(stmt)

    return statements


# ---------------------------------------------------------------------------
# Statement parsers
# ---------------------------------------------------------------------------

def _strip_comments(sql: str) -> str:
    """Remove SQL line comments (--) and block comments (/* */)."""
    # Block comments
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    # Line comments
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


def _parse_col_type(type_str: str) -> tuple[str, Optional[list[str]]]:
    """
    Extract the canonical data type and any inline enum values.

    Handles:
    - Simple types: VARCHAR(255), BIGINT, UUID
    - CHECK constraints: status TEXT CHECK (status IN ('a', 'b'))
    - ENUM literals: ENUM('a', 'b')  (MySQL-style)
    """
    type_str = type_str.strip()
    enum_values: Optional[list[str]] = None

    # MySQL-style ENUM('a', 'b')
    m = re.match(r"ENUM\s*\(([^)]+)\)", type_str, re.IGNORECASE)
    if m:
        enum_values = [v.strip().strip("'\"") for v in m.group(1).split(",")]
        return "ENUM", enum_values

    # Check for type_name REFERENCES other(col)
    type_str = re.sub(r"\s+REFERENCES\s+.+", "", type_str, flags=re.IGNORECASE)

    # Strip trailing CHECK (...) from the type string
    check_m = re.search(r"\bCHECK\s*\((.+)\)\s*$", type_str, re.IGNORECASE | re.DOTALL)
    if check_m:
        check_expr = check_m.group(1)
        in_values = re.findall(r"IN\s*\(([^)]+)\)", check_expr, re.IGNORECASE)
        if in_values:
            enum_values = [v.strip().strip("'\"") for v in in_values[0].split(",")]
        type_str = type_str[: check_m.start()].strip()

    # Normalise: remove trailing constraints from what we treat as the base type
    base_type = re.split(
        r"\s+(NOT\s+NULL|NULL|DEFAULT|UNIQUE|PRIMARY|CHECK|REFERENCES)\b",
        type_str,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    return base_type.upper() if base_type else "TEXT", enum_values


def _parse_fk(col_def_str: str) -> Optional[str]:
    """Extract REFERENCES target(col) from a column definition string."""
    m = re.search(
        r"REFERENCES\s+\"?(\w+)\"?\s*\(\"?(\w+)\"?\)",
        col_def_str,
        re.IGNORECASE,
    )
    return f"{m.group(1)}({m.group(2)})" if m else None


def _col_is_nullable(col_def_str: str) -> bool:
    return "NOT NULL" not in col_def_str.upper()


def _col_default(col_def_str: str) -> Optional[str]:
    m = re.search(r"\bDEFAULT\s+([^\s,)]+)", col_def_str, re.IGNORECASE)
    return m.group(1).strip("'\"") if m else None


def _parse_columns(cols_text: str) -> list[tuple[str, ColumnDef]]:
    """
    Parse a comma-separated column definition block from CREATE TABLE.

    Returns list of (name, ColumnDef) preserving order.
    Skips table-level constraints (PRIMARY KEY, UNIQUE, CHECK, FOREIGN KEY).
    """
    # Split on commas that are NOT inside parentheses
    cols: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in cols_text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            cols.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        cols.append("".join(current).strip())

    result: list[tuple[str, ColumnDef]] = []
    for col_str in cols:
        col_str = col_str.strip()
        if not col_str:
            continue

        # Skip table-level constraints
        upper = col_str.upper().lstrip()
        if re.match(
            r"(PRIMARY\s+KEY|UNIQUE|CHECK|FOREIGN\s+KEY|CONSTRAINT|INDEX|KEY)\b",
            upper,
        ):
            continue

        # Column name (possibly quoted)
        name_m = re.match(r'"?([a-zA-Z_]\w*)"?\s+(.*)', col_str, re.DOTALL)
        if not name_m:
            continue

        col_name = name_m.group(1)
        rest = name_m.group(2)

        data_type, enum_values = _parse_col_type(rest)
        is_pk = bool(re.search(r"\bPRIMARY\s+KEY\b", rest, re.IGNORECASE))
        is_unique = bool(re.search(r"\bUNIQUE\b", rest, re.IGNORECASE))
        nullable = not is_pk and _col_is_nullable(rest)
        default = _col_default(rest)
        references = _parse_fk(rest)

        result.append(
            (
                col_name,
                ColumnDef(
                    name=col_name,
                    data_type=data_type,
                    nullable=nullable,
                    default_value=default,
                    is_primary_key=is_pk,
                    is_unique=is_unique,
                    enum_values=enum_values,
                    references=references,
                ),
            )
        )
    return result


# ---------------------------------------------------------------------------
# CREATE TYPE ... AS ENUM
# ---------------------------------------------------------------------------

_CREATE_ENUM_RE = re.compile(
    r"CREATE\s+TYPE\s+\"?(?:\w+\.)?(\w+)\"?\s+AS\s+ENUM\s*\(([^)]+)\)",
    re.IGNORECASE | re.DOTALL,
)


def parse_create_enum(sql: str, state: SchemaState) -> bool:
    m = _CREATE_ENUM_RE.search(sql)
    if not m:
        return False
    name = m.group(1).lower()
    values = [v.strip().strip("'\"") for v in m.group(2).split(",")]
    state.enums[name] = values
    return True


# ---------------------------------------------------------------------------
# CREATE TABLE
# ---------------------------------------------------------------------------

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+(?:UNLOGGED\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\"?(?:\w+\.)?(\w+)\"?\s*\((.+)\)",
    re.IGNORECASE | re.DOTALL,
)


def parse_create_table(sql: str, state: SchemaState) -> bool:
    m = _CREATE_TABLE_RE.search(sql)
    if not m:
        return False

    table_name = m.group(1).lower()
    cols_text = m.group(2)

    table = TableSchema(name=table_name)

    for col_name, col_def in _parse_columns(cols_text):
        table.columns[col_name] = col_def
        if col_def.is_primary_key:
            table.primary_key.append(col_name)

    # Table-level PRIMARY KEY (...) override
    pk_m = re.search(
        r"\bPRIMARY\s+KEY\s*\(([^)]+)\)",
        cols_text,
        re.IGNORECASE,
    )
    if pk_m:
        table.primary_key = [
            c.strip().strip('"') for c in pk_m.group(1).split(",")
        ]
        for col_name in table.primary_key:
            if col_name in table.columns:
                table.columns[col_name].is_primary_key = True
                table.columns[col_name].nullable = False

    # Partition by clause (outside the parentheses)
    part_m = re.search(
        r"\)\s*PARTITION\s+BY\s+(\w+\s*\([^)]+\))",
        sql,
        re.IGNORECASE,
    )
    if part_m:
        table.partition_by = part_m.group(1).strip()

    state.tables[table_name] = table
    return True


# ---------------------------------------------------------------------------
# ALTER TABLE
# ---------------------------------------------------------------------------

_ALTER_TABLE_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?\"?(?:\w+\.)?(\w+)\"?\s+(.+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_alter_table(sql: str, state: SchemaState) -> bool:
    m = _ALTER_TABLE_RE.search(sql)
    if not m:
        return False

    table_name = m.group(1).lower()
    if table_name not in state.tables:
        return False

    table = state.tables[table_name]
    action = m.group(2).strip()

    # ADD CONSTRAINT must be checked BEFORE ADD COLUMN because ADD COLUMN
    # uses an optional COLUMN keyword and would otherwise match "CONSTRAINT"
    # as the column name.
    constraint_m = re.match(
        r"ADD\s+CONSTRAINT\s+\w+\s+CHECK\b",
        action,
        re.IGNORECASE,
    )
    if constraint_m:
        in_m = re.search(
            r"\"?(\w+)\"?\s+IN\s*\(([^)]+)\)", action, re.IGNORECASE
        )
        if in_m:
            col_name = in_m.group(1)
            values = [v.strip().strip("'\"") for v in in_m.group(2).split(",")]
            if col_name in table.columns:
                table.columns[col_name].enum_values = values
        return True

    # ADD COLUMN
    add_m = re.match(
        r"ADD\s+(?:COLUMN\s+)?(?:IF\s+NOT\s+EXISTS\s+)?\"?(\w+)\"?\s+(.+)",
        action,
        re.IGNORECASE | re.DOTALL,
    )
    if add_m:
        col_name = add_m.group(1)
        rest = add_m.group(2).strip().rstrip(")")
        data_type, enum_values = _parse_col_type(rest)
        is_pk = bool(re.search(r"\bPRIMARY\s+KEY\b", rest, re.IGNORECASE))
        nullable = not is_pk and _col_is_nullable(rest)
        table.columns[col_name] = ColumnDef(
            name=col_name,
            data_type=data_type,
            nullable=nullable,
            default_value=_col_default(rest),
            is_primary_key=is_pk,
            enum_values=enum_values,
            references=_parse_fk(rest),
        )
        return True

    # DROP COLUMN
    drop_m = re.match(
        r"DROP\s+COLUMN\s+(?:IF\s+EXISTS\s+)?\"?(\w+)\"?",
        action,
        re.IGNORECASE,
    )
    if drop_m:
        table.columns.pop(drop_m.group(1), None)
        return True

    # RENAME COLUMN
    rename_m = re.match(
        r"RENAME\s+COLUMN\s+\"?(\w+)\"?\s+TO\s+\"?(\w+)\"?",
        action,
        re.IGNORECASE,
    )
    if rename_m:
        old, new = rename_m.group(1), rename_m.group(2)
        if old in table.columns:
            col = table.columns.pop(old)
            col.name = new
            table.columns[new] = col
        return True

    # ALTER COLUMN ... TYPE
    retype_m = re.match(
        r"ALTER\s+COLUMN\s+\"?(\w+)\"?\s+(?:SET\s+DATA\s+)?TYPE\s+(.+?)(?:\s+USING\s+.+)?$",
        action,
        re.IGNORECASE,
    )
    if retype_m:
        col_name = retype_m.group(1)
        new_type = retype_m.group(2).strip()
        if col_name in table.columns:
            data_type, enum_values = _parse_col_type(new_type)
            table.columns[col_name].data_type = data_type
            if enum_values:
                table.columns[col_name].enum_values = enum_values
        return True

    # ALTER COLUMN ... SET DEFAULT / DROP DEFAULT / SET NOT NULL / DROP NOT NULL
    alter_col_m = re.match(
        r"ALTER\s+COLUMN\s+\"?(\w+)\"?\s+(SET\s+DEFAULT\s+(.+)|DROP\s+DEFAULT|SET\s+NOT\s+NULL|DROP\s+NOT\s+NULL)",
        action,
        re.IGNORECASE,
    )
    if alter_col_m:
        col_name = alter_col_m.group(1)
        sub_action = alter_col_m.group(2).upper()
        if col_name in table.columns:
            if sub_action.startswith("SET DEFAULT"):
                default_m = re.match(r"SET\s+DEFAULT\s+(.+)", alter_col_m.group(2), re.IGNORECASE)
                if default_m:
                    table.columns[col_name].default_value = default_m.group(1).strip().strip("'\"")
            elif sub_action == "DROP DEFAULT":
                table.columns[col_name].default_value = None
            elif sub_action == "SET NOT NULL":
                table.columns[col_name].nullable = False
            elif sub_action == "DROP NOT NULL":
                table.columns[col_name].nullable = True
        return True

    return False


# ---------------------------------------------------------------------------
# DROP TABLE / DROP TYPE
# ---------------------------------------------------------------------------

_DROP_TABLE_RE = re.compile(
    r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?\"?(?:\w+\.)?(\w+)\"?",
    re.IGNORECASE,
)

_DROP_TYPE_RE = re.compile(
    r"DROP\s+TYPE\s+(?:IF\s+EXISTS\s+)?\"?(\w+)\"?",
    re.IGNORECASE,
)


def parse_drop_table(sql: str, state: SchemaState) -> bool:
    m = _DROP_TABLE_RE.search(sql)
    if not m:
        return False
    state.tables.pop(m.group(1).lower(), None)
    return True


def parse_drop_type(sql: str, state: SchemaState) -> bool:
    m = _DROP_TYPE_RE.search(sql)
    if not m:
        return False
    state.enums.pop(m.group(1).lower(), None)
    return True


# ---------------------------------------------------------------------------
# CREATE INDEX
# ---------------------------------------------------------------------------

_CREATE_INDEX_RE = re.compile(
    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?\w+\s+ON\s+\"?(?:\w+\.)?(\w+)\"?\s*\(([^)]+)\)",
    re.IGNORECASE,
)


def parse_create_index(sql: str, state: SchemaState) -> bool:
    m = _CREATE_INDEX_RE.search(sql)
    if not m:
        return False
    table_name = m.group(1).lower()
    cols = m.group(2).strip()
    if table_name in state.tables:
        state.tables[table_name].indexes.append(f"({cols})")
    return True


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

def extract_schema(migrations_dir: str) -> SchemaState:
    """
    Replay all migrations in lexicographic order to build the current SchemaState.

    Supports dbmate-style files (-- migrate:up / -- migrate:down) and plain .sql.
    """
    state = SchemaState()
    migration_files = discover_migrations(migrations_dir)

    for filepath in migration_files:
        try:
            sql = extract_up_section(filepath)
        except (OSError, UnicodeDecodeError):
            continue

        sql = _strip_comments(sql)
        for stmt in split_statements(sql):
            stmt = stmt.strip()
            if not stmt:
                continue
            upper = stmt.upper()
            if "CREATE TYPE" in upper and "AS ENUM" in upper:
                parse_create_enum(stmt, state)
            elif "CREATE TABLE" in upper:
                parse_create_table(stmt, state)
            elif "ALTER TABLE" in upper:
                parse_alter_table(stmt, state)
            elif "DROP TABLE" in upper:
                parse_drop_table(stmt, state)
            elif "DROP TYPE" in upper:
                parse_drop_type(stmt, state)
            elif "CREATE INDEX" in upper or "CREATE UNIQUE INDEX" in upper:
                parse_create_index(stmt, state)

    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.schema.schema <migrations_dir>", file=sys.stderr)
        sys.exit(1)

    state = extract_schema(sys.argv[1])

    output = {
        "enums": state.enums,
        "tables": {
            name: {
                "primary_key": t.primary_key,
                "indexes": t.indexes,
                "columns": {
                    col_name: {
                        "type": c.data_type,
                        "nullable": c.nullable,
                        "default": c.default_value,
                        "pk": c.is_primary_key,
                        "unique": c.is_unique,
                        "enum_values": c.enum_values,
                        "references": c.references,
                    }
                    for col_name, c in t.columns.items()
                },
            }
            for name, t in state.tables.items()
        },
    }
    print(json.dumps(output, indent=2))
