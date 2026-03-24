"""Tests for the SQL schema migration replay engine."""

import textwrap
import tempfile
from pathlib import Path
import pytest

from src.schema.schema import (
    extract_schema,
    discover_migrations,
    extract_up_section,
    split_statements,
    parse_create_enum,
    parse_create_table,
    parse_alter_table,
    parse_drop_table,
    SchemaState,
    TableSchema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state() -> SchemaState:
    return SchemaState()


def _migration(tmp_path: Path, name: str, sql: str) -> Path:
    p = tmp_path / name
    p.write_text(sql, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# discover_migrations
# ---------------------------------------------------------------------------


# TEST: discover_migrations returns sorted .sql files
def test_discover_migrations_sorted(tmp_path):
    (tmp_path / "20230102_add_orders.sql").write_text(";")
    (tmp_path / "20230101_create_users.sql").write_text(";")
    (tmp_path / "20230103_add_payments.sql").write_text(";")

    results = discover_migrations(str(tmp_path))
    names = [Path(p).name for p in results]
    assert names == [
        "20230101_create_users.sql",
        "20230102_add_orders.sql",
        "20230103_add_payments.sql",
    ]


# TEST: discover_migrations returns empty list for non-existent dir
def test_discover_migrations_missing_dir():
    results = discover_migrations("/does/not/exist")
    assert results == []


# ---------------------------------------------------------------------------
# extract_up_section
# ---------------------------------------------------------------------------


# TEST: extract_up_section returns content between migrate:up and migrate:down
def test_extract_up_section_dbmate(tmp_path):
    sql = textwrap.dedent("""\
        -- migrate:up
        CREATE TABLE users (id UUID PRIMARY KEY);

        -- migrate:down
        DROP TABLE users;
    """)
    f = tmp_path / "migration.sql"
    f.write_text(sql)
    result = extract_up_section(str(f))
    assert "CREATE TABLE users" in result
    assert "DROP TABLE" not in result


# TEST: extract_up_section falls back to whole file when no markers
def test_extract_up_section_plain_sql(tmp_path):
    sql = "CREATE TABLE orders (id UUID PRIMARY KEY);\n"
    f = tmp_path / "plain.sql"
    f.write_text(sql)
    result = extract_up_section(str(f))
    assert "CREATE TABLE orders" in result


# ---------------------------------------------------------------------------
# split_statements
# ---------------------------------------------------------------------------


# TEST: split_statements separates on semicolons
def test_split_statements_basic():
    sql = "CREATE TABLE a (id INT); CREATE TABLE b (id INT);"
    stmts = split_statements(sql)
    assert len(stmts) == 2
    assert any("CREATE TABLE a" in s for s in stmts)
    assert any("CREATE TABLE b" in s for s in stmts)


# TEST: split_statements respects semicolons inside string literals
def test_split_statements_respects_strings():
    sql = "INSERT INTO t VALUES ('a;b'); CREATE TABLE c (id INT);"
    stmts = split_statements(sql)
    # The INSERT with ; inside string and the CREATE TABLE must be separate
    assert len(stmts) == 2


# ---------------------------------------------------------------------------
# parse_create_enum
# ---------------------------------------------------------------------------


# TEST: parse_create_enum extracts enum name and values
def test_parse_create_enum_basic():
    state = _state()
    sql = "CREATE TYPE order_status AS ENUM ('pending', 'active', 'cancelled')"
    result = parse_create_enum(sql, state)
    assert result is True
    assert "order_status" in state.enums
    assert state.enums["order_status"] == ["pending", "active", "cancelled"]


# TEST: parse_create_enum returns False for non-matching SQL
def test_parse_create_enum_no_match():
    state = _state()
    result = parse_create_enum("CREATE TABLE foo (id INT)", state)
    assert result is False
    assert len(state.enums) == 0


# ---------------------------------------------------------------------------
# parse_create_table
# ---------------------------------------------------------------------------


# TEST: parse_create_table creates table with columns
def test_parse_create_table_basic():
    state = _state()
    sql = textwrap.dedent("""\
        CREATE TABLE users (
            id UUID NOT NULL,
            name VARCHAR(255) NOT NULL,
            email TEXT,
            created_at TIMESTAMP
        )
    """)
    result = parse_create_table(sql, state)
    assert result is True
    assert "users" in state.tables
    table = state.tables["users"]
    assert "id" in table.columns
    assert "name" in table.columns
    assert table.columns["id"].nullable is False
    assert table.columns["email"].nullable is True


# TEST: parse_create_table detects PRIMARY KEY column constraint
def test_parse_create_table_pk_inline():
    state = _state()
    sql = "CREATE TABLE items (id UUID PRIMARY KEY, val TEXT)"
    parse_create_table(sql, state)
    table = state.tables["items"]
    assert table.columns["id"].is_primary_key is True
    assert "id" in table.primary_key


# TEST: parse_create_table detects table-level PRIMARY KEY constraint
def test_parse_create_table_pk_table_level():
    state = _state()
    sql = textwrap.dedent("""\
        CREATE TABLE orders (
            id UUID NOT NULL,
            user_id UUID NOT NULL,
            PRIMARY KEY (id)
        )
    """)
    parse_create_table(sql, state)
    table = state.tables["orders"]
    assert "id" in table.primary_key


# TEST: parse_create_table detects FOREIGN KEY references
def test_parse_create_table_foreign_key():
    state = _state()
    sql = textwrap.dedent("""\
        CREATE TABLE orders (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id)
        )
    """)
    parse_create_table(sql, state)
    table = state.tables["orders"]
    assert table.columns["user_id"].references == "users(id)"


# TEST: parse_create_table detects CHECK constraint enum values
def test_parse_create_table_check_enum():
    state = _state()
    sql = textwrap.dedent("""\
        CREATE TABLE orders (
            id UUID PRIMARY KEY,
            status TEXT CHECK (status IN ('pending', 'active', 'cancelled'))
        )
    """)
    parse_create_table(sql, state)
    col = state.tables["orders"].columns["status"]
    assert col.enum_values == ["pending", "active", "cancelled"]


# TEST: parse_create_table detects DEFAULT values
def test_parse_create_table_default():
    state = _state()
    sql = "CREATE TABLE cfg (id INT, active BOOLEAN NOT NULL DEFAULT true)"
    parse_create_table(sql, state)
    col = state.tables["cfg"].columns["active"]
    assert col.default_value == "true"


# ---------------------------------------------------------------------------
# parse_alter_table
# ---------------------------------------------------------------------------


# TEST: parse_alter_table ADD COLUMN adds a new column
def test_alter_table_add_column():
    state = _state()
    parse_create_table("CREATE TABLE users (id UUID PRIMARY KEY)", state)
    sql = "ALTER TABLE users ADD COLUMN email TEXT"
    parse_alter_table(sql, state)
    assert "email" in state.tables["users"].columns


# TEST: parse_alter_table DROP COLUMN removes a column
def test_alter_table_drop_column():
    state = _state()
    parse_create_table("CREATE TABLE users (id UUID PRIMARY KEY, temp TEXT)", state)
    parse_alter_table("ALTER TABLE users DROP COLUMN temp", state)
    assert "temp" not in state.tables["users"].columns


# TEST: parse_alter_table RENAME COLUMN renames a column
def test_alter_table_rename_column():
    state = _state()
    parse_create_table("CREATE TABLE users (id UUID PRIMARY KEY, user_name TEXT)", state)
    parse_alter_table("ALTER TABLE users RENAME COLUMN user_name TO name", state)
    table = state.tables["users"]
    assert "name" in table.columns
    assert "user_name" not in table.columns


# TEST: parse_alter_table ADD CONSTRAINT CHECK sets enum_values
def test_alter_table_add_check_constraint():
    state = _state()
    parse_create_table("CREATE TABLE orders (id UUID, status TEXT)", state)
    sql = "ALTER TABLE orders ADD CONSTRAINT chk_status CHECK (status IN ('a', 'b', 'c'))"
    parse_alter_table(sql, state)
    assert state.tables["orders"].columns["status"].enum_values == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# parse_drop_table
# ---------------------------------------------------------------------------


# TEST: parse_drop_table removes the table from state
def test_parse_drop_table():
    state = _state()
    parse_create_table("CREATE TABLE users (id UUID)", state)
    assert "users" in state.tables
    parse_drop_table("DROP TABLE users", state)
    assert "users" not in state.tables


# ---------------------------------------------------------------------------
# extract_schema integration
# ---------------------------------------------------------------------------


# TEST: extract_schema replays multiple migrations in order
def test_extract_schema_replay_order(tmp_path):
    _migration(tmp_path, "001_create_users.sql", textwrap.dedent("""\
        -- migrate:up
        CREATE TABLE users (
            id UUID PRIMARY KEY,
            name TEXT NOT NULL
        );
        -- migrate:down
        DROP TABLE users;
    """))
    _migration(tmp_path, "002_add_email.sql", textwrap.dedent("""\
        -- migrate:up
        ALTER TABLE users ADD COLUMN email TEXT;
        -- migrate:down
        ALTER TABLE users DROP COLUMN email;
    """))

    state = extract_schema(str(tmp_path))
    assert "users" in state.tables
    table = state.tables["users"]
    assert "id" in table.columns
    assert "name" in table.columns
    assert "email" in table.columns


# TEST: extract_schema discards migrate:down content
def test_extract_schema_ignores_down(tmp_path):
    _migration(tmp_path, "001.sql", textwrap.dedent("""\
        -- migrate:up
        CREATE TABLE keep_table (id UUID);
        -- migrate:down
        DROP TABLE keep_table;
        CREATE TABLE should_not_exist (id UUID);
    """))

    state = extract_schema(str(tmp_path))
    assert "keep_table" in state.tables
    assert "should_not_exist" not in state.tables


# TEST: extract_schema handles plain .sql files with no markers
def test_extract_schema_plain_sql(tmp_path):
    _migration(tmp_path, "create_orders.sql",
               "CREATE TABLE orders (id UUID PRIMARY KEY, total DECIMAL);\n")

    state = extract_schema(str(tmp_path))
    assert "orders" in state.tables


# TEST: extract_schema picks up named enums and links them
def test_extract_schema_named_enums(tmp_path):
    _migration(tmp_path, "001.sql", textwrap.dedent("""\
        -- migrate:up
        CREATE TYPE order_status AS ENUM ('pending', 'active', 'cancelled');
        CREATE TABLE orders (
            id UUID PRIMARY KEY,
            status order_status NOT NULL DEFAULT 'pending'
        );
    """))

    state = extract_schema(str(tmp_path))
    assert "order_status" in state.enums
    assert state.enums["order_status"] == ["pending", "active", "cancelled"]
    assert "orders" in state.tables
