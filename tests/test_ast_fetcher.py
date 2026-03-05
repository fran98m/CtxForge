"""Tests for the AST fetcher — the codebase compression engine."""

import textwrap
import pytest
from src.ast_fetcher.fetcher import extract_signatures, _build_signature, _get_header_comment


# TEST: Simple function extracts to signature + docstring, no body
def test_simple_function_extraction():
    source = textwrap.dedent('''
        def greet(name: str) -> str:
            """Say hello to someone."""
            greeting = f"Hello, {name}!"
            return greeting
    ''').strip()

    result = extract_signatures(source)
    assert "def greet(name: str) -> str:" in result
    assert "Say hello to someone." in result
    assert "greeting = " not in result  # Body must be stripped
    assert "return greeting" not in result


# TEST: Class extracts to declaration + method signatures, no method bodies
def test_class_extraction():
    source = textwrap.dedent('''
        class UserService:
            """Handles user operations."""

            def create_user(self, name: str, email: str) -> dict:
                """Create a new user."""
                user = {"name": name, "email": email}
                db.save(user)
                return user

            def delete_user(self, user_id: int) -> bool:
                """Delete a user by ID."""
                return db.delete(user_id)
    ''').strip()

    result = extract_signatures(source)
    assert "class UserService:" in result
    assert "def create_user(self, name: str, email: str) -> dict:" in result
    assert "def delete_user(self, user_id: int) -> bool:" in result
    assert "db.save" not in result  # Body must be stripped
    assert "db.delete" not in result


# TEST: Header comments (# TEST: ...) are preserved in extraction
def test_header_comment_preservation():
    source = textwrap.dedent('''
        # TEST: User login returns valid session token
        def test_user_login():
            result = login("user", "pass")
            assert result.token is not None
    ''').strip()

    result = extract_signatures(source)
    assert "# TEST: User login returns valid session token" in result
    assert "def test_user_login():" in result
    assert "result = login" not in result  # Body stripped


# TEST: Import statements are preserved in extraction
def test_imports_preserved():
    source = textwrap.dedent('''
        import os
        from pathlib import Path
        from typing import Optional

        def process(path: Path) -> Optional[str]:
            """Process a file."""
            return path.read_text()
    ''').strip()

    result = extract_signatures(source)
    assert "import os" in result
    assert "from pathlib import Path" in result
    assert "from typing import Optional" in result


# TEST: Syntax errors produce clear error message, don't crash
def test_syntax_error_handling():
    source = "def broken(:"
    result = extract_signatures(source, "broken.py")
    assert "PARSE ERROR" in result
    assert "broken.py" in result


# TEST: Default parameter values are included in signatures
def test_default_values():
    source = textwrap.dedent('''
        def fetch(url: str, timeout: int = 30, retries: int = 3) -> dict:
            """Fetch a URL."""
            pass
    ''').strip()

    result = extract_signatures(source)
    assert "timeout: int = 30" in result
    assert "retries: int = 3" in result


# TEST: Async functions are labeled correctly in extraction
def test_async_function():
    source = textwrap.dedent('''
        async def fetch_data(url: str) -> dict:
            """Fetch data asynchronously."""
            async with session.get(url) as resp:
                return await resp.json()
    ''').strip()

    result = extract_signatures(source)
    assert "async def fetch_data(url: str) -> dict:" in result
    assert "session.get" not in result  # Body stripped


# TEST: Empty file produces empty output without errors
def test_empty_file():
    result = extract_signatures("")
    assert result.strip() == ""


# TEST: Decorators on top-level functions (e.g. Flask/FastAPI routes) are preserved
def test_function_decorators():
    source = textwrap.dedent('''
        @app.get("/users")
        def get_users() -> list:
            """Return all users."""
            return db.query(User).all()
    ''').strip()

    result = extract_signatures(source)
    assert "@app.get('/users')" in result
    assert "def get_users() -> list:" in result
    assert "db.query" not in result  # Body stripped


# TEST: Multiple decorators on a class method are all preserved
def test_method_decorators():
    source = textwrap.dedent('''
        class UserRouter:
            @router.get("/users/{id}")
            @requires_auth
            def get_user(self, id: int) -> dict:
                """Get a user by ID."""
                return db.get(id)
    ''').strip()

    result = extract_signatures(source)
    assert "@router.get('/users/{id}')" in result
    assert "@requires_auth" in result
    assert "def get_user(self, id: int) -> dict:" in result
    assert "db.get" not in result  # Body stripped


# TEST: Class-level decorators like @dataclass are preserved
def test_class_decorators():
    source = textwrap.dedent('''
        @dataclass
        class Config:
            """Application configuration."""
            host: str = "localhost"
            port: int = 8080
    ''').strip()

    result = extract_signatures(source)
    assert "@dataclass" in result
    assert "class Config:" in result


# TEST: Pydantic/dataclass annotated class variables are extracted with types and defaults
def test_class_annotated_variables():
    source = textwrap.dedent('''
        class UserModel(BaseModel):
            id: int
            name: str
            email: str
            is_active: bool = True
    ''').strip()

    result = extract_signatures(source)
    assert "id: int" in result
    assert "name: str" in result
    assert "email: str" in result
    assert "is_active: bool = True" in result


# TEST: SQLAlchemy-style plain class assignments (__tablename__ etc.) are extracted
def test_class_plain_assignments():
    source = textwrap.dedent('''
        class User(Base):
            __tablename__ = "users"
            id: int
    ''').strip()

    result = extract_signatures(source)
    assert "__tablename__ = 'users'" in result


# TEST: Header comment placed above a decorator is correctly captured
def test_header_comment_above_decorator():
    source = textwrap.dedent('''
        # TEST: Health endpoint returns status ok
        @app.get("/health")
        def health_check() -> dict:
            return {"status": "ok"}
    ''').strip()

    result = extract_signatures(source)
    assert "# TEST: Health endpoint returns status ok" in result
    assert "@app.get('/health')" in result
    assert "def health_check() -> dict:" in result


# TEST: Module docstring is included (first line only)
def test_module_docstring():
    source = textwrap.dedent('''
        """This module handles authentication and session management."""

        def login(user: str, password: str) -> bool:
            """Authenticate a user."""
            pass
    ''').strip()

    result = extract_signatures(source)
    assert "This module handles authentication" in result
