# Using the AST Fetcher on Other Projects

The AST fetcher compresses any Python codebase into a token-efficient context string
ready to paste into a coding agent prompt. No model calls — fully deterministic.

---

## Quick Start

### On a directory

```bash
# From the context-framework root
uv run python src/ast_fetcher/fetcher.py /path/to/your/project
```

Output goes to stdout. Pipe it or redirect it:

```bash
uv run python src/ast_fetcher/fetcher.py /path/to/your/project > context.txt
```

### On a single file

```bash
uv run python src/ast_fetcher/fetcher.py /path/to/your/project/module.py
```

### Exclude test files

```bash
uv run python src/ast_fetcher/fetcher.py /path/to/your/project --no-tests
```

---

## What the Output Looks Like

For each Python file the fetcher emits:
- Module docstring (first line only)
- All import statements
- Class declarations with:
  - Class-level decorators (`@dataclass`, `@app.route(...)`)
  - Annotated variables (`id: int`, `name: str = "default"`)
  - Plain assignments (`__tablename__ = "users"`)
  - Method signatures with decorators and first docstring line
- Function signatures with decorators, return type, and first docstring line
- `# TEST:` header comments above test functions

Bodies, fixtures, mock setups, and implementation detail are stripped.

**Example input** (`routes.py`):
```python
@app.get("/users")
async def list_users(db: Session = Depends(get_db)) -> list[UserOut]:
    """Return all active users."""
    return db.query(User).filter(User.active == True).all()
```

**Example output**:
```
@app.get('/users')
async def list_users(db: Session = Depends(get_db)) -> list[UserOut]:
    """Return all active users."""
    ...
```

---

## Feeding Output to a Coding Agent

Paste the context output as a system or user message before your task description:

```
<context>
# --- src/models.py ---
class User(Base):
    __tablename__ = 'users'
    id: int
    name: str
    email: str

    def to_dict(self) -> dict:
        """Serialize user to dict."""
        ...

# --- src/routes.py ---
@app.get('/users')
async def list_users(...) -> list[UserOut]:
    ...
</context>

Task: Add a DELETE /users/{id} endpoint that soft-deletes by setting active=False.
```

The agent sees the full API surface and schema without drowning in implementation noise.

---

## What Gets Excluded by Default

| Excluded | Why |
|---|---|
| `spikes/` | Phase 0 experiment code, not production |
| `__pycache__/` | Build artifacts |
| `.git/`, `.venv/`, `venv/` | Not source code |
| `node_modules/` | Not Python |

Add your own exclusions via the `exclude_patterns` parameter when calling
`extract_from_directory()` programmatically.

---

## Programmatic Usage

```python
from pathlib import Path
from src.ast_fetcher.fetcher import extract_from_directory, format_context

sigs = extract_from_directory(
    Path("/path/to/your/project"),
    include_tests=True,
    exclude_patterns=["migrations", "alembic"],  # add project-specific exclusions
)

context = format_context(sigs, max_tokens_approx=4000)
print(context)
```

`format_context` prioritizes test files (they are the behavioral contracts) and
truncates gracefully if the codebase exceeds the token budget.

---

## Tips for Best Results

**Place `# TEST:` comments above decorators, not just `def`:**
```python
# TEST: Expired tokens are rejected with 401          ← put it here
@app.post("/login")
def login(...):
```

**Add a module docstring to every file** — it becomes the one-line description
in the compressed context and helps the agent understand file purpose instantly.

**Adjust `max_tokens_approx`** based on your model's context window:
- Qwen 3.5 27B Q4: stay under 4000 tokens for the context block
- Larger models: can push to 8000–12000

**Run fetcher on itself** to verify it works on your setup:
```bash
uv run python src/ast_fetcher/fetcher.py src/
```

---

## Known Limitations

- Only Python files (`.py`). Other languages are not supported.
- `ast.unparse` normalizes string literals to single quotes — this is cosmetic only,
  does not affect correctness.
- Very large files with hundreds of methods may still be verbose; use
  `max_tokens_approx` to budget aggressively.
