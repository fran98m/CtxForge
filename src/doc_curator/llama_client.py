"""
Llama.cpp server client: HTTP interface to a local llama-server.

The only external model dependency in the entire framework.
Talks to a llama-server instance running on localhost via the
OpenAI-compatible /v1/chat/completions endpoint.

No vendor SDK, no langchain, no abstractions — just requests + JSON.
If the server is down, calls fail loudly with a clear message.

Qwen 3.5 uses a hybrid attention architecture (3:1 Gated DeltaNet : Full
Attention) that reduces KV-cache by 19x, enabling 32K+ context on a single
RTX 3090 at Q4 quantization. The budget system auto-detects the server's
context window and uses exact tokenization when available, filling the
window optimally instead of guessing with char/token ratios.

Typical setup:
    llama-server -m qwen3.5-27b-q4.gguf --port 8080 --ctx-size 32768 --n-gpu-layers 99
"""

import sys
import time
from typing import Optional

import requests


DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_TIMEOUT = 120  # local inference can be slow at Q4


def complete(
    prompt: str,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: int = DEFAULT_TIMEOUT,
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    """
    Send a completion request to the llama-server.

    Uses the OpenAI-compatible /v1/chat/completions endpoint.
    Returns the response text, or None if the request fails.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason", "")
        content = choice["message"]["content"]
        if finish_reason == "length":
            print(
                f"[llama] WARNING: response truncated (hit context limit). "
                f"Output may be incomplete. Consider reducing input size "
                f"or increasing server context (--ctx-size).",
                file=sys.stderr,
            )
        return content
    except requests.ConnectionError:
        print(
            f"[llama] Connection refused at {base_url}. "
            f"Is llama-server running?",
            file=sys.stderr,
        )
        return None
    except requests.Timeout:
        print(
            f"[llama] Request timed out after {timeout}s. "
            f"Model may be too large for available VRAM.",
            file=sys.stderr,
        )
        return None
    except requests.RequestException as e:
        print(f"[llama] Request failed: {e}", file=sys.stderr)
        return None
    except (KeyError, IndexError) as e:
        print(f"[llama] Unexpected response format: {e}", file=sys.stderr)
        return None


def health_check(base_url: str = DEFAULT_BASE_URL) -> dict:
    """
    Check if the llama-server is running and what model is loaded.

    Returns a dict with:
        - 'status': 'ok' | 'error'
        - 'message': human-readable status
        - 'model': model name if available
    """
    try:
        # Try /health first (native llama-server endpoint)
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            status = data.get("status", "ok")
            return {
                "status": "ok" if status == "ok" else "loading",
                "message": f"Server is {status}",
                "model": data.get("model", "unknown"),
            }

        # Fallback: try /v1/models (OpenAI-compatible)
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        model_id = models[0]["id"] if models else "unknown"
        return {
            "status": "ok",
            "message": f"Server running with model: {model_id}",
            "model": model_id,
        }
    except requests.ConnectionError:
        return {
            "status": "error",
            "message": f"Cannot connect to {base_url}. Is llama-server running?",
            "model": None,
        }
    except requests.RequestException as e:
        return {
            "status": "error",
            "message": f"Health check failed: {e}",
            "model": None,
        }


def get_server_ctx_size(base_url: str = DEFAULT_BASE_URL) -> Optional[int]:
    """
    Query the llama-server for its context window size.

    Uses the /props endpoint (native llama.cpp), which returns
    default_generation_settings.n_ctx. Returns None if the endpoint
    isn't available or the response is unexpected.
    """
    try:
        resp = requests.get(f"{base_url}/props", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        gen_settings = data.get("default_generation_settings", {})
        n_ctx = gen_settings.get("n_ctx")
        if isinstance(n_ctx, int) and n_ctx > 0:
            return n_ctx
    except (requests.RequestException, KeyError, ValueError):
        pass
    return None


def tokenize(text: str, base_url: str = DEFAULT_BASE_URL) -> Optional[int]:
    """
    Get exact token count for text using the server's actual tokenizer.

    Uses llama.cpp's /tokenize endpoint. Returns None if unavailable,
    in which case callers should fall back to char-based estimates.
    """
    try:
        resp = requests.post(
            f"{base_url}/tokenize",
            json={"content": text, "add_special": False},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        tokens = data.get("tokens", [])
        return len(tokens)
    except (requests.RequestException, KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Context budget calculation
# ---------------------------------------------------------------------------

# Fallback chars-per-token estimate when /tokenize is unavailable.
# Conservative for mixed markdown/code; actual Qwen 3.5 tokenizer is ~3.5-4.
_FALLBACK_CHARS_PER_TOKEN = 3.2

# Overhead of the curator prompt template (without content) + system prompt
# + chat template framing. Measured at ~950 chars ≈ ~300 tokens.
_TEMPLATE_OVERHEAD_TOKENS = 300

# Minimum tokens reserved for the LLM's JSON response.
# Scales with content — more signatures found = more output needed.
_MIN_RESPONSE_TOKENS = 1024
_MAX_RESPONSE_TOKENS = 2048

# Fraction of context window reserved for the response (25%).
# Qwen 3.5's DeltaNet layers are efficient for long prompts,
# so we lean toward larger prompt budgets.
_RESPONSE_FRACTION = 0.25


def _estimate_tokens(text: str, base_url: Optional[str] = None) -> int:
    """
    Get token count — exact via /tokenize, or estimated from char count.
    """
    if base_url:
        exact = tokenize(text, base_url)
        if exact is not None:
            return exact
    return int(len(text) / _FALLBACK_CHARS_PER_TOKEN)


def estimate_content_budget(
    ctx_size: int = 8192,
    base_url: Optional[str] = None,
    template_text: Optional[str] = None,
) -> int:
    """
    Calculate the maximum markdown chars that fit in the LLM context window.

    When base_url is provided, uses exact tokenization for the template.
    Otherwise falls back to char-based estimates.

    The response budget is 25% of ctx_size (capped at _MAX_RESPONSE_TOKENS),
    which is optimal for Qwen 3.5's hybrid attention — the DeltaNet layers
    handle long prompts efficiently, so we maximize input over output.
    """
    response_budget = min(_MAX_RESPONSE_TOKENS,
                          max(_MIN_RESPONSE_TOKENS, int(ctx_size * _RESPONSE_FRACTION)))

    if template_text and base_url:
        template_tokens = _estimate_tokens(template_text, base_url)
    else:
        template_tokens = _TEMPLATE_OVERHEAD_TOKENS

    available_tokens = ctx_size - response_budget - template_tokens
    if available_tokens < 512:
        available_tokens = 512

    return int(available_tokens * _FALLBACK_CHARS_PER_TOKEN)


def curate_document(
    markdown: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    ctx_size: int = 0,
) -> Optional[str]:
    """
    Send documentation markdown to the LLM for categorization and signature extraction.

    Fully dynamic pipeline:
    1. Auto-detect ctx_size from server (or use provided value)
    2. Build prompt with estimated budget
    3. Tokenize the actual prompt for exact fit
    4. If it doesn't fit, rebuild with tighter budget
    5. Calculate exact remaining tokens for response

    ctx_size=0 (default) means auto-detect from the server.
    """
    if ctx_size <= 0:
        detected = get_server_ctx_size(base_url)
        if detected:
            ctx_size = detected
            print(f"[llama] Auto-detected context window: {ctx_size} tokens", file=sys.stderr)
        else:
            ctx_size = 8192
            print(f"[llama] Could not detect context window, using default: {ctx_size}", file=sys.stderr)

    try:
        from .curator import build_curator_prompt
    except ImportError:
        from curator import build_curator_prompt

    # Response budget: 25% of ctx, capped
    response_budget = min(_MAX_RESPONSE_TOKENS,
                          max(_MIN_RESPONSE_TOKENS, int(ctx_size * _RESPONSE_FRACTION)))

    # Phase 1: Build prompt with char-based budget estimate
    char_budget = estimate_content_budget(ctx_size, base_url=base_url)
    prompt = build_curator_prompt(markdown, max_chars=char_budget)

    # Phase 2: Exact-tokenize the prompt if possible
    prompt_tokens = _estimate_tokens(prompt, base_url)

    # Phase 3: Check fit — if prompt + response > ctx, rebuild tighter
    if prompt_tokens + response_budget > ctx_size:
        # How many tokens do we need to shed?
        overshoot = (prompt_tokens + response_budget) - ctx_size
        # Convert back to chars approximately
        tighter_budget = max(2000, char_budget - int(overshoot * _FALLBACK_CHARS_PER_TOKEN * 1.2))
        prompt = build_curator_prompt(markdown, max_chars=tighter_budget)
        prompt_tokens = _estimate_tokens(prompt, base_url)
        print(
            f"[llama] Prompt overshot by ~{overshoot} tok, rebuilt with {tighter_budget} char budget",
            file=sys.stderr,
        )

    # Phase 4: Set response budget to whatever's left
    max_tokens = ctx_size - prompt_tokens
    if max_tokens > _MAX_RESPONSE_TOKENS:
        max_tokens = _MAX_RESPONSE_TOKENS
    if max_tokens < 256:
        max_tokens = 256

    print(
        f"[llama] ctx={ctx_size}, prompt={prompt_tokens} tok, "
        f"response={max_tokens} tok, content={len(markdown)} chars "
        f"({'exact' if tokenize(prompt, base_url) is not None else 'estimated'} tokenization)",
        file=sys.stderr,
    )

    return complete(
        prompt,
        base_url=base_url,
        temperature=0.1,
        max_tokens=max_tokens,
        timeout=timeout,
        system_prompt="You are a documentation analysis tool. Respond only with valid JSON.",
    )


if __name__ == "__main__":
    # Quick health check + optional test completion
    import argparse

    parser = argparse.ArgumentParser(description="Test llama-server connection")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Server base URL")
    parser.add_argument("--test", action="store_true", help="Send a test prompt")
    args = parser.parse_args()

    health = health_check(args.url)
    print(f"Status: {health['status']}")
    print(f"Message: {health['message']}")
    if health.get("model"):
        print(f"Model: {health['model']}")

    if args.test and health["status"] == "ok":
        print("\nSending test prompt...", file=sys.stderr)
        start = time.time()
        result = complete(
            "Respond with exactly: LLAMA_OK",
            base_url=args.url,
            max_tokens=16,
        )
        elapsed = time.time() - start
        print(f"Response: {result}")
        print(f"Latency: {elapsed:.2f}s")
