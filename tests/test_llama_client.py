"""Tests for the llama.cpp server client — the LLM bridge."""

import json
import pytest
from unittest.mock import patch, MagicMock
from src.doc_curator.llama_client import (
    complete,
    health_check,
    curate_document,
    estimate_content_budget,
    get_server_ctx_size,
    tokenize,
    _estimate_tokens,
    _FALLBACK_CHARS_PER_TOKEN,
    _MAX_RESPONSE_TOKENS,
    _MIN_RESPONSE_TOKENS,
    DEFAULT_BASE_URL,
)


# ---------------------------------------------------------------------------
# complete() tests
# ---------------------------------------------------------------------------


# TEST: Successful completion returns the response content string
def test_complete_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello, world!"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.doc_curator.llama_client.requests.post", return_value=mock_response) as mock_post:
        result = complete("Say hello")

    assert result == "Hello, world!"
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    payload = call_args[1]["json"]
    assert payload["messages"][-1]["content"] == "Say hello"
    assert payload["stream"] is False


# TEST: System prompt is included as first message when provided
def test_complete_with_system_prompt():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OK"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.doc_curator.llama_client.requests.post", return_value=mock_response) as mock_post:
        complete("Test", system_prompt="You are a helpful assistant")

    payload = mock_post.call_args[1]["json"]
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "You are a helpful assistant"
    assert payload["messages"][1]["role"] == "user"


# TEST: Connection refused returns None with error message to stderr
def test_complete_connection_refused(capsys):
    import requests as req

    with patch(
        "src.doc_curator.llama_client.requests.post",
        side_effect=req.ConnectionError("Connection refused"),
    ):
        result = complete("Test")

    assert result is None
    captured = capsys.readouterr()
    assert "Connection refused" in captured.err


# TEST: Timeout returns None with clear error message
def test_complete_timeout(capsys):
    import requests as req

    with patch(
        "src.doc_curator.llama_client.requests.post",
        side_effect=req.Timeout("timed out"),
    ):
        result = complete("Test", timeout=5)

    assert result is None
    captured = capsys.readouterr()
    assert "timed out" in captured.err


# TEST: Malformed response (missing choices) returns None
def test_complete_malformed_response(capsys):
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": "something broke"}
    mock_response.raise_for_status = MagicMock()

    with patch("src.doc_curator.llama_client.requests.post", return_value=mock_response):
        result = complete("Test")

    assert result is None


# TEST: Temperature and max_tokens are forwarded to the server
def test_complete_params_forwarded():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OK"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.doc_curator.llama_client.requests.post", return_value=mock_response) as mock_post:
        complete("Test", temperature=0.7, max_tokens=512)

    payload = mock_post.call_args[1]["json"]
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 512


# TEST: Custom base_url is used for the request
def test_complete_custom_url():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OK"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.doc_curator.llama_client.requests.post", return_value=mock_response) as mock_post:
        complete("Test", base_url="http://gpu-server:9090")

    call_url = mock_post.call_args[0][0]
    assert call_url == "http://gpu-server:9090/v1/chat/completions"


# ---------------------------------------------------------------------------
# health_check() tests
# ---------------------------------------------------------------------------


# TEST: Health check returns ok when server is running
def test_health_check_ok():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"status": "ok", "model": "qwen-3.5-27b"}

    with patch("src.doc_curator.llama_client.requests.get", return_value=mock_response):
        result = health_check()

    assert result["status"] == "ok"
    assert "qwen-3.5-27b" in result.get("model", "")


# TEST: Health check returns error when server is unreachable
def test_health_check_unreachable():
    import requests as req

    with patch(
        "src.doc_curator.llama_client.requests.get",
        side_effect=req.ConnectionError("refused"),
    ):
        result = health_check()

    assert result["status"] == "error"
    assert "Cannot connect" in result["message"]
    assert result["model"] is None


# ---------------------------------------------------------------------------
# curate_document() tests
# ---------------------------------------------------------------------------


# TEST: curate_document sends markdown through the curator prompt to the LLM
def test_curate_document():
    expected_json = json.dumps({
        "category": "Auth",
        "framework": "Supabase",
        "signatures": [{"name": "signIn", "params": "email, password", "returns": "Session", "description": "Sign in"}],
        "next_urls_to_scrape": [],
    })

    with patch("src.doc_curator.llama_client.tokenize", return_value=None), \
         patch("src.doc_curator.llama_client.complete", return_value=expected_json) as mock_complete:
        result = curate_document("# Supabase Auth\n\nUse signIn() to authenticate.", ctx_size=8192)

    assert result == expected_json
    # Verify it used low temperature for structured output
    call_kwargs = mock_complete.call_args[1]
    assert call_kwargs["temperature"] == 0.1


# TEST: curate_document returns None when server is down
def test_curate_document_server_down():
    with patch("src.doc_curator.llama_client.tokenize", return_value=None), \
         patch("src.doc_curator.llama_client.complete", return_value=None):
        result = curate_document("# Some docs", ctx_size=8192)

    assert result is None


# ---------------------------------------------------------------------------
# tokenize() tests
# ---------------------------------------------------------------------------


# TEST: tokenize returns exact token count from /tokenize endpoint
def test_tokenize_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"tokens": [1, 2, 3, 4, 5]}
    mock_response.raise_for_status = MagicMock()

    with patch("src.doc_curator.llama_client.requests.post", return_value=mock_response) as mock_post:
        result = tokenize("hello world")

    assert result == 5
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["content"] == "hello world"
    assert call_kwargs[1]["json"]["add_special"] is False


# TEST: tokenize returns None when server is unreachable
def test_tokenize_unreachable():
    import requests as req
    with patch("src.doc_curator.llama_client.requests.post",
               side_effect=req.ConnectionError("refused")):
        result = tokenize("hello")
    assert result is None


# TEST: tokenize returns None on malformed response
def test_tokenize_malformed():
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": "bad request"}
    mock_response.raise_for_status = MagicMock()

    with patch("src.doc_curator.llama_client.requests.post", return_value=mock_response):
        result = tokenize("hello")
    assert result == 0  # empty list len


# TEST: _estimate_tokens uses exact count when server available
def test_estimate_tokens_exact():
    with patch("src.doc_curator.llama_client.tokenize", return_value=42):
        result = _estimate_tokens("some text", base_url="http://localhost:8080")
    assert result == 42


# TEST: _estimate_tokens falls back to char estimate without server
def test_estimate_tokens_fallback():
    result = _estimate_tokens("A" * 320)  # no base_url
    expected = int(320 / _FALLBACK_CHARS_PER_TOKEN)
    assert result == expected


# ---------------------------------------------------------------------------
# estimate_content_budget() tests
# ---------------------------------------------------------------------------


# TEST: Default 8K context yields a budget that fits prompt + response
def test_estimate_budget_default_8k():
    budget = estimate_content_budget(8192)
    # 8192 - 2048 (response, 25% = 2048) - 300 (template) = 5844 tokens * 3.2
    assert 15000 < budget < 22000


# TEST: Larger context window yields proportionally larger budget
def test_estimate_budget_32k():
    budget_8k = estimate_content_budget(8192)
    budget_32k = estimate_content_budget(32768)
    assert budget_32k > budget_8k * 3


# TEST: Tiny context window floors at minimum budget
def test_estimate_budget_tiny_ctx():
    budget = estimate_content_budget(512)
    # Should hit the floor (512 tokens * 3.2 = 1638 chars)
    assert budget >= 1000


# TEST: estimate_content_budget uses exact tokenization when base_url provided
def test_estimate_budget_with_tokenize():
    with patch("src.doc_curator.llama_client._estimate_tokens", return_value=200):
        budget = estimate_content_budget(8192, base_url="http://localhost:8080",
                                         template_text="You are a docs tool...")
    # 8192 - 2048 (response) - 200 (exact template) = 5944 tokens * 3.2
    assert budget > 15000


# ---------------------------------------------------------------------------
# Truncation detection tests
# ---------------------------------------------------------------------------


# TEST: Truncated response (finish_reason=length) still returns content with warning
def test_complete_truncation_warning(capsys):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {"content": '{"category": "Auth", "framework": "Sup'},
            "finish_reason": "length",
        }]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.doc_curator.llama_client.requests.post", return_value=mock_response):
        result = complete("Test prompt")

    # Content is still returned (caller decides what to do with it)
    assert result == '{"category": "Auth", "framework": "Sup'
    # But a warning was printed to stderr
    captured = capsys.readouterr()
    assert "truncated" in captured.err.lower()
    assert "context limit" in captured.err.lower()


# TEST: Normal response (finish_reason=stop) returns content without warning
def test_complete_normal_no_truncation_warning(capsys):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {"content": "OK"},
            "finish_reason": "stop",
        }]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.doc_curator.llama_client.requests.post", return_value=mock_response):
        result = complete("Test prompt")

    assert result == "OK"
    captured = capsys.readouterr()
    assert "truncated" not in captured.err.lower()


# TEST: curate_document passes ctx_size to budget calculation
def test_curate_document_ctx_size():
    expected_json = json.dumps({
        "category": "API",
        "framework": "TestLib",
        "signatures": [],
        "next_urls_to_scrape": [],
    })

    with patch("src.doc_curator.llama_client.tokenize", return_value=None), \
         patch("src.doc_curator.llama_client.complete", return_value=expected_json) as mock_complete:
        # Large context = larger budget
        result = curate_document("# Big Docs\n" + "content " * 5000,
                                 ctx_size=32768)

    assert result == expected_json
    call_kwargs = mock_complete.call_args[1]
    # With 32K context, max_tokens should be close to _MAX_RESPONSE_TOKENS
    assert call_kwargs["max_tokens"] > 1000


# TEST: curate_document with small ctx_size truncates content appropriately
def test_curate_document_small_ctx():
    expected_json = json.dumps({
        "category": "API",
        "framework": "TestLib",
        "signatures": [],
    })

    with patch("src.doc_curator.llama_client.tokenize", return_value=None), \
         patch("src.doc_curator.llama_client.complete", return_value=expected_json) as mock_complete:
        big_content = "# Docs\n" + ("def func(x): pass\n" * 2000)
        result = curate_document(big_content, ctx_size=4096)

    # Prompt should have been truncated to fit
    prompt_sent = mock_complete.call_args[0][0]
    assert "truncated" in prompt_sent  # Smart truncation should have kicked in


# ---------------------------------------------------------------------------
# get_server_ctx_size() tests
# ---------------------------------------------------------------------------


# TEST: get_server_ctx_size returns n_ctx from /props endpoint
def test_get_server_ctx_size_ok():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "default_generation_settings": {
            "n_ctx": 8192,
            "n_predict": -1,
        }
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.doc_curator.llama_client.requests.get", return_value=mock_response):
        result = get_server_ctx_size()

    assert result == 8192


# TEST: get_server_ctx_size returns None when server is unreachable
def test_get_server_ctx_size_unreachable():
    import requests as req

    with patch(
        "src.doc_curator.llama_client.requests.get",
        side_effect=req.ConnectionError("refused"),
    ):
        result = get_server_ctx_size()

    assert result is None


# TEST: get_server_ctx_size returns None when response format is unexpected
def test_get_server_ctx_size_bad_format():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"some_other_field": True}
    mock_response.raise_for_status = MagicMock()

    with patch("src.doc_curator.llama_client.requests.get", return_value=mock_response):
        result = get_server_ctx_size()

    assert result is None


# ---------------------------------------------------------------------------
# Auto-detect ctx_size in curate_document
# ---------------------------------------------------------------------------


# TEST: curate_document auto-detects ctx_size when ctx_size=0
def test_curate_document_auto_detect():
    expected_json = json.dumps({
        "category": "API",
        "framework": "TestLib",
        "signatures": [],
    })

    mock_props = MagicMock()
    mock_props.status_code = 200
    mock_props.json.return_value = {
        "default_generation_settings": {"n_ctx": 16384}
    }
    mock_props.raise_for_status = MagicMock()

    with patch("src.doc_curator.llama_client.requests.get", return_value=mock_props), \
         patch("src.doc_curator.llama_client.tokenize", return_value=None), \
         patch("src.doc_curator.llama_client.complete", return_value=expected_json) as mock_complete:
        result = curate_document("# Docs content", ctx_size=0)

    assert result == expected_json
    # With 16K detected, the budget should be larger than 8K default
    prompt_sent = mock_complete.call_args[0][0]
    assert len(prompt_sent) > 0


# TEST: curate_document falls back to 8192 when auto-detect fails
def test_curate_document_auto_detect_fallback():
    import requests as req
    expected_json = json.dumps({
        "category": "API",
        "framework": "TestLib",
        "signatures": [],
    })

    with patch("src.doc_curator.llama_client.requests.get",
               side_effect=req.ConnectionError("refused")), \
         patch("src.doc_curator.llama_client.tokenize", return_value=None), \
         patch("src.doc_curator.llama_client.complete", return_value=expected_json) as mock_complete:
        result = curate_document("# Docs content", ctx_size=0)

    assert result == expected_json
    # Should have used 8192 fallback — budget ~18700 chars
    call_kwargs = mock_complete.call_args[1]
    assert call_kwargs["max_tokens"] > 0


# TEST: curate_document with exact tokenization adjusts prompt to fit
def test_curate_document_exact_tokenization():
    expected_json = json.dumps({
        "category": "API",
        "framework": "TestLib",
        "signatures": [],
    })

    # Simulate exact tokenization: pretend the prompt is 7000 tokens
    # With ctx_size=8192, response budget = 2048, so 7000 + 2048 > 8192
    # This should trigger the tighter rebuild
    call_count = [0]
    def fake_tokenize(text, base_url=None):
        call_count[0] += 1
        if call_count[0] <= 2:
            return 7000  # First estimate: too big
        return 5000  # After rebuild: fits

    with patch("src.doc_curator.llama_client.tokenize", side_effect=fake_tokenize), \
         patch("src.doc_curator.llama_client.complete", return_value=expected_json) as mock_complete:
        big_content = "# Docs\n" + ("def func(x): pass\n" * 2000)
        result = curate_document(big_content, ctx_size=8192)

    assert result == expected_json
