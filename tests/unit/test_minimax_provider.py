"""MiniMax defaults and requests serialized by the real SDK clients, offline."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import SecretStr

from agentic_architectures.config import Settings
from agentic_architectures.llm import factory
from agentic_architectures.llm.providers import PROVIDERS

ENDPOINTS = [
    "https://api.minimax.io/v1",
    "https://api.minimaxi.com/v1",
    "https://api.minimax.io/anthropic",
    "https://api.minimaxi.com/anthropic",
]
FLASH = "MiniMax-M3.1-Flash"


@pytest.fixture
def requests(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Keep LangChain, SDKs and serialization real; intercept only HTTP transport."""
    captured: list[httpx.Request] = []

    def respond(_transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        request.read()
        captured.append(request)
        model = json.loads(request.content)["model"]
        if request.url.path.endswith("/messages"):
            body = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        else:
            body = {
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    return captured


def configure(monkeypatch: pytest.MonkeyPatch, base_url: str, **kwargs: Any) -> None:
    pytest.importorskip("langchain_anthropic" if base_url.endswith("/anthropic") else "langchain_openai")
    monkeypatch.setattr(
        factory,
        "settings",
        Settings(
            _env_file=None,
            llm_provider="minimax",
            minimax_api_key=SecretStr("test-key"),
            minimax_base_url=base_url,
            **kwargs,
        ),
    )


def payload(requests: list[httpx.Request], base_url: str) -> dict[str, Any]:
    assert len(requests) == 1
    suffix = "/v1/messages" if base_url.endswith("/anthropic") else "/chat/completions"
    assert str(requests[0].url) == base_url + suffix
    assert requests[0].method == "POST"
    return json.loads(requests[0].content)


def test_minimax_settings_and_registry() -> None:
    configured = Settings(_env_file=None, llm_provider="minimax", minimax_api_key=SecretStr("test-key"))
    key = configured.api_key_for("minimax")
    assert key is not None and key.get_secret_value() == "test-key"
    assert PROVIDERS["minimax"].recommended_model == FLASH
    assert "MiniMax-M3" in PROVIDERS["minimax"].notes
    assert "MiniMax-M2" not in PROVIDERS["minimax"].notes
    assert factory.provider_supports_tools("minimax")
    assert factory.provider_supports_structured_output("minimax")


def test_minimax_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "settings", Settings(_env_file=None, minimax_api_key=None))
    with pytest.raises(ValueError, match="MINIMAX_API_KEY is not configured"):
        factory.get_llm(provider="minimax")


@pytest.mark.parametrize("base_url", ENDPOINTS)
@pytest.mark.parametrize(
    ("saved_model", "explicit_model", "expected"),
    [
        (None, None, FLASH),
        (None, "MiniMax-M3", "MiniMax-M3"),
        (None, "MiniMax-M2.7", "MiniMax-M2.7"),
        (None, "custom-model", "custom-model"),
        ("MiniMax-M2.7", None, "MiniMax-M2.7"),
        ("saved-custom", None, "saved-custom"),
        ("meta-llama/Llama-3.3-70B-Instruct", None, "meta-llama/Llama-3.3-70B-Instruct"),
        ("saved-custom", FLASH, FLASH),
    ],
)
def test_model_resolution_on_wire(
    monkeypatch: pytest.MonkeyPatch,
    requests: list[httpx.Request],
    base_url: str,
    saved_model: str | None,
    explicit_model: str | None,
    expected: str,
) -> None:
    if saved_model is not None:
        monkeypatch.setenv("LLM_MODEL", saved_model)
    configure(monkeypatch, base_url)
    llm = factory.get_llm(model=explicit_model, max_tokens=4096)
    assert llm.invoke("Hello").content == "ok"
    body = payload(requests, base_url)
    assert body["model"] == expected
    assert "service_tier" not in body
    assert "reasoning_split" not in body
    assert "thinking" not in body
    assert "reasoning_effort" not in body
    assert "output_config" not in body


@pytest.mark.parametrize("base_url", ENDPOINTS)
def test_explicit_provider_uses_flash_without_configured_model(
    monkeypatch: pytest.MonkeyPatch, requests: list[httpx.Request], base_url: str
) -> None:
    configure(monkeypatch, base_url)
    monkeypatch.setattr(factory.settings, "llm_provider", "nebius")
    factory.get_llm(provider="minimax").invoke("Hello")
    assert payload(requests, base_url)["model"] == FLASH


@pytest.mark.parametrize("base_url", ENDPOINTS)
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_flash_effort_on_wire(
    monkeypatch: pytest.MonkeyPatch, requests: list[httpx.Request], base_url: str, effort: str
) -> None:
    configure(monkeypatch, base_url)
    anthropic = base_url.endswith("/anthropic")
    options = (
        {"thinking": {"type": "adaptive"}, "model_kwargs": {"output_config": {"effort": effort}}}
        if anthropic
        else {"reasoning_effort": effort}
    )
    factory.get_llm(**options).invoke("Hello")
    body = payload(requests, base_url)
    if anthropic:
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": effort}
    else:
        assert body["reasoning_effort"] == effort
    assert body["model"] == FLASH
    assert "service_tier" not in body


@pytest.mark.parametrize("base_url", ENDPOINTS)
@pytest.mark.parametrize("thinking", ["adaptive", "disabled"])
def test_m3_thinking_on_wire(
    monkeypatch: pytest.MonkeyPatch, requests: list[httpx.Request], base_url: str, thinking: str
) -> None:
    configure(monkeypatch, base_url)
    options = (
        {"thinking": {"type": thinking}}
        if base_url.endswith("/anthropic")
        else {"extra_body": {"thinking": {"type": thinking}}}
    )
    factory.get_llm(model="MiniMax-M3", **options).invoke("Hello")
    body = payload(requests, base_url)
    assert body["model"] == "MiniMax-M3"
    assert body["thinking"] == {"type": thinking}


@pytest.mark.parametrize("base_url", ENDPOINTS)
def test_multimodal_and_tool_history_on_wire(
    monkeypatch: pytest.MonkeyPatch, requests: list[httpx.Request], base_url: str
) -> None:
    configure(monkeypatch, base_url)
    image_url = "https://example.test/image.png"
    tool = {"name": "lookup", "description": "Look up an item.", "parameters": {"type": "object", "properties": {}}}
    content = [
        {"type": "text", "text": "Look up this picture."},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    assistant_content = (
        [{"type": "thinking", "thinking": "Keep this reasoning.", "signature": "test-signature"}]
        if base_url.endswith("/anthropic")
        else ""
    )
    history = [
        HumanMessage(content=content),
        AIMessage(content=assistant_content, tool_calls=[{"id": "call_test", "name": "lookup", "args": {}}]),
        ToolMessage(content="Found it.", tool_call_id="call_test"),
    ]
    factory.get_llm().bind_tools([tool]).invoke(history)
    body = payload(requests, base_url)
    assert body["model"] == FLASH
    assert body["tools"]
    if base_url.endswith("/anthropic"):
        assert body["messages"][0]["content"][1] == {"type": "image", "source": {"type": "url", "url": image_url}}
        assert body["messages"][1]["content"][0] == assistant_content[0]
        assert body["messages"][1]["content"][1]["id"] == "call_test"
        assert body["messages"][2]["content"][0]["tool_use_id"] == "call_test"
    else:
        assert body["messages"][0]["content"] == content
        assert body["messages"][1]["tool_calls"][0]["id"] == "call_test"
        assert body["messages"][2]["tool_call_id"] == "call_test"
        assert body["messages"][2]["content"] == "Found it."
