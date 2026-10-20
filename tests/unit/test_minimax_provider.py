"""MiniMax defaults and requests serialized by the real SDK clients, offline."""

from __future__ import annotations

import importlib
import importlib.util
import json
import socket
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
    http_libraries = [httpx]
    if importlib.util.find_spec("httpx2") is not None:
        http_libraries.append(importlib.import_module("httpx2"))

    def deny_network(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("SDK transport tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)

    def respond(_transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        request.read()
        captured.append(request)
        request_body = json.loads(request.content)
        model = request_body["model"]
        library = next(library for library in http_libraries if isinstance(request, library.Request))
        if request_body.get("stream"):
            chunks = [
                {"role": "assistant", "reasoning_content": "Keep "},
                {"reasoning_content": "this reasoning."},
                {"content": "ok"},
            ]
            events = [
                {
                    "id": "chatcmpl_test",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                for delta in chunks
            ]
            stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
            return library.Response(200, content=stream, headers={"content-type": "text/event-stream"}, request=request)
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
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok", "reasoning_content": "Keep this reasoning."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return library.Response(200, json=body, request=request)

    async def arespond(transport: Any, request: Any) -> Any:
        await request.aread()
        return respond(transport, request)

    for library in http_libraries:
        monkeypatch.setattr(library.HTTPTransport, "handle_request", respond)
        monkeypatch.setattr(library.AsyncHTTPTransport, "handle_async_request", arespond)
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
    video = (
        {"type": "video", "source": {"type": "url", "url": "https://example.test/video.mp4"}}
        if base_url.endswith("/anthropic")
        else {"type": "video_url", "video_url": {"url": "https://example.test/video.mp4"}}
    )
    content.append(video)
    assistant_content = (
        [{"type": "thinking", "thinking": "Keep this reasoning.", "signature": "test-signature"}]
        if base_url.endswith("/anthropic")
        else ""
    )
    history = [
        HumanMessage(content=content),
        AIMessage(
            content=assistant_content,
            additional_kwargs={"reasoning_content": "Keep this reasoning."},
            tool_calls=[{"id": "call_test", "name": "lookup", "args": {}}],
        ),
        ToolMessage(content="Found it.", tool_call_id="call_test"),
    ]
    factory.get_llm().bind_tools([tool]).invoke(history)
    body = payload(requests, base_url)
    assert body["model"] == FLASH
    assert body["tools"]
    assert body["messages"][0]["content"][2] == video
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
        assert body["messages"][1]["reasoning_content"] == "Keep this reasoning."


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", ENDPOINTS)
async def test_async_default_request(
    monkeypatch: pytest.MonkeyPatch, requests: list[httpx.Request], base_url: str
) -> None:
    configure(monkeypatch, base_url)
    llm = factory.get_llm()
    answer = await llm.ainvoke("Hello")
    assert payload(requests, base_url)["model"] == FLASH
    if not base_url.endswith("/anthropic"):
        assert answer.additional_kwargs["reasoning_content"] == "Keep this reasoning."
        requests.clear()
        await llm.ainvoke([HumanMessage(content="Hello"), answer, HumanMessage(content="Continue")])
        assert payload(requests, base_url)["messages"][1]["reasoning_content"] == "Keep this reasoning."


@pytest.mark.parametrize("base_url", ENDPOINTS[:2])
@pytest.mark.parametrize("streaming", [False, True])
def test_openai_reasoning_response_replayed(
    monkeypatch: pytest.MonkeyPatch, requests: list[httpx.Request], base_url: str, streaming: bool
) -> None:
    configure(monkeypatch, base_url)
    llm = factory.get_llm()
    if streaming:
        chunks = list(llm.stream("Hello"))
        answer = chunks[0]
        for chunk in chunks[1:]:
            answer += chunk
    else:
        answer = llm.invoke("Hello")
    assert answer.additional_kwargs["reasoning_content"] == "Keep this reasoning."
    requests.clear()
    llm.invoke([HumanMessage(content="Hello"), answer, HumanMessage(content="Continue")])
    body = payload(requests, base_url)
    assert body["messages"][1]["reasoning_content"] == "Keep this reasoning."
