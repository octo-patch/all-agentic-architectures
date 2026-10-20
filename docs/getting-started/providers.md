# Switching LLM providers

Every architecture in this library is provider-agnostic. The same `Reflection` class works against Nebius Llama, OpenAI GPT-4o, Anthropic Claude, Groq, Ollama, Together, Fireworks, Mistral, Google Gemini, or MiniMax — you change one config setting.

## The `get_llm()` factory

```python
from agentic_architectures import get_llm

# Use whatever's in .env / settings
llm = get_llm()

# Override per-call
llm = get_llm(provider="openai", model="gpt-4o-mini", temperature=0.4)
llm = get_llm(provider="anthropic", model="claude-haiku-4-5-20251001")
llm = get_llm(provider="minimax", model="MiniMax-M3.1-Flash")
llm = get_llm(provider="minimax", model="MiniMax-M3")
llm = get_llm(provider="ollama", model="llama3.3:70b")
```

Pass the returned LLM into any architecture:

```python
from agentic_architectures.architectures import Reflection
arch = Reflection(llm=llm)
```

## Provider compatibility matrix

| Provider | Extra to install | Env var | Notes |
|---|---|---|---|
| **Nebius** (default) | `[nebius]` | `NEBIUS_API_KEY` | Llama-3.3-70B + Qwen3-Thinking. Cheapest for the included demos. |
| **OpenAI** | `[openai]` | `OPENAI_API_KEY` | All architectures work. Highest quality for reasoning architectures. |
| **Anthropic** | `[anthropic]` | `ANTHROPIC_API_KEY` | Strong on long context. Required for true Computer-Use (nb 34) production. |
| **Groq** | `[groq]` | `GROQ_API_KEY` | Fast inference; good for high-volume Self-Consistency (nb 21). |
| **Ollama** | `[ollama]` | — | Fully local; no API key. Tool calling depends on model — Llama-3.3 instruct works. |
| **Together** | `[together]` | `TOGETHER_API_KEY` | Wide model catalogue. |
| **Fireworks** | `[fireworks]` | `FIREWORKS_API_KEY` | Function-calling first-class. |
| **Mistral** | `[mistral]` | `MISTRAL_API_KEY` | EU-hosted option. |
| **Google** | `[google]` | `GOOGLE_API_KEY` | Gemini-2.x via Generative AI API. |
| **MiniMax** | `[minimax]` | `MINIMAX_API_KEY` | MiniMax-M3.1-Flash (default) and MiniMax-M3 through OpenAI- or Anthropic-compatible APIs. |

## Environment configuration

Settings load from `.env` at the repo root. Minimum:

```ini
# .env
LLM_PROVIDER=nebius
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct
LLM_TEMPERATURE=0.2

NEBIUS_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=ls-...           # optional, for tracing
LANGSMITH_PROJECT=all-agentic-architectures
```

To switch globally, change `LLM_PROVIDER` + the corresponding key. No code changes needed.

### MiniMax endpoints and models

`MINIMAX_BASE_URL` selects both the region and protocol. The factory uses the
Anthropic client when the URL ends in `/anthropic`; all other values use the
OpenAI client.

| Region | Protocol | `MINIMAX_BASE_URL` |
|---|---|---|
| Global | OpenAI | `https://api.minimax.io/v1` |
| Global | Anthropic | `https://api.minimax.io/anthropic` |
| China | OpenAI | `https://api.minimaxi.com/v1` |
| China | Anthropic | `https://api.minimaxi.com/anthropic` |

Keep the Anthropic base URL ending at `/anthropic`; the client appends
`/v1/messages` when it sends a request.

When MiniMax is selected and `LLM_MODEL` is unset, `get_llm()` defaults to
`MiniMax-M3.1-Flash`. Explicit `model=` arguments and configured `LLM_MODEL`
values, including custom or legacy IDs, are passed through unchanged. If you
copied `.env.example`, replace its `LLM_MODEL` with `MiniMax-M3.1-Flash` or
remove that entry to use the MiniMax default.

| Model | Total context window | Input modalities | Thinking |
|---|---:|---|---|
| `MiniMax-M3.1-Flash` (default) | 1,000,000 tokens | Text, image, video | Always adaptive; `disabled` is rejected |
| `MiniMax-M3` | 1,000,000 tokens | Text, image, video | `adaptive` or `disabled`; defaults on through OpenAI and off through Anthropic |

Flash supports up to 524,288 output tokens. Its thinking effort accepts `low`,
`medium`, `high`, `xhigh`, and `max` (the default); `none` is rejected. Pass
`reasoning_effort="high"` to `get_llm()` for the OpenAI-compatible endpoint, or
`model_kwargs={"output_config": {"effort": "high"}}` for the Anthropic-compatible
endpoint. MiniMax-M3 does not tune thinking depth with effort. Flash does not
support `service_tier` or `reasoning_split=false`. Preserve complete assistant
thinking and tool-call content when supplying conversation history.

Pay-as-you-go standard prices below are in USD per million tokens. MiniMax-M3
priority pricing is 1.5 times standard pricing; Flash has no priority tier.

| Model / input length | Input | Output | Cache read |
|---|---:|---:|---:|
| `MiniMax-M3.1-Flash` | $0.15 | $0.60 | $0.015 |
| `MiniMax-M3`, up to 512k | $0.30 | $1.20 | $0.06 |
| `MiniMax-M3`, above 512k | $0.60 | $2.40 | $0.12 |

See the official [model guide](https://platform.minimax.io/docs/guides/models-intro),
[pricing](https://platform.minimax.io/docs/pricing/overview), and
[Anthropic compatibility reference](https://platform.minimax.io/docs/api-reference/text-anthropic-api)
for current limits and availability. Legacy IDs remain caller-controlled; their
continued availability depends on the endpoint.

## When you'd override per-architecture

Reasoning architectures (nb 19, 20, 21, 22) default to Qwen3-Thinking for higher quality on multi-step structured reasoning. If you want them on a cheaper model:

```python
from agentic_architectures.architectures import LATS
arch = LATS(llm=get_llm(model="meta-llama/Llama-3.3-70B-Instruct"))
```

Or the opposite — force a single architecture onto a stronger model:

```python
from agentic_architectures.architectures import ChainOfVerification
arch = ChainOfVerification(llm=get_llm(provider="openai", model="gpt-4o"))
```

## Capability matrix

| Feature | Required for | Providers known good |
|---|---|---|
| `with_structured_output(Schema)` | nb 04, 06, 09, 10, 13, 15, 17-35 | All listed above |
| `bind_tools([...])` | nb 02, 03, 05, 11 | All except some Ollama tool-uncapable models |
| `bind(temperature=...)` per-call override | nb 21 Self-Consistency | All |

If an architecture fails on a provider, the failure is usually a missing capability — see the notebook's §10 "Try other providers" cell for graceful-skip patterns.
