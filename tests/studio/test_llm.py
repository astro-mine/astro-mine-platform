"""STUDIO-05 — optional, provider-abstracted LLM intent capture (no live model in CI)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from astro_mine.studio.intent import MetricVocabulary, ObjectiveGateError
from astro_mine.studio.intent.llm import (
    AnthropicProvider,
    LLMConfig,
    LLMDraftResult,
    LLMObjectiveDraft,
    LLMProvider,
    LLMUsage,
    accept_draft,
    draft_objective,
    objective_draft_schema,
)
from astro_mine.studio.workspace import InMemoryWorkspace

_VALID: dict[str, Any] = {
    "objective_version": "0.1",
    "objective": {
        "id": "ice",
        "name": "Lunar ice",
        "success_criteria": [
            {
                "id": "water",
                "binding": {
                    "metric": "water_rate",
                    "unit": "kg/day",
                    "direction": "higher_better",
                    "target": 40.0,
                    "tolerance": 10.0,
                },
            }
        ],
    },
}


class _StubProvider:
    """A recorded-transcript provider — the mock the CI eval set runs against."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.models: list[str] = []

    def draft(
        self, intent_text: str, *, schema: Any, system: str, model: str, config: LLMConfig
    ) -> LLMDraftResult:
        self.models.append(model)
        return LLMDraftResult(
            payload=dict(self.payload), model=model, usage=LLMUsage(input_tokens=10)
        )


def test_stub_provider_satisfies_the_protocol() -> None:
    assert isinstance(_StubProvider(_VALID), LLMProvider)


def test_objective_draft_schema_is_constraint_free() -> None:
    schema = objective_draft_schema()
    assert schema["required"] == ["objective_version", "objective"]
    text = json.dumps(schema)
    assert "minimum" not in text and "minLength" not in text  # structured outputs reject these


def test_draft_objective_validates_without_persisting() -> None:
    provider = _StubProvider(_VALID)
    draft = draft_objective(
        "Produce 40 kg/day of water",
        provider=provider,
        vocabulary=MetricVocabulary(metrics={"water_rate": ""}),
    )
    assert isinstance(draft, LLMObjectiveDraft)
    assert draft.document.objective.id == "ice"
    assert draft.model == LLMConfig().synthesis_model == provider.models[0]


def test_draft_objective_rejects_unknown_metric() -> None:
    with pytest.raises(ObjectiveGateError):
        draft_objective(
            "x", provider=_StubProvider(_VALID), vocabulary=MetricVocabulary(metrics={"other": ""})
        )


def test_draft_objective_rejects_malformed_model_output() -> None:
    with pytest.raises(ObjectiveGateError, match="not a valid ObjectiveSpec"):
        draft_objective(
            "x", provider=_StubProvider({"objective_version": "0.1", "objective": {"id": "x"}})
        )


def test_accept_draft_persists_and_records_model() -> None:
    draft = draft_objective("Produce water", provider=_StubProvider(_VALID))
    ws = InMemoryWorkspace()
    captured = accept_draft(draft, workspace=ws, author="designer")
    assert ws.has(captured.digest)
    entry = ws.audit()[0]
    assert entry.model == LLMConfig().synthesis_model and entry.author == "designer"


# --- AnthropicProvider request construction (fake client — never a live model) --------- #


def _client(create: Any) -> Any:
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def _response(payload: dict[str, Any], *, model: str, with_usage: bool = True) -> Any:
    usage = (
        SimpleNamespace(input_tokens=120, output_tokens=40, cache_read_input_tokens=100)
        if with_usage
        else None
    )
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking"),
            SimpleNamespace(type="text", text=json.dumps(payload)),
        ],
        usage=usage,
        model=model,
    )


def test_anthropic_provider_builds_structured_cached_thinking_request() -> None:
    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _response(_VALID, model=kwargs["model"])

    provider = AnthropicProvider(client=_client(create))
    result = provider.draft(
        "intent text",
        schema=objective_draft_schema(),
        system="SYS",
        model="claude-opus-4-8",
        config=LLMConfig(),
    )
    assert result.payload == _VALID and result.model == "claude-opus-4-8"
    assert result.usage.input_tokens == 120 and result.usage.cache_read_tokens == 100
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["output_config"]["format"] == {
        "type": "json_schema",
        "schema": objective_draft_schema(),
    }
    assert captured["output_config"]["effort"] == "high"
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured["messages"] == [{"role": "user", "content": "intent text"}]


def test_anthropic_provider_honors_disabled_caching_and_thinking() -> None:
    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _response(_VALID, model=kwargs["model"], with_usage=False)

    provider = AnthropicProvider(client=_client(create))
    result = provider.draft(
        "intent",
        schema={},
        system="SYS",
        model="m",
        config=LLMConfig(prompt_caching=False, adaptive_thinking=False),
    )
    assert captured["system"] == "SYS"  # plain string, no cache_control breakpoint
    assert "thinking" not in captured
    assert result.usage.input_tokens == 0  # no usage on the response → zeros


def test_anthropic_provider_requires_a_text_block() -> None:
    def create(**kwargs: Any) -> Any:
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking")], usage=None, model=kwargs["model"]
        )

    with pytest.raises(ObjectiveGateError, match="no text block"):
        AnthropicProvider(client=_client(create)).draft(
            "i", schema={}, system="s", model="m", config=LLMConfig()
        )
