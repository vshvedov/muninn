"""Factories for the local Ollama provider/model and the two agents."""
from __future__ import annotations

import httpx
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import Tool


# Per-level addendum appended to the muninn system prompt. The bias text
# is short on purpose: local Qwen3-coder responds better to a few crisp
# rules than a long manifesto. The headings carry the level name in caps
# so the model can self-reference ("at HIGH level I should...").
_LEVEL_ADDENDA: dict[str, str] = {
    "low": (
        "\n\n# Freedom level: LOW\n"
        "When you face genuine ambiguity, prefer ask_user with concrete\n"
        "options over deciding silently. The user wants to be in the loop\n"
        "on routine decisions; pause and ask."
    ),
    "medium": (
        "\n\n# Freedom level: MEDIUM\n"
        "Decide routine ambiguity yourself when the project files give a\n"
        "clear answer. Reserve ask_user for real forks the project cannot\n"
        "resolve (naming, scope, behavior the user has not stated). Do\n"
        "not call ask_user for facts you can verify by reading more files."
    ),
    "high": (
        "\n\n# Freedom level: HIGH\n"
        "You operate autonomously. Decide, act, verify, iterate. Call\n"
        "ask_user ONLY when (a) a destructive choice has no\n"
        "project-derivable answer, or (b) verification has failed twice\n"
        "and you cannot find the cause. When work is incomplete, keep\n"
        "going: read more, edit more, run more tests. Never stop after\n"
        "only announcing intent."
    ),
}


def compose_muninn_prompt(base: str, level: str) -> str:
    """Append the per-level addendum to the bundled muninn system prompt.

    Unknown level falls back to the LOW addendum (the safest bias). The
    base text is unchanged; the addendum is concatenated so existing
    user-level / project-level prompt overrides stack cleanly.
    """
    return base + _LEVEL_ADDENDA.get(level, _LEVEL_ADDENDA["low"])


def make_provider(base_url: str, http_client: httpx.AsyncClient) -> OllamaProvider:
    return OllamaProvider(base_url=base_url, http_client=http_client)


def make_local_model(model_id: str, provider: OllamaProvider) -> OllamaModel:
    return OllamaModel(model_id, provider=provider)


def num_ctx_settings(num_ctx: int) -> ModelSettings:
    """Pass num_ctx to Ollama via the OpenAI-compat layer's extra_body.

    Ollama's OpenAI-compatible endpoint accepts a top-level `options` object
    that maps to native model parameters (num_ctx, num_predict, etc.).
    pydantic-ai forwards `extra_body` straight into the OpenAI request payload.
    """
    return ModelSettings(extra_body={"options": {"num_ctx": num_ctx}})


def muninn_agent(model: OllamaModel, tools: list[Tool], system_prompt: str) -> Agent:
    return Agent(model, system_prompt=system_prompt, tools=tools)


def huginn_agent(model: OllamaModel, system_prompt: str) -> Agent:
    """Stateless cold-reader; NO tools."""
    return Agent(model, system_prompt=system_prompt)
