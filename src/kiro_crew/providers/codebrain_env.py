"""Codebrain-style provider environment injection.

Codebrain does not talk to provider endpoints itself: it launches the vendor CLI
and *routes* it with environment variables, so one CLI can serve many endpoints
(``pane-spawn.ts``, the ``isAnthropicCompat`` / ``isMimo`` / ``isGeminiCompat`` /
``isOpenAICompat`` branches).  This module is the Python spelling of that routing
table.

It is deliberately pure: it takes a resolved provider plus the ambient
environment and RETURNS the overlay to apply.  Nothing here spawns a process,
reads a config file, or mutates ``os.environ``, so each routing rule is unit
testable without a CLI on PATH.

Not ported: OpenRouter's cross-provider model rewriting.  That branch rewrites
``argv`` as well as the environment and is specific to one vendor's model-id
namespace; porting it blind would produce rules no test here could justify.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from .codebrain_resolver import ProviderDefinition

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: ``/anthropic`` endpoints expose an OpenAI-compatible sibling at ``/v1``.
_ANTHROPIC_SUFFIX_RE = re.compile(r"/anthropic/?$")

#: Every Claude-CLI model default, so a NON-Anthropic model served over an
#: Anthropic-compatible endpoint (MIMO, DeepSeek, Fireworks) is not silently
#: replaced by the CLI's built-in ``claude-*`` fallback.
_ANTHROPIC_MODEL_DEFAULTS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
)

TYPE_ANTHROPIC_COMPAT = "anthropic-compat"
TYPE_MIMO_COMPAT = "mimo-compat"
TYPE_GEMINI_COMPAT = "gemini-compat"
TYPE_GEMINI_CLI = "gemini-cli"
TYPE_OPENAI_COMPAT = "openai-compat"
TYPE_CODEX = "codex"
TYPE_OAUTH = "oauth"


def openai_sibling_base_url(base_url: str) -> str:
    """The OpenAI-compatible sibling of an ``/anthropic`` base URL."""
    return _ANTHROPIC_SUFFIX_RE.sub("/v1", base_url)


def _token(provider: ProviderDefinition, ambient: Mapping[str, str], *names: str) -> str:
    """First non-empty token for *provider*, profile env before ambient env.

    A stored profile is more specific than the shell the gateway happens to run
    in, so it wins; the ambient fallback is what lets a user export a key once
    instead of storing it per profile.
    """
    candidates = [provider.token_env_var, *names]
    for source in (provider.env, ambient):
        for name in candidates:
            if not name:
                continue
            value = source.get(name, "")
            if isinstance(value, str) and value.strip():
                return value
    return ""


def build_provider_env(
    provider: ProviderDefinition,
    model: str | None = None,
    *,
    ambient: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment overlay routing a vendor CLI at *provider*.

    Returns only the keys this routing decides, so a caller can layer it over
    ``os.environ`` and see exactly what the provider changed.
    """
    ambient = ambient or {}
    model = (model or "").strip()
    overlay: dict[str, str] = {}

    # Diagnostic markers Codebrain sets for every provider-routed CLI. They are
    # what makes "which profile is this pane on?" answerable from inside the
    # child process rather than only from the spawning side.
    if provider.label:
        overlay["CLAUDE_CODE_PROVIDER_NAME"] = provider.label
    if model:
        overlay["CLAUDE_CODE_MODEL_NAME"] = model
    overlay["CLAUDE_CODE_PROVIDER_PROFILE_ENV_APPLIED"] = "1"

    provider_type = provider.type
    base_url = provider.base_url.strip()

    if provider_type in (TYPE_ANTHROPIC_COMPAT, TYPE_OAUTH, TYPE_MIMO_COMPAT):
        token = _token(provider, ambient, "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "MIMO_API_KEY")
        if base_url:
            overlay["ANTHROPIC_BASE_URL"] = base_url
        if token:
            # The Claude CLI reads ANTHROPIC_AUTH_TOKEN as a bearer token, which
            # is what allows a non-Anthropic endpoint to serve it at all.
            overlay["ANTHROPIC_AUTH_TOKEN"] = token
        if model:
            overlay["MODEL"] = model
            for key in _ANTHROPIC_MODEL_DEFAULTS:
                overlay[key] = model

        if provider_type == TYPE_MIMO_COMPAT:
            sibling = openai_sibling_base_url(base_url or overlay.get("ANTHROPIC_BASE_URL", ""))
            if sibling:
                overlay["OPENAI_BASE_URL"] = sibling
            if token:
                overlay["MIMO_API_KEY"] = token
                overlay["OPENAI_API_KEY"] = token
            if model:
                overlay["OPENAI_MODEL"] = model

    elif provider_type in (TYPE_GEMINI_COMPAT, TYPE_GEMINI_CLI):
        overlay["CLAUDE_CODE_USE_GEMINI"] = "1"
        token = _token(provider, ambient, "GEMINI_API_KEY", "GOOGLE_API_KEY")
        if token:
            # The CLI reads whichever of these it was built against; setting all
            # three is Codebrain's own answer to that ambiguity.
            overlay["GEMINI_API_KEY"] = token
            overlay["GOOGLE_API_KEY"] = token
        if base_url:
            overlay["GEMINI_BASE_URL"] = base_url
            overlay["CLAUDE_CODE_DISABLE_PROXY"] = "1"
        if model:
            overlay["GEMINI_MODEL"] = model
            overlay["MODEL"] = model

    elif provider_type in (TYPE_OPENAI_COMPAT, TYPE_CODEX):
        token = _token(provider, ambient, "OPENAI_API_KEY")
        if base_url:
            overlay["OPENAI_BASE_URL"] = base_url
        if token:
            overlay["OPENAI_API_KEY"] = token
            lowered = base_url.lower()
            # Vendors that speak the OpenAI protocol but read their own key name.
            if "x.ai" in lowered:
                overlay["XAI_API_KEY"] = token
            if "deepseek" in lowered:
                overlay["DEEPSEEK_API_KEY"] = token
        if model:
            overlay["MODEL"] = model
            overlay["OPENAI_MODEL"] = model

    return {key: value for key, value in overlay.items() if _ENV_KEY_RE.fullmatch(key)}


def provider_cli_flag(provider: ProviderDefinition, *, agent: str | None = None) -> list[str]:
    """The ``--provider`` argv fragment for *provider*, empty when env suffices.

    Anthropic-compatible endpoints route purely by environment, and the native
    Claude CLI has no ``--provider`` flag at all -- passing one there is an
    immediate argv error rather than a fallback.
    """
    if (agent or "").strip().lower() == "claude":
        return []
    mapping = {
        TYPE_OPENAI_COMPAT: "openai",
        TYPE_GEMINI_COMPAT: "gemini",
    }
    flag = mapping.get(provider.type, "")
    return ["--provider", flag] if flag else []


__all__ = ["build_provider_env", "openai_sibling_base_url", "provider_cli_flag"]
