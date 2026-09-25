"""Dashboard API for the Codebrain-style provider store.

The Settings > Codebrain panel is the UI over
``kiro_crew.providers.codebrain_resolver``: the registry of known profiles, the
native CLIs actually present on this machine, and the user's own stored profiles
(OpenRouter, DeepSeek, a team gateway).

Tokens are WRITE-ONLY across this boundary.  A GET reports whether a profile
carries a token, never the token, and a POST that omits one keeps the stored
value -- so re-saving a profile from the UI cannot blank a key the user cannot
see, and a read-only dashboard session cannot exfiltrate one.
"""

from __future__ import annotations

import shutil
from typing import Any

from aiohttp import web

from kiro_crew.providers.codebrain_resolver import (
    PROVIDER_REGISTRY,
    ProviderDefinition,
    ProviderStore,
    _definition_from_mapping,
)

#: The env key a profile's token is stored under when the profile itself names
#: none. Mirrors the resolver's own fallback order.
_DEFAULT_TOKEN_KEYS = ("ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "GEMINI_API_KEY")


def _token_key(provider: ProviderDefinition) -> str:
    if provider.token_env_var:
        return provider.token_env_var
    for key in _DEFAULT_TOKEN_KEYS:
        if key in provider.env:
            return key
    return ""


def _public_view(provider: ProviderDefinition, *, detected: bool) -> dict[str, Any]:
    """One profile as the dashboard may see it -- no secret material."""
    token_key = _token_key(provider)
    return {
        "id": provider.id,
        "label": provider.label,
        "type": provider.type,
        "host": provider.host,
        "models": list(provider.models),
        "baseUrl": provider.base_url,
        "tokenEnvVar": provider.token_env_var,
        "isVirtual": provider.is_virtual,
        "detected": detected,
        # Presence only. The value never crosses this boundary.
        "hasToken": bool(token_key and provider.env.get(token_key, "").strip()),
    }


async def api_codebrain_providers(request: web.Request) -> web.Response:
    """GET /api/codebrain/providers — registry, detected CLIs, stored profiles."""
    store = ProviderStore()
    stored = store.load()
    detected_hosts = {
        template.host for template in PROVIDER_REGISTRY if shutil.which(template.host)
    }
    detected_hosts.update(
        provider.host for provider in stored if shutil.which(provider.host)
    )
    return web.json_response(
        {
            "storePath": str(store.path),
            "detectedHosts": sorted(detected_hosts),
            "registry": [
                _public_view(template, detected=template.host in detected_hosts)
                for template in PROVIDER_REGISTRY
            ],
            "providers": [
                _public_view(provider, detected=provider.host in detected_hosts)
                for provider in stored
            ],
        }
    )


async def api_codebrain_providers_save(request: web.Request) -> web.Response:
    """POST /api/codebrain/providers — replace the stored profile list.

    The body is ``{"providers": [...]}`` in the same shape the GET returns, plus
    an optional ``token`` per profile. A profile that sends no ``token`` keeps
    whatever the store already holds for that id.
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    rows = body.get("providers") if isinstance(body, dict) else None
    if not isinstance(rows, list):
        return web.json_response({"error": "providers must be a list"}, status=400)

    store = ProviderStore()
    existing = {provider.id: provider for provider in store.load()}

    accepted: list[ProviderDefinition] = []
    rejected: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            rejected.append({"index": str(index), "reason": "not an object"})
            continue
        # A blank token means "unchanged", so the incoming env starts from what
        # is already stored and only a non-empty token replaces it.
        previous = existing.get(row.get("id", ""))
        env = dict(previous.env) if previous is not None else {}
        token = row.get("token")
        candidate = _definition_from_mapping({**row, "env": env})
        if candidate is None:
            rejected.append(
                {"index": str(index), "reason": "id, label, type and host are required"}
            )
            continue
        if isinstance(token, str) and token.strip():
            key = _token_key(candidate) or "OPENAI_API_KEY"
            env = {**env, key: token.strip()}
            candidate = ProviderDefinition(
                id=candidate.id,
                label=candidate.label,
                type=candidate.type,
                host=candidate.host,
                models=candidate.models,
                base_url=candidate.base_url,
                token_env_var=candidate.token_env_var or key,
                env=env,
            )
        accepted.append(candidate)

    if rejected:
        return web.json_response({"error": "invalid providers", "rejected": rejected}, status=400)

    store.save(accepted)
    return web.json_response({"ok": True, "saved": len(accepted)})


__all__ = ["api_codebrain_providers", "api_codebrain_providers_save"]
