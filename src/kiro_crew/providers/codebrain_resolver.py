"""Codebrain-style direct provider registry and resolver.

This module deliberately owns *selection* only.  It has no ACP dependency and
never invokes a provider endpoint: callers receive a resolved CLI/profile and
are responsible for running it.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping

_PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ProviderDefinition:
    """One stored or virtual provider profile.

    ``env`` is intentionally excluded from ``repr`` so diagnostic output cannot
    accidentally disclose a configured API token.
    """

    id: str
    label: str
    type: str
    host: str
    models: tuple[str, ...] = ()
    base_url: str = ""
    token_env_var: str = ""
    env: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)
    is_virtual: bool = False


@dataclass(frozen=True)
class ResolvedProvider:
    """The provider/CLI/model combination selected for one direct session."""

    agent: str
    provider: ProviderDefinition | None
    provider_id: str | None
    model: str | None
    error: str | None = None
    model_corrected: bool = False


# This is intentionally a transport-agnostic subset of Codebrain's registry.
# API-backed templates become candidates only when the user stores a profile;
# native CLI templates become candidates when their executable is present.
PROVIDER_REGISTRY: tuple[ProviderDefinition, ...] = (
    ProviderDefinition(
        id="claude-oauth",
        label="Claude",
        type="anthropic-compat",
        host="claude",
        models=(
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
            "claude-opus-4-8",
        ),
        is_virtual=True,
    ),
    ProviderDefinition(
        id="codex-oauth",
        label="Codex",
        type="codex",
        host="codex",
        models=("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"),
        is_virtual=True,
    ),
    ProviderDefinition(
        id="gemini-cli",
        label="Gemini CLI",
        type="gemini-cli",
        host="gemini",
        models=("gemini-3.5-flash",),
        is_virtual=True,
    ),
    ProviderDefinition(
        id="anthropic",
        label="Anthropic",
        type="anthropic-compat",
        host="claude",
        models=(
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
            "claude-opus-4-8",
        ),
        base_url="https://api.anthropic.com",
        token_env_var="ANTHROPIC_API_KEY",
    ),
    ProviderDefinition(
        id="codex",
        label="OpenAI Codex",
        type="codex",
        host="codex",
        models=("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"),
        token_env_var="OPENAI_API_KEY",
    ),
    ProviderDefinition(
        id="gemini",
        label="Google Gemini",
        type="gemini-compat",
        host="gemini",
        models=("gemini-3.5-flash", "gemini-3.1-pro-preview"),
        base_url="https://generativelanguage.googleapis.com/v1beta",
        token_env_var="GEMINI_API_KEY",
    ),
    ProviderDefinition(
        id="mimo-claude",
        label="MIMO via Claude",
        type="anthropic-compat",
        host="claude",
        models=("mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro", "mimo-v2-flash"),
        base_url="https://token-plan-ams.xiaomimimo.com/anthropic",
        token_env_var="ANTHROPIC_AUTH_TOKEN",
    ),
)

_TEMPLATE_BY_ID = {provider.id: provider for provider in PROVIDER_REGISTRY}
_NATIVE_PROVIDER_BY_AGENT = {
    "claude": "claude-oauth",
    "codex": "codex-oauth",
    "gemini": "gemini-cli",
    "gemini-cli": "gemini-cli",
}


def _normalise_models(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    models: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        model = item.strip()
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return tuple(models)


def _normalise_env(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str)
        and _ENV_KEY_RE.fullmatch(key)
        and isinstance(item, str)
    }


def _definition_from_mapping(raw: object) -> ProviderDefinition | None:
    if not isinstance(raw, dict):
        return None
    provider_id = raw.get("id")
    if not isinstance(provider_id, str) or not _PROVIDER_ID_RE.fullmatch(provider_id):
        return None
    template = _TEMPLATE_BY_ID.get(provider_id)
    label = raw.get("label") if isinstance(raw.get("label"), str) else ""
    provider_type = raw.get("type") if isinstance(raw.get("type"), str) else ""
    host = raw.get("host") if isinstance(raw.get("host"), str) else ""
    base_url = raw.get("baseUrl") if isinstance(raw.get("baseUrl"), str) else ""
    token_env_var = raw.get("tokenEnvVar") if isinstance(raw.get("tokenEnvVar"), str) else ""
    models = _normalise_models(raw.get("models"))
    if template is not None:
        label = label.strip() or template.label
        provider_type = provider_type.strip() or template.type
        host = host.strip() or template.host
        base_url = base_url.strip() or template.base_url
        token_env_var = token_env_var.strip() or template.token_env_var
        models = models or template.models
    if not (label.strip() and provider_type.strip() and host.strip()):
        return None
    return ProviderDefinition(
        id=provider_id,
        label=label.strip(),
        type=provider_type.strip(),
        host=host.strip(),
        models=models,
        base_url=base_url.strip(),
        token_env_var=token_env_var.strip(),
        env=_normalise_env(raw.get("env")),
    )


def _default_store_path() -> Path:
    # Lazy import keeps this module usable by small unit tests without booting
    # configuration machinery, while ensuring the store belongs to KiroCrew,
    # not to a neighboring Codebrain install.
    from kiro_crew.config.paths import config_dir

    return config_dir() / "providers.json"


class ProviderStore:
    """JSON-backed provider profile store owned by KiroCrew.

    The resolver never writes this store.  ``save`` exists for the future UI/API
    surface and uses a caller-supplied record list, so selecting a provider is
    always read-only.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else _default_store_path()

    def load(self) -> list[ProviderDefinition]:
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError):
            return []
        raw_providers = parsed.get("providers") if isinstance(parsed, dict) else parsed
        if not isinstance(raw_providers, list):
            return []
        providers: list[ProviderDefinition] = []
        seen: set[str] = set()
        for raw in raw_providers:
            provider = _definition_from_mapping(raw)
            if provider is not None and provider.id not in seen:
                seen.add(provider.id)
                providers.append(provider)
        return providers

    def save(self, providers: Iterable[ProviderDefinition]) -> None:
        rows = []
        for provider in providers:
            rows.append(
                {
                    "id": provider.id,
                    "label": provider.label,
                    "type": provider.type,
                    "host": provider.host,
                    "models": list(provider.models),
                    "baseUrl": provider.base_url,
                    "tokenEnvVar": provider.token_env_var,
                    "env": dict(provider.env),
                }
            )
        from kiro_crew.atomic_write import atomic_write

        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(self.path, json.dumps({"providers": rows}, indent=2) + "\n", mode=0o600)


class ProviderResolver:
    """Resolve direct-provider requests with Codebrain's native-CLI precedence."""

    def __init__(
        self,
        store: ProviderStore | None = None,
        *,
        cli_lookup: Callable[[str], str | None] = shutil.which,
    ):
        self._store = store or ProviderStore()
        self._cli_lookup = cli_lookup

    def get_enhanced_providers(self) -> list[ProviderDefinition]:
        """Return detected virtual profiles followed by persisted profiles."""
        stored = self._store.load()
        stored_by_id = {provider.id: provider for provider in stored}
        providers: list[ProviderDefinition] = []
        emitted: set[str] = set()
        for template in PROVIDER_REGISTRY:
            if not template.is_virtual or not self._cli_lookup(template.host):
                continue
            saved = stored_by_id.get(template.id)
            provider = replace(saved, is_virtual=True) if saved is not None else template
            providers.append(replace(provider, is_virtual=True))
            emitted.add(provider.id)
        for provider in stored:
            if provider.id not in emitted:
                providers.append(provider)
                emitted.add(provider.id)
        return providers

    @staticmethod
    def _model_family(model: str) -> str:
        lowered = model.lower()
        if lowered.startswith(("gpt-", "o1", "o3", "o4")):
            return "codex"
        if lowered.startswith("claude-"):
            return "anthropic-compat"
        if lowered.startswith("gemini-"):
            return "gemini"
        return ""

    @staticmethod
    def _compatible_model(provider: ProviderDefinition, requested: str | None) -> tuple[str | None, bool]:
        requested = (requested or "").strip()
        if requested and requested in provider.models:
            return requested, False
        if provider.models:
            return provider.models[0], bool(requested)
        return requested or None, False

    def resolve(
        self,
        agent: str | None = None,
        provider_id: str | None = None,
        model: str | None = None,
    ) -> ResolvedProvider:
        requested_agent = (agent or "").strip().lower()
        requested_provider = (provider_id or "").strip()
        requested_model = (model or "").strip() or None
        if requested_agent == "shell":
            return ResolvedProvider(agent="shell", provider=None, provider_id=None, model=None)

        providers = self.get_enhanced_providers()
        provider: ProviderDefinition | None = None

        if requested_provider:
            provider = next((item for item in providers if item.id == requested_provider), None)
            if provider is None:
                available = ", ".join(item.id for item in providers) or "none"
                return ResolvedProvider(
                    agent=requested_agent or "direct",
                    provider=None,
                    provider_id=None,
                    model=requested_model,
                    error=f'Provider "{requested_provider}" is not configured or detected (available: {available}).',
                )
        else:
            native_id = _NATIVE_PROVIDER_BY_AGENT.get(requested_agent)
            if native_id:
                provider = next((item for item in providers if item.id == native_id), None)
                if provider is None:
                    return ResolvedProvider(
                        agent=requested_agent,
                        provider=None,
                        provider_id=None,
                        model=requested_model,
                        error=f'Native CLI "{requested_agent}" is not installed or configured.',
                    )
            if provider is None and requested_model:
                family = self._model_family(requested_model)
                provider = next((item for item in providers if requested_model in item.models), None)
                if provider is None and family:
                    provider = next(
                        (
                            item
                            for item in providers
                            if item.type == family
                            or (family == "gemini" and item.type.startswith("gemini"))
                        ),
                        None,
                    )
            if provider is None and requested_agent:
                provider = next((item for item in providers if item.host == requested_agent), None)
            if provider is None and providers:
                provider = providers[0]

        if provider is None:
            return ResolvedProvider(
                agent=requested_agent or "direct",
                provider=None,
                provider_id=None,
                model=requested_model,
                error="No configured or detected direct provider is available.",
            )

        resolved_model, corrected = self._compatible_model(provider, requested_model)
        return ResolvedProvider(
            agent=provider.host,
            provider=provider,
            provider_id=provider.id,
            model=resolved_model,
            model_corrected=corrected,
        )


_resolver: ProviderResolver | None = None


def get_resolver() -> ProviderResolver:
    global _resolver
    if _resolver is None:
        _resolver = ProviderResolver()
    return _resolver


def resolve_provider(
    agent: str | None = None,
    provider_id: str | None = None,
    model: str | None = None,
) -> ResolvedProvider:
    """Resolve a direct provider using the process-level KiroCrew store."""

    return get_resolver().resolve(agent=agent, provider_id=provider_id, model=model)
