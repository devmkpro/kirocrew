"""Tests for Codebrain-style provider environment routing."""

from __future__ import annotations

from pathlib import Path

from kiro_crew.providers.codebrain import CodebrainProvider
from kiro_crew.providers.codebrain_env import (
    build_provider_env,
    openai_sibling_base_url,
    provider_cli_flag,
)
from kiro_crew.providers.codebrain_resolver import (
    ProviderDefinition,
    ProviderResolver,
    ProviderStore,
)


def test_anthropic_compat_overrides_every_claude_model_default() -> None:
    provider = ProviderDefinition(
        id="mimo-claude",
        label="MIMO via Claude",
        type="anthropic-compat",
        host="claude",
        models=("mimo-v2.5-pro",),
        base_url="https://token-plan-ams.example.com/anthropic",
        token_env_var="ANTHROPIC_AUTH_TOKEN",
        env={"ANTHROPIC_AUTH_TOKEN": "profile-token"},
    )

    overlay = build_provider_env(provider, "mimo-v2.5-pro")

    assert overlay["ANTHROPIC_BASE_URL"] == "https://token-plan-ams.example.com/anthropic"
    assert overlay["ANTHROPIC_AUTH_TOKEN"] == "profile-token"
    # A non-Anthropic model must displace every built-in claude-* default, or the
    # CLI answers "model not found" and silently falls back.
    for key in (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
    ):
        assert overlay[key] == "mimo-v2.5-pro"


def test_mimo_compat_derives_the_openai_sibling_endpoint() -> None:
    provider = ProviderDefinition(
        id="mimo",
        label="MIMO",
        type="mimo-compat",
        host="openclaude",
        models=("mimo-v2.5",),
        base_url="https://token-plan-ams.example.com/anthropic",
        token_env_var="MIMO_API_KEY",
        env={"MIMO_API_KEY": "k"},
    )

    overlay = build_provider_env(provider, "mimo-v2.5")

    assert overlay["OPENAI_BASE_URL"] == "https://token-plan-ams.example.com/v1"
    assert overlay["OPENAI_API_KEY"] == "k"
    assert overlay["MIMO_API_KEY"] == "k"


def test_profile_token_wins_over_the_ambient_shell() -> None:
    provider = ProviderDefinition(
        id="anthropic",
        label="Anthropic",
        type="anthropic-compat",
        host="claude",
        base_url="https://api.anthropic.com",
        token_env_var="ANTHROPIC_API_KEY",
        env={"ANTHROPIC_API_KEY": "from-profile"},
    )

    overlay = build_provider_env(provider, None, ambient={"ANTHROPIC_API_KEY": "from-shell"})

    assert overlay["ANTHROPIC_AUTH_TOKEN"] == "from-profile"


def test_ambient_token_is_the_fallback_when_the_profile_stores_none() -> None:
    provider = ProviderDefinition(
        id="anthropic",
        label="Anthropic",
        type="anthropic-compat",
        host="claude",
        base_url="https://api.anthropic.com",
        token_env_var="ANTHROPIC_API_KEY",
    )

    overlay = build_provider_env(provider, None, ambient={"ANTHROPIC_API_KEY": "from-shell"})

    assert overlay["ANTHROPIC_AUTH_TOKEN"] == "from-shell"


def test_gemini_sets_every_key_name_the_cli_may_read() -> None:
    provider = ProviderDefinition(
        id="gemini",
        label="Gemini",
        type="gemini-compat",
        host="gemini",
        base_url="https://generativelanguage.example.com/v1beta",
        token_env_var="GEMINI_API_KEY",
        env={"GEMINI_API_KEY": "g"},
    )

    overlay = build_provider_env(provider, "gemini-3.5-flash")

    assert overlay["CLAUDE_CODE_USE_GEMINI"] == "1"
    assert overlay["GEMINI_API_KEY"] == "g"
    assert overlay["GOOGLE_API_KEY"] == "g"
    assert overlay["CLAUDE_CODE_DISABLE_PROXY"] == "1"
    assert overlay["GEMINI_MODEL"] == "gemini-3.5-flash"


def test_openai_compat_aliases_the_vendor_key_by_base_url() -> None:
    provider = ProviderDefinition(
        id="deepseek",
        label="DeepSeek",
        type="openai-compat",
        host="openclaude",
        base_url="https://api.deepseek.com/v1",
        token_env_var="OPENAI_API_KEY",
        env={"OPENAI_API_KEY": "d"},
    )

    overlay = build_provider_env(provider, "deepseek-chat")

    assert overlay["OPENAI_BASE_URL"] == "https://api.deepseek.com/v1"
    assert overlay["DEEPSEEK_API_KEY"] == "d"
    assert "XAI_API_KEY" not in overlay


def test_oauth_profile_injects_no_endpoint_or_token() -> None:
    """A CLI already logged in must not be handed an empty base URL or key."""
    provider = ProviderDefinition(
        id="codex-oauth",
        label="Codex",
        type="codex",
        host="codex",
        models=("gpt-5.6-terra",),
        is_virtual=True,
    )

    overlay = build_provider_env(provider, "gpt-5.6-terra")

    assert "OPENAI_BASE_URL" not in overlay
    assert "OPENAI_API_KEY" not in overlay
    assert overlay["OPENAI_MODEL"] == "gpt-5.6-terra"


def test_openai_sibling_leaves_a_non_anthropic_url_untouched() -> None:
    assert openai_sibling_base_url("https://example.com/v1") == "https://example.com/v1"
    assert openai_sibling_base_url("https://example.com/anthropic/") == "https://example.com/v1"


def test_native_claude_cli_never_receives_a_provider_flag() -> None:
    provider = ProviderDefinition(
        id="p", label="P", type="openai-compat", host="claude", base_url="https://x/v1"
    )

    assert provider_cli_flag(provider, agent="claude") == []
    assert provider_cli_flag(provider) == ["--provider", "openai"]


def test_provider_environment_reaches_the_spawned_cli(
    tmp_path: Path, monkeypatch
) -> None:
    """End to end: a stored profile routes the child process, extra_env wins."""
    store_path = tmp_path / "providers.json"
    store = ProviderStore(store_path)
    store.save(
        [
            ProviderDefinition(
                id="team-codex",
                label="Team Codex",
                type="codex",
                host="codex",
                models=("team-gpt",),
                base_url="https://gateway.example.com/v1",
                token_env_var="OPENAI_API_KEY",
                env={"OPENAI_API_KEY": "team-key"},
            )
        ]
    )
    monkeypatch.setattr("kiro_crew.providers.codebrain.shutil.which", lambda _: "/fake/codex")

    provider = CodebrainProvider(
        work_dir=tmp_path,
        provider_id="team-codex",
        extra_env={"OPENAI_BASE_URL": "https://override.example.com/v1"},
        resolver=ProviderResolver(store, cli_lookup=lambda host: None),
    )
    import asyncio

    asyncio.run(provider.start())
    environment = provider._environment()

    assert environment["OPENAI_API_KEY"] == "team-key"
    assert environment["CLAUDE_CODE_PROVIDER_NAME"] == "Team Codex"
    assert environment["OPENAI_MODEL"] == "team-gpt"
    # extra_env is the top layer: a session override survives the profile.
    assert environment["OPENAI_BASE_URL"] == "https://override.example.com/v1"
