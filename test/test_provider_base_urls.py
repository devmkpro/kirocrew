"""``agent.provider_base_urls``: the public endpoint map beside the credential one.

Why a separate field exists at all, and so why these tests: ``agent.deepseek_env``
carries a provider key as a ``secret://`` reference, and its validator REQUIRES
each name to match the harness's child-scrub class (``KEY|PASSWORD|SECRET|TOKEN``)
— the property that proves the harness withholds the name from the shells it
spawns. A base URL satisfies neither that class nor the ``secret://`` value rule,
so an OpenAI-compatible endpoint had nowhere to go.

The pair therefore splits by SENSITIVITY, and each field's validator enforces the
invariant the other one cannot: the credential field keeps a key out of
``config.json`` and out of the model's bash tool, and the public field — these
tests — keeps a credential from being smuggled through a mapping whose values ARE
forwarded to those same shells. The refusals below are all of the second kind, so
each one is a hole the pair would otherwise have.
"""

from __future__ import annotations

import pytest

from kiro_crew.acp.client import _validate_provider_base_url_mapping
from kiro_crew.config import sections


class TestCoercion:
    """Type coercion only — the shape rules live at spawn, deliberately."""

    def test_non_dict_is_empty(self) -> None:
        for raw in (None, "", 0, [], ("OPENAI_BASE_URL", "https://x")):
            assert sections.coerce_provider_base_urls(raw) == {}

    def test_non_string_entries_dropped(self) -> None:
        assert sections.coerce_provider_base_urls(
            {"OPENAI_BASE_URL": "https://openrouter.ai/api/v1", "N": 1, 2: "https://x"}
        ) == {"OPENAI_BASE_URL": "https://openrouter.ai/api/v1"}

    def test_whitespace_is_not_stripped(self) -> None:
        """A padded name is not a POSIX identifier; spawn refuses it BY NAME.

        Repairing it here would inject a different variable than the operator
        wrote, silently — which is the failure mode the split exists to avoid.
        """
        assert sections.coerce_provider_base_urls({" OPENAI_BASE_URL ": "https://x.ai"}) == {
            " OPENAI_BASE_URL ": "https://x.ai"
        }

    def test_field_defaults_empty(self) -> None:
        assert sections.AgentConfig().provider_base_urls == {}


class TestAcceptance:
    def test_empty_mapping_accepted(self) -> None:
        _validate_provider_base_url_mapping({})

    def test_openai_compatible_endpoint_accepted(self) -> None:
        _validate_provider_base_url_mapping({"OPENAI_BASE_URL": "https://openrouter.ai/api/v1"})

    def test_several_providers_accepted(self) -> None:
        _validate_provider_base_url_mapping(
            {
                "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
            }
        )


class TestRefusesCredentialSmuggling:
    """The invariant the credential field cannot enforce from its side."""

    def test_secret_reference_refused(self) -> None:
        """A vault value here would be forwarded to the harness's shell children."""
        with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
            _validate_provider_base_url_mapping({"OPENAI_BASE_URL": "secret://my-key"})

    @pytest.mark.parametrize(
        "name",
        ["OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "MY_PASSWORD", "PROVIDER_SECRET", "lower_key"],
    )
    def test_credential_shaped_name_refused(self, name: str) -> None:
        """The entry that LOOKS like it works is the dangerous one.

        A name inside the harness's child-scrub class belongs in
        ``deepseek_env``, which withholds it from the model's bash tool. Accepted
        here it would reach every shell the harness spawns.
        """
        with pytest.raises(ValueError, match="credential-shaped"):
            _validate_provider_base_url_mapping({name: "https://x.ai"})

    def test_refusal_never_echoes_the_value(self) -> None:
        """Messages name the operator's KEY only — they reach logs and error cards."""
        secret = "secret://super-sensitive-vault-name"
        with pytest.raises(ValueError) as caught:
            _validate_provider_base_url_mapping({"OPENAI_BASE_URL": secret})
        assert secret not in str(caught.value)
        assert "super-sensitive-vault-name" not in str(caught.value)


class TestRefusesUnusableEndpoints:
    @pytest.mark.parametrize(
        "url",
        ["http://openrouter.ai/api/v1", "ftp://x.ai", "openrouter.ai", "", "//x.ai", "HTTPS://X.AI"],
    )
    def test_non_https_refused(self, url: str) -> None:
        """The key from ``deepseek_env`` travels here; plaintext would expose it."""
        with pytest.raises(ValueError, match="https://"):
            _validate_provider_base_url_mapping({"OPENAI_BASE_URL": url})

    @pytest.mark.parametrize("name", ["BAD NAME", "1LEADING_DIGIT", "has-dash", "", "WITH.DOT"])
    def test_non_posix_identifier_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="POSIX"):
            _validate_provider_base_url_mapping({name: "https://x.ai"})

    @pytest.mark.parametrize("name", ["DSH_BASE_URL", "KIROCREW_BASE_URL"])
    def test_reserved_namespace_refused(self, name: str) -> None:
        """Crew writes its own names onto this child AFTER this injection."""
        with pytest.raises(ValueError, match="namespaces"):
            _validate_provider_base_url_mapping({name: "https://x.ai"})


class TestRefusalOrdering:
    """A single entry breaking several rules still refuses, with ONE reason.

    Pinned because the arm turns any :exc:`ValueError` into a refused session: a
    rule reordering must not turn a refusal into an acceptance.
    """

    def test_credential_name_and_http_still_refuses(self) -> None:
        with pytest.raises(ValueError):
            _validate_provider_base_url_mapping({"OPENAI_API_KEY": "http://x.ai"})

    def test_secret_value_with_bad_name_still_refuses(self) -> None:
        with pytest.raises(ValueError):
            _validate_provider_base_url_mapping({"bad name": "secret://k"})
