"""Focused tests for the Codebrain-style direct provider seam."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest

from kiro_crew.acp.types import EVENT_COMPLETE, EVENT_TEXT_CHUNK
from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.providers.codebrain import CodebrainProvider
from kiro_crew.providers.codebrain_resolver import ProviderResolver, ProviderStore


def _resolver(tmp_path: Path, *, detected: set[str] | None = None) -> ProviderResolver:
    detected = detected or set()
    return ProviderResolver(
        ProviderStore(tmp_path / "providers.json"),
        cli_lookup=lambda host: f"/fake/{host}" if host in detected else None,
    )


def test_codex_pin_corrects_an_incompatible_claude_model(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path, detected={"codex"})

    resolved = resolver.resolve(
        agent="codex",
        provider_id="codex-oauth",
        model="claude-haiku-4-5-20251001",
    )

    assert resolved.error is None
    assert resolved.agent == "codex"
    assert resolved.provider_id == "codex-oauth"
    assert resolved.model in resolved.provider.models
    assert not resolved.model.startswith("claude-")
    assert resolved.model_corrected is True


def test_explicit_stored_provider_wins_over_model_heuristic(tmp_path: Path) -> None:
    store_path = tmp_path / "providers.json"
    store_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "id": "team-codex",
                        "label": "Team Codex",
                        "type": "codex",
                        "host": "codex",
                        "models": ["team-gpt"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    resolver = _resolver(tmp_path)

    resolved = resolver.resolve(provider_id="team-codex", model="gpt-5.6-terra")

    assert resolved.error is None
    assert resolved.provider_id == "team-codex"
    assert resolved.agent == "codex"
    assert resolved.model == "team-gpt"
    assert resolved.model_corrected is True


def test_resolver_refuses_to_invent_an_undetected_provider(tmp_path: Path) -> None:
    resolved = _resolver(tmp_path).resolve(agent="codex")

    assert resolved.provider is None
    assert resolved.error == 'Native CLI "codex" is not installed or configured.'


@pytest.mark.asyncio
async def test_codex_provider_streams_jsonl_without_dangerous_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Writer:
        def __init__(self) -> None:
            self.data = b""
            self.closed = False

        def write(self, data: bytes) -> None:
            self.data += data

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class _Reader:
        def __init__(self, lines: list[bytes]) -> None:
            self.lines = lines

        async def readline(self) -> bytes:
            return self.lines.pop(0) if self.lines else b""

        async def read(self) -> bytes:
            return b""

    class _Process:
        def __init__(self) -> None:
            self.stdin = _Writer()
            self.stdout = _Reader(
                [
                    b'{"type":"thread.started","thread_id":"thread-42"}\n',
                    b'{"type":"item.completed","item":{"type":"agent_message","text":"hello"}}\n',
                ]
            )
            self.stderr = _Reader([])
            self.returncode = 0

        async def wait(self) -> int:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    captured: dict[str, object] = {}
    process = _Process()

    async def _spawn(*argv: str, **kwargs: object) -> _Process:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr("kiro_crew.providers.codebrain.shutil.which", lambda _: "/fake/codex")
    monkeypatch.setattr("kiro_crew.providers.codebrain.asyncio.create_subprocess_exec", _spawn)

    provider = CodebrainProvider(
        work_dir=tmp_path,
        model="gpt-5.6-terra",
        resolver=_resolver(tmp_path, detected={"codex"}),
    )
    events = [event async for event in provider.stream("say hello")]

    argv = captured["argv"]
    assert argv[:3] == ("/fake/codex", "exec", "--json")
    # `--approve-for-me` is what actually lets a mounted MCP tool run: codex's
    # default reviewer is the human, and a KiroCrew session has no codex TUI to
    # answer the prompt. It is mutually exclusive with `--sandbox` (the parser
    # rejects the pair), and reviews inside the same workspace-write sandbox, so
    # the explicit sandbox flag must NOT be present.
    assert "--approve-for-me" in argv
    assert "--sandbox" not in argv
    # "never ask" is not "always allow": this value REFUSES every MCP call.
    assert 'approval_policy="never"' not in argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert process.stdin.data == b"say hello"
    assert provider.session_id == "thread-42"
    assert [(event.kind, event.text) for event in events] == [
        (EVENT_TEXT_CHUNK, "hello"),
        (EVENT_COMPLETE, ""),
    ]


def test_codebrain_factory_constructs_a_direct_provider(tmp_path: Path) -> None:
    cfg = KiroCrewConfig()
    cfg = dataclasses.replace(
        cfg,
        agent=dataclasses.replace(
            cfg.agent,
            provider="codebrain",
            model="gpt-5.6-terra",
        ),
    )

    provider = cfg.create_provider_factory()(session_key="test", cwd=str(tmp_path))

    assert isinstance(provider, CodebrainProvider)
    assert provider.context_provider_type == "codebrain"
