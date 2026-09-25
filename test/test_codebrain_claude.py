"""The Claude Code transport: argv shape and stream-json translation.

Shapes measured against Claude Code CLI 2.1.282, not inferred from docs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kiro_crew.acp.types import EVENT_TEXT_CHUNK, EVENT_TOOL_CALL, EVENT_TOOL_RESULT
from kiro_crew.providers.codebrain import CodebrainProvider
from kiro_crew.providers.codebrain_resolver import (
    PROVIDER_REGISTRY,
    ProviderResolver,
    ProviderStore,
)


def _claude_provider(tmp_path: Path, monkeypatch) -> CodebrainProvider:
    monkeypatch.setattr("kiro_crew.providers.codebrain.shutil.which", lambda host: f"/fake/{host}")
    provider = CodebrainProvider(
        work_dir=tmp_path,
        agent="claude",
        resolver=ProviderResolver(
            ProviderStore(tmp_path / "providers.json"),
            cli_lookup=lambda host: f"/fake/{host}",
        ),
    )
    asyncio.run(provider.start())
    return provider


def test_registry_offers_only_transports_this_build_implements() -> None:
    """A profile whose transport does not exist turns config into a runtime failure."""
    assert {p.host for p in PROVIDER_REGISTRY} == {"claude", "codex"}
    assert all(p.is_virtual for p in PROVIDER_REGISTRY)


def test_claude_argv_is_headless_streaming_and_non_interactive(tmp_path, monkeypatch) -> None:
    argv = _claude_provider(tmp_path, monkeypatch)._argv()

    assert argv[0] == "/fake/claude"
    assert "-p" in argv
    # stream-json REQUIRES --verbose; the CLI refuses the pair without it.
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv
    # A KiroCrew session has no Claude TUI, so a prompting mode would hang.
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    # NOT the blanket bypass: that disables checks instead of answering them.
    assert "--dangerously-skip-permissions" not in argv


def test_claude_grants_permission_to_exactly_our_own_mcp_servers(tmp_path, monkeypatch) -> None:
    """`dontAsk` does NOT auto-approve MCP tools -- measured as a denial.

    The grant is per tool namespace and the server prefix suffices, so exactly the
    control plane this provider mounted is allowed and nothing else is.
    """
    argv = _claude_provider(tmp_path, monkeypatch)._argv()

    assert "--mcp-config" in argv
    granted = argv[argv.index("--allowedTools") + 1 :]
    assert all(g.startswith("mcp__kirocrew-") for g in granted if g.startswith("mcp__"))
    config = json.loads(argv[argv.index("--mcp-config") + 1])
    assert set(config["mcpServers"]) <= {"kirocrew-core", "kirocrew-cron"}


def test_claude_resumes_its_session_on_later_turns(tmp_path, monkeypatch) -> None:
    provider = _claude_provider(tmp_path, monkeypatch)
    assert "--resume" not in provider._argv()

    provider._remember_native_session_id(
        {"type": "system", "subtype": "init", "session_id": "sess-1"}
    )
    argv = provider._argv()

    assert argv[argv.index("--resume") + 1] == "sess-1"


def test_claude_blocks_become_text_tool_call_and_tool_result(tmp_path, monkeypatch) -> None:
    """Claude's vocabulary is block lists across two frames, not codex's item.*."""
    provider = _claude_provider(tmp_path, monkeypatch)

    assistant = provider._claude_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "checking"},
                    {
                        "type": "tool_use",
                        "id": "tu-1",
                        "name": "mcp__kirocrew-core__resource_status",
                        "input": {},
                    },
                ]
            },
        }
    )
    assert [e.kind for e in assistant] == [EVENT_TEXT_CHUNK, EVENT_TOOL_CALL]
    call = assistant[1]
    # The mcp__server__tool name is split so the host attributes the call.
    assert call.mcp_server_name == "kirocrew-core"
    assert call.tool_name == "resource_status"
    assert call.title == "@kirocrew-core/resource_status"

    result = provider._claude_events(
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu-1", "content": "5.7 GB"}
                ]
            },
        }
    )
    assert [e.kind for e in result] == [EVENT_TOOL_RESULT]
    assert result[0].tool_call_id == "tu-1"
    assert result[0].tool_output == "5.7 GB"
    assert result[0].tool_final is True


def test_a_denied_claude_tool_reports_its_denial(tmp_path, monkeypatch) -> None:
    provider = _claude_provider(tmp_path, monkeypatch)

    events = provider._claude_events(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-1",
                        "is_error": True,
                        "content": "Permission denied",
                    }
                ]
            },
        }
    )

    assert events[0].tool_output == "[failed] Permission denied"
    assert events[0].tool_status == "failed"


def test_an_auth_failure_is_surfaced_not_swallowed(tmp_path, monkeypatch) -> None:
    """The not-logged-in case arrives as an error frame with no content blocks.

    Dropping it would render an empty turn instead of the one sentence that
    explains why nothing happened.
    """
    provider = _claude_provider(tmp_path, monkeypatch)

    events = provider._claude_events(
        {"type": "result", "is_error": True, "result": "Not logged in - Please run /login"}
    )

    assert [(e.kind, e.text) for e in events] == [
        (EVENT_TEXT_CHUNK, "Not logged in - Please run /login")
    ]
