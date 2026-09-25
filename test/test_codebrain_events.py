"""Tests for the Codex JSONL -> host event translation."""

from __future__ import annotations

from pathlib import Path

from kiro_crew.acp.types import (
    EVENT_TEXT_CHUNK,
    EVENT_THINKING_CHUNK,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
)
from kiro_crew.providers.codebrain import CodebrainProvider


def _provider(tmp_path: Path) -> CodebrainProvider:
    return CodebrainProvider(work_dir=tmp_path)


def test_spawn_launch_text_lands_in_tool_output(tmp_path: Path) -> None:
    """The transcript's sub-agent card parses ``meta.output``, not the text body.

    The host persists ``tool_output`` into that field, so a launch delivered as
    ``text`` renders as assistant prose and the card never appears even though the
    spawn succeeded. This asserts the field, and that the ``Spawned N`` header
    stays at the START of a line -- the card's regex is line-anchored.
    """
    launch = "Spawned 1 subagent(s). Results will arrive as completion events:\n  93b2b25c38e1: hello"
    events = _provider(tmp_path)._events_from_json_event(
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "mcp_tool_call",
                "server": "kirocrew_core",
                "tool": "spawn_run",
                "status": "completed",
                "result": {"content": [{"type": "text", "text": launch}]},
            },
        }
    )

    assert len(events) == 1
    event = events[0]
    assert event.kind == EVENT_TOOL_RESULT
    assert event.tool_output == launch
    assert event.tool_output.startswith("Spawned 1 subagent(s).")
    assert event.tool_status == "completed"
    assert event.tool_final is True
    # The body must NOT also ride `text`: that field renders as assistant prose.
    assert event.text == ""


def test_start_creates_the_session_work_dir(tmp_path: Path, monkeypatch) -> None:
    """The child's ``cwd`` must exist, or the run dies before reaching the model.

    ``_session_work_dir`` composes the path without creating it, so a sub-agent
    (key ``subagent:<id>`` -> ``subagent_<id>`` under the workspace root) failed
    with a bare FileNotFoundError reported as the RUN's output.
    """
    import asyncio

    from kiro_crew.providers.codebrain_resolver import ProviderResolver, ProviderStore

    work_dir = tmp_path / "workspace" / "subagent_8a6d51369b7736d5"
    assert not work_dir.exists()
    monkeypatch.setattr("kiro_crew.providers.codebrain.shutil.which", lambda _: "/fake/codex")
    provider = CodebrainProvider(
        work_dir=work_dir,
        agent="codex",
        resolver=ProviderResolver(
            ProviderStore(tmp_path / "providers.json"),
            cli_lookup=lambda host: f"/fake/{host}" if host == "codex" else None,
        ),
    )

    asyncio.run(provider.start())

    assert work_dir.is_dir()


def test_started_tool_call_is_announced_with_its_title(tmp_path: Path) -> None:
    events = _provider(tmp_path)._events_from_json_event(
        {
            "type": "item.started",
            "item": {
                "id": "item_1",
                "type": "mcp_tool_call",
                "server": "kirocrew_core",
                "tool": "spawn_run",
                "status": "in_progress",
                "arguments": {"task": "hello"},
            },
        }
    )

    assert [e.kind for e in events] == [EVENT_TOOL_CALL]
    assert events[0].title == "@kirocrew_core/spawn_run"
    assert events[0].tool_kind == "mcp"
    assert "hello" in events[0].tool_input
    # Identity fields, not just the display title: is_coding_event, the crew log,
    # the SEL record and the turn's identity tracker all read these, so an empty
    # pair makes a real MCP call arrive as an unnamed tool.
    assert events[0].tool_name == "spawn_run"
    assert events[0].mcp_server_name == "kirocrew_core"
    assert events[0].is_shell is False


def test_a_failed_call_reports_its_error_rather_than_silence(tmp_path: Path) -> None:
    """A refusal must be visible, or the model narrates success over it."""
    events = _provider(tmp_path)._events_from_json_event(
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "mcp_tool_call",
                "server": "kirocrew_core",
                "tool": "spawn_run",
                "status": "failed",
                "error": {"message": "requires approval"},
            },
        }
    )

    assert events[0].tool_output == "[failed] requires approval"
    assert events[0].tool_status == "failed"


def test_reasoning_becomes_a_thinking_chunk(tmp_path: Path) -> None:
    events = _provider(tmp_path)._events_from_json_event(
        {"type": "item.completed", "item": {"type": "reasoning", "text": "pondering"}}
    )

    assert [(e.kind, e.text) for e in events] == [(EVENT_THINKING_CHUNK, "pondering")]


def test_agent_message_becomes_a_text_chunk(tmp_path: Path) -> None:
    events = _provider(tmp_path)._events_from_json_event(
        {"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}
    )

    assert [(e.kind, e.text) for e in events] == [(EVENT_TEXT_CHUNK, "hi")]


def test_an_unknown_item_kind_yields_nothing(tmp_path: Path) -> None:
    """Codex adds item kinds between releases; one must not end a turn."""
    assert _provider(tmp_path)._events_from_json_event(
        {"type": "item.completed", "item": {"type": "some_future_kind"}}
    ) == []


def test_thread_id_is_never_taken_from_an_item_id(tmp_path: Path) -> None:
    """An item id must not be mistaken for the thread id.

    Every ``item.*`` frame carries ``item_0``, so accepting it overwrote the real
    thread id on the first tool call and the next turn resumed a session that does
    not exist. Codex answers that with a fresh EMPTY thread instead of an error,
    so the symptom was "the model forgot", not a visible failure.
    """
    provider = _provider(tmp_path)

    provider._remember_native_session_id({"type": "thread.started", "thread_id": "th-42"})
    assert provider._native_session_id == "th-42"

    provider._remember_native_session_id(
        {"type": "item.completed", "item": {"id": "item_0", "type": "agent_message"}}
    )
    assert provider._native_session_id == "th-42"


def test_a_resumed_turn_keeps_history_and_tools(tmp_path: Path, monkeypatch) -> None:
    """Both halves of a session must work at once.

    ``resume`` rejects ``--approve-for-me`` and requires every option BEFORE the
    positional session id, so the approval must come from the config key behind
    that flag -- otherwise a resumed turn keeps its history and loses every tool.
    """
    import asyncio

    from kiro_crew.providers.codebrain_resolver import ProviderResolver, ProviderStore

    monkeypatch.setattr("kiro_crew.providers.codebrain.shutil.which", lambda _: "/fake/codex")
    provider = CodebrainProvider(
        work_dir=tmp_path,
        agent="codex",
        resolver=ProviderResolver(
            ProviderStore(tmp_path / "providers.json"),
            cli_lookup=lambda host: f"/fake/{host}" if host == "codex" else None,
        ),
    )
    asyncio.run(provider.start())

    first = provider._argv()
    assert "resume" not in first
    assert "--approve-for-me" in first

    provider._native_session_id = "th-42"
    resumed = provider._argv()

    assert resumed[1:3] == ["exec", "resume"]
    # Approval survives the loss of the flag.
    assert "--approve-for-me" not in resumed
    assert 'approvals_reviewer="auto_review"' in resumed
    # The id is positional, last before the stdin marker, and every option
    # precedes it -- a flag after the id is a usage error.
    assert resumed[-2:] == ["th-42", "-"]
