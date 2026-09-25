"""Direct Codebrain-style CLI provider.

The provider intentionally bypasses ``AcpProvider`` / ``AcpRuntime``.  Its first
supported executable is the locally installed Codex CLI, whose documented
``codex exec --json`` transport produces JSONL suitable for the KiroCrew event
surface.  Other catalog entries resolve normally but fail explicitly until a
matching direct stream profile is implemented.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from pathlib import Path
from typing import Any

from kiro_crew.acp.types import (
    EVENT_COMPLETE,
    EVENT_TEXT_CHUNK,
    EVENT_THINKING_CHUNK,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    STOP_REASON_END_TURN,
)
from kiro_crew.agent_sdk.provider_identity import PROVIDER_CODEBRAIN
from kiro_crew.providers.base import CancelOutcome, LLMEvent, LLMProvider

from .codebrain_env import build_provider_env
from .codebrain_resolver import ProviderResolver, ResolvedProvider, get_resolver

_ENV_KEY_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")

#: Seconds codex waits for a mounted MCP server's handshake. Crew's own control
#: plane is a Python process importing a large dependency tree, which on a cold or
#: loaded host outlasts codex's short default and gets the server dropped -- with
#: the tools silently absent rather than reported missing.
_MCP_STARTUP_TIMEOUT_SECS = 60


class CodebrainProvider(LLMProvider):
    """A direct native-CLI LLM provider.

    Direct CLI tools run inside the selected CLI process.  Accordingly this
    adapter does not fabricate ACP permission or tool-call events; its host
    stream contains assistant text and terminal status only.
    """

    def __init__(
        self,
        *,
        work_dir: Path | str,
        model: str | None = None,
        agent: str | None = None,
        provider_id: str | None = None,
        session_key: str | None = None,
        extra_env: Mapping[str, str] | None = None,
        resolver: ProviderResolver | None = None,
    ):
        self._work_dir = Path(work_dir)
        self._requested_model = model or None
        self._requested_agent = agent or None
        self._requested_provider_id = provider_id or None
        self._session_key = session_key or ""
        self._extra_env = dict(extra_env or {})
        self._resolver = resolver or get_resolver()
        self._resolved: ResolvedProvider | None = None
        self._binary = ""
        self._process: asyncio.subprocess.Process | None = None
        self._started = False
        self._closed = False
        self._active_turn = False
        self._native_session_id = ""

    @property
    def context_provider_type(self) -> str:
        return PROVIDER_CODEBRAIN

    @property
    def provider_label(self) -> str:
        """Stable session-map label read by the generic session helper."""
        return PROVIDER_CODEBRAIN

    @property
    def cwd(self) -> str:
        return str(self._work_dir)

    @property
    def session_id(self) -> str:
        return self._native_session_id or self._session_key

    @property
    def served_model(self) -> str:
        return (self._resolved.model if self._resolved else self._requested_model) or ""

    def context_usage_pct(self) -> float:
        return 0.0

    def context_usage_unknown(self) -> bool:
        return True

    def has_active_turn(self) -> bool:
        return self._active_turn

    def is_alive(self) -> bool:
        # A direct CLI process is intentionally short-lived per prompt.  The
        # provider remains reusable after a successful subprocess exit.
        return self._started and not self._closed

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Codebrain direct provider has been shut down")
        if self._started:
            return
        resolved = self._resolver.resolve(
            agent=self._requested_agent,
            provider_id=self._requested_provider_id,
            model=self._requested_model,
        )
        if resolved.error or resolved.provider is None:
            raise RuntimeError(resolved.error or "Direct provider resolution failed")
        if resolved.provider.host != "codex" or resolved.provider.type != "codex":
            raise RuntimeError(
                f"Direct CLI profile {resolved.provider.id!r} is registered but this build "
                "currently supports only the Codex JSONL transport."
            )
        binary = shutil.which("codex")
        if not binary:
            raise RuntimeError("Codex CLI is not available on PATH")
        self._resolved = resolved
        self._binary = binary
        self._started = True

    def _argv(self) -> list[str]:
        assert self._resolved is not None
        argv = [self._binary, "exec", "--json"]
        # Codex refuses a cwd that is not a git repo ("Not inside a trusted
        # directory") because its own trust model is per repository. A KiroCrew
        # session's work dir is frequently NOT a repo -- a per-session scratch
        # directory under the workspace root is the normal case -- so that guard
        # fails every such turn before the model is ever reached.
        argv.append("--skip-git-repo-check")
        # Approval routing, and the ONLY combination in which a mounted MCP tool
        # actually runs here. Measured, not assumed:
        #
        #   * plain `--sandbox workspace-write` leaves codex's approval reviewer
        #     set to the human, and a KiroCrew session has no codex TUI to answer
        #     the prompt, so every MCP call ends `status: failed` with
        #     "MCP tool call requires approval".
        #   * forcing `-c approval_policy="never"` does NOT grant the call; it
        #     REFUSES it, with the same message. "never ask" is not "always
        #     allow".
        #   * `--approve-for-me` routes approvals through automatic review and is
        #     mutually exclusive with `--sandbox` (the parser rejects the pair
        #     outright), so the explicit sandbox flag is dropped here -- not
        #     weakened: this flag reviews inside the SAME workspace-write sandbox,
        #     per codex's own help text.
        #
        # Crew's authorization is unaffected either way: these tools reach the
        # gateway over its own IPC and are gated there.
        argv.append("--approve-for-me")
        argv.extend(self._mcp_config_overrides())
        if self._resolved.model:
            argv.extend(["--model", self._resolved.model])
        # '-' asks Codex to read the prompt from stdin, preserving exact
        # whitespace and avoiding shell quoting entirely.
        argv.append("-")
        return argv

    def _mcp_config_overrides(self) -> list[str]:
        """``-c`` overrides mounting KiroCrew's own control plane in the child CLI.

        Without this the child gets NO KiroCrew tools at all -- no ``spawn_run``,
        no memory, no cron -- and a model asked to delegate answers as though it
        had, because nothing tells it the tool is absent. That is the failure this
        exists to prevent, and it is worse than a visible error: the turn LOOKS
        successful and no work happened.

        ``approval_policy`` is deliberately NOT overridden here. Setting it to
        ``"never"`` reads like "do not prompt" but means "refuse anything that
        would prompt", and it makes every MCP call fail with
        "requires approval, but approval policy is never". The approval routing
        that DOES work is the ``--approve-for-me`` flag in :meth:`_argv`.

        The server names are TOML bare keys, so ``kirocrew-core`` is mounted as
        ``kirocrew_core`` -- a hyphen would need quoting inside a ``-c`` value and
        codex's key parser does not accept it.
        """
        from kiro_crew.agent import managed_mcp_spec_entry

        overrides: list[str] = []
        for name in ("kirocrew-core", "kirocrew-cron"):
            try:
                entry = managed_mcp_spec_entry(name)
            except Exception:
                entry = None
            if not isinstance(entry, dict):
                continue
            command = entry.get("command")
            if not isinstance(command, str) or not command:
                continue
            key = name.replace("-", "_")
            env = {
                env_key: env_value
                for env_key, env_value in (entry.get("env") or {}).items()
                if isinstance(env_key, str) and isinstance(env_value, str)
            }
            if self._session_key:
                # The control-plane tools resolve WHICH session is calling from
                # this variable; without it a strict-identity tool refuses rather
                # than acting on the wrong session.
                env["KIROCREW_SESSION_KEY"] = self._session_key
            inline_env = ", ".join(f"{k}={json.dumps(v)}" for k, v in env.items())
            overrides.extend(
                [
                    "-c",
                    f"mcp_servers.{key}.command={json.dumps(command)}",
                    "-c",
                    f"mcp_servers.{key}.args={json.dumps(list(entry.get('args') or []))}",
                    "-c",
                    f"mcp_servers.{key}.env={{{inline_env}}}",
                    # The Python control plane imports a large dependency tree at
                    # boot; codex's default startup window is short enough to drop
                    # the server on a cold, loaded host.
                    "-c",
                    f"mcp_servers.{key}.startup_timeout_sec={_MCP_STARTUP_TIMEOUT_SECS}",
                ]
            )
        return overrides

    @staticmethod
    def _valid_env_key(key: object) -> bool:
        return (
            isinstance(key, str)
            and bool(key)
            and (key[0].isalpha() or key[0] == "_")
            and all(char in _ENV_KEY_CHARS for char in key)
        )

    def _environment(self) -> dict[str, str]:
        """The child environment, routed at the resolved provider.

        Layered lowest to highest: the gateway's own environment, the stored
        profile's literal ``env``, the routing overlay derived from the profile's
        base URL / token / model, and finally the caller's ``extra_env``. The
        derived overlay sits above the literal profile env because that is what
        turns a bare ``{"ANTHROPIC_AUTH_TOKEN": ...}`` profile into a CLI
        actually pointed at the endpoint; ``extra_env`` stays on top so a
        session-specific override is never silently reverted by a profile.
        """
        environment = os.environ.copy()
        sources: list[Mapping[str, str]] = []
        if self._resolved is not None and self._resolved.provider is not None:
            sources.append(self._resolved.provider.env)
            sources.append(
                build_provider_env(
                    self._resolved.provider,
                    self._resolved.model,
                    ambient=environment,
                )
            )
        sources.append(self._extra_env)
        for source in sources:
            for key, value in source.items():
                if self._valid_env_key(key) and isinstance(value, str):
                    environment[key] = value
        return environment

    def _remember_native_session_id(self, payload: Mapping[str, Any]) -> None:
        candidates: list[object] = [
            payload.get("thread_id"),
            payload.get("threadId"),
            payload.get("session_id"),
            payload.get("sessionId"),
        ]
        for nested_key in ("thread", "item"):
            nested = payload.get(nested_key)
            if isinstance(nested, Mapping):
                candidates.extend(
                    (
                        nested.get("id"),
                        nested.get("thread_id"),
                        nested.get("threadId"),
                        nested.get("session_id"),
                    )
                )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                self._native_session_id = candidate.strip()
                return

    @staticmethod
    def _text_from_json_event(payload: Mapping[str, Any]) -> str:
        """Extract only agent output from known Codex JSONL event shapes."""
        delta = payload.get("delta")
        if isinstance(delta, str):
            return delta
        item = payload.get("item")
        if isinstance(item, Mapping):
            item_type = item.get("type")
            if item_type in {"agent_message", "message"}:
                text = item.get("text") or item.get("message")
                if isinstance(text, str):
                    return text
                content = item.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, Mapping) and isinstance(part.get("text"), str)
                    )
        if payload.get("type") in {"agent_message", "message"}:
            text = payload.get("text") or payload.get("message")
            if isinstance(text, str):
                return text
        return ""

    def _events_from_json_event(self, payload: Mapping[str, Any]) -> list[LLMEvent]:
        """Translate one Codex JSONL frame into the host's event vocabulary.

        Text alone is not enough. A turn that delegates work, runs a command, or
        calls an MCP tool produces NO assistant text while it happens, so a
        text-only translation renders a session that sits silent and then claims
        to have done something -- the dashboard shows no tool pill, no subagent
        row, and the user cannot tell a real spawn from a model that only said it
        spawned. These frames are the evidence, so they are forwarded.

        Unknown item types yield nothing rather than raising: codex adds item
        kinds between releases, and an unrecognised frame must not end a turn.
        """
        kind = payload.get("type")
        item = payload.get("item") if isinstance(payload.get("item"), Mapping) else {}
        item_type = item.get("type") if isinstance(item, Mapping) else None

        # Streaming text delta, or a completed assistant message.
        text = self._text_from_json_event(payload)
        if text:
            return [LLMEvent(kind=EVENT_TEXT_CHUNK, text=text)]

        if item_type == "reasoning":
            summary = item.get("text") or item.get("summary") or ""
            if isinstance(summary, str) and summary.strip():
                return [LLMEvent(kind=EVENT_THINKING_CHUNK, text=summary)]
            return []

        if item_type in ("mcp_tool_call", "command_execution", "local_shell_call", "web_search"):
            call_id = str(item.get("id") or "")
            title = self._call_title(item, item_type)
            if kind == "item.started":
                return [
                    LLMEvent(
                        kind=EVENT_TOOL_CALL,
                        tool_call_id=call_id,
                        title=title,
                        wire_title=title,
                        tool_kind="mcp" if item_type == "mcp_tool_call" else "execute",
                        tool_input=self._call_input(item),
                    )
                ]
            if kind == "item.completed":
                status = str(item.get("status") or "completed")
                return [
                    LLMEvent(
                        kind=EVENT_TOOL_RESULT,
                        tool_call_id=call_id,
                        title=title,
                        wire_title=title,
                        tool_kind="mcp" if item_type == "mcp_tool_call" else "execute",
                        # The RESULT BODY, not just the verdict. Carrying only the
                        # status is what let a refused call look like a successful
                        # one: the model narrates "agent started" from a refusal
                        # the host never displayed, and the user sees a tool pill
                        # with no output and no error. The status leads so a
                        # failure is legible even when the body is empty.
                        text=self._call_result_text(item, status),
                    )
                ]
        return []

    @staticmethod
    def _call_result_text(item: Mapping[str, Any], status: str) -> str:
        """The call's own output, prefixed by its verdict when it did not succeed.

        Codex reports a failure in ``error.message`` and success content in
        ``result``; an MCP result is the content-block list the protocol defines,
        so its text parts are joined rather than dumped as JSON.
        """
        error = item.get("error")
        if isinstance(error, Mapping):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return f"[{status}] {message}"
        result = item.get("result")
        body = ""
        if isinstance(result, str):
            body = result
        elif isinstance(result, Mapping):
            content = result.get("content")
            if isinstance(content, list):
                body = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, Mapping) and isinstance(part.get("text"), str)
                )
            if not body:
                try:
                    body = json.dumps(result)
                except (TypeError, ValueError):
                    body = str(result)
        elif isinstance(result, list):
            body = "".join(
                part.get("text", "")
                for part in result
                if isinstance(part, Mapping) and isinstance(part.get("text"), str)
            )
        if status not in ("completed", "success", "ok"):
            return f"[{status}] {body}".strip()
        return body

    @staticmethod
    def _call_title(item: Mapping[str, Any], item_type: str) -> str:
        """A display label for a tool/command frame, from what codex reported."""
        server = item.get("server")
        tool = item.get("tool") or item.get("name")
        if isinstance(server, str) and server and isinstance(tool, str) and tool:
            return f"@{server}/{tool}"
        if isinstance(tool, str) and tool:
            return tool
        command = item.get("command")
        if isinstance(command, str) and command.strip():
            return command
        if isinstance(command, list):
            return " ".join(str(part) for part in command)
        return item_type.replace("_", " ")

    @staticmethod
    def _call_input(item: Mapping[str, Any]) -> str:
        """The call's arguments as a display string, never as parsed authority."""
        for key in ("arguments", "args", "input", "command"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, (list, dict)) and value:
                try:
                    return json.dumps(value)
                except (TypeError, ValueError):
                    return str(value)
        return ""

    async def _stream_direct(self, message: str) -> AsyncIterator[LLMEvent]:
        if not self._started:
            await self.start()
        if self._active_turn:
            raise RuntimeError("Direct provider already has an active turn")
        if not message:
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)
            return

        self._active_turn = True
        process: asyncio.subprocess.Process | None = None
        stderr_task: asyncio.Task[bytes] | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *self._argv(),
                cwd=str(self._work_dir),
                env=self._environment(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._process = process
            assert process.stdin is not None and process.stdout is not None and process.stderr is not None
            process.stdin.write(message.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
            stderr_task = asyncio.create_task(process.stderr.read())

            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    # A future Codex version may print human-readable output
                    # despite --json. Preserve it as text rather than losing it.
                    yield LLMEvent(kind=EVENT_TEXT_CHUNK, text=raw + "\n")
                    continue
                if not isinstance(event, Mapping):
                    continue
                self._remember_native_session_id(event)
                for translated in self._events_from_json_event(event):
                    yield translated

            return_code = await process.wait()
            if stderr_task is not None:
                await stderr_task
            if return_code != 0:
                # Do not echo stderr: CLI output can include provider diagnostics
                # or data supplied in the user's prompt.
                raise RuntimeError(f"Codex direct CLI exited with status {return_code}")
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)
        except asyncio.CancelledError:
            await self.cancel(wait_ack_timeout=1.0)
            raise
        finally:
            if stderr_task is not None and not stderr_task.done():
                stderr_task.cancel()
            self._process = None
            self._active_turn = False

    async def stream(self, message: str) -> AsyncIterator[LLMEvent]:
        """Send through the shared essential-context receipt before the CLI call."""
        async with aclosing(
            self.essential_delivery.stream(
                message,
                self._stream_direct,
                lambda: self.context_incarnation,
            )
        ) as events:
            async for event in events:
                yield event

    async def approve_tool(self, request_id: str | int, *, always: bool = False) -> None:
        # Direct CLI profiles own their own approval channel and do not emit ACP
        # permission frames for the host to answer.
        return None

    async def reject_tool(self, request_id: str | int) -> None:
        return None

    async def cancel(self, *, wait_ack_timeout: float = 0.0) -> CancelOutcome:
        process = self._process
        if process is None or process.returncode is not None:
            return "no_turn"
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=max(wait_ack_timeout, 0.1))
            return "acked"
        except (ProcessLookupError, asyncio.TimeoutError):
            try:
                process.kill()
                await process.wait()
                return "acked"
            except (ProcessLookupError, OSError):
                return "error"

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.cancel(wait_ack_timeout=1.0)


__all__ = ["CodebrainProvider"]
