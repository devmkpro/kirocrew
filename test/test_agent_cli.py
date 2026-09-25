"""``agent_cli``: the allowlist that keeps a terminal from becoming an exec surface.

The tests that matter here are the refusals. ``dashboard/handlers/terminal.py``
states the invariant — "client is never given a say in WHICH shell is spawned" —
and this module exists to let a caller pick an agent CLI without breaking it. So
the pins below are mostly about what a caller CANNOT reach: an id outside the
table, a path, an argv, an injected argument.

The PATH tests are the other half, and they are not hypothetical. The spawn runs
with the chat's PROJECT directory as cwd, so a bare program name would be resolved
a second time there. A project that ships its own ``codex`` on a relative PATH
entry would then win a race the validation already decided, which is why every
returned path is absolute.
"""

from __future__ import annotations

import os
import stat

import pytest

from kiro_crew.dashboard import agent_cli


def _make_exe(directory, name: str) -> str:
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


class TestTable:
    def test_claude_and_codex_are_declared(self) -> None:
        """The two the user asked for, by id."""
        assert "claude" in agent_cli.known_ids()
        assert "codex" in agent_cli.known_ids()

    def test_every_entry_is_complete(self) -> None:
        """A row with no install hint is a dead end in the picker."""
        for entry in agent_cli.AGENT_CLIS:
            assert entry.id and entry.display and entry.binary
            assert entry.install_hint, f"{entry.id} has no install hint"
            assert entry.credential_note, f"{entry.id} does not say where its key lives"

    def test_ids_are_unique(self) -> None:
        ids = agent_cli.known_ids()
        assert len(ids) == len(set(ids))

    def test_known_ids_keeps_declaration_order(self) -> None:
        """Sorted order would bury the most-asked-for choice."""
        assert agent_cli.known_ids() == tuple(e.id for e in agent_cli.AGENT_CLIS)

    def test_no_entry_declares_shell_metacharacters(self) -> None:
        """argv is exec'd, not shell-interpreted; a metachar here would be a smell."""
        for entry in agent_cli.AGENT_CLIS:
            for token in (entry.binary, *entry.args):
                assert not set(token) & set(";|&$`><\n"), f"{entry.id}: {token!r}"

    def test_get_returns_none_for_unknown(self) -> None:
        assert agent_cli.get("definitely-not-a-cli") is None
        assert agent_cli.get("") is None


class TestResolveRefusals:
    """What a caller must not be able to reach through the id parameter."""

    def test_unknown_id_is_refused(self) -> None:
        assert agent_cli.resolve("not-in-the-table") is None

    @pytest.mark.parametrize(
        "hostile",
        [
            "/bin/sh",
            "../../bin/sh",
            "claude; rm -rf /",
            "claude && whoami",
            "$(which sh)",
            "claude\nsh",
            "",
            "CLAUDE",
        ],
    )
    def test_a_path_or_injection_is_not_an_id(self, hostile: str) -> None:
        """The table is the only way in: nothing path-shaped resolves."""
        assert agent_cli.resolve(hostile) is None

    def test_absent_binary_is_refused(self, monkeypatch, tmp_path) -> None:
        """An empty PATH means nothing is launchable -- not a guessed location."""
        monkeypatch.setenv("PATH", str(tmp_path))
        assert agent_cli.resolve("claude") is None
        assert agent_cli.resolve("codex") is None


class TestResolvePinsAbsolutePaths:
    def test_resolved_argv_is_absolute(self, monkeypatch, tmp_path) -> None:
        _make_exe(tmp_path, "codex")
        monkeypatch.setenv("PATH", str(tmp_path))
        argv = agent_cli.resolve("codex")
        assert argv is not None
        assert os.path.isabs(argv[0]), argv
        assert argv[0] == str(tmp_path / "codex")

    def test_a_relative_path_entry_is_anchored(self, monkeypatch, tmp_path) -> None:
        """The race the pinning exists to lose: PATH=bin, cwd re-resolution.

        With a relative PATH entry, ``which`` returns a relative result. Anchoring
        it in the gateway's cwd is what stops the session's project directory from
        resolving the name a second time to a different program.
        """
        bindir = tmp_path / "bin"
        bindir.mkdir()
        _make_exe(bindir, "claude")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", "bin")
        argv = agent_cli.resolve("claude")
        assert argv is not None
        assert os.path.isabs(argv[0]), f"relative PATH entry leaked through: {argv[0]!r}"
        assert argv[0] == str(bindir / "claude")

    def test_declared_args_follow_the_program(self, monkeypatch, tmp_path) -> None:
        _make_exe(tmp_path, "claude")
        monkeypatch.setenv("PATH", str(tmp_path))
        argv = agent_cli.resolve("claude")
        entry = agent_cli.get("claude")
        assert entry is not None
        assert argv == (str(tmp_path / "claude"), *entry.args)


class TestDetect:
    def test_reports_every_declared_cli_even_when_absent(self, monkeypatch, tmp_path) -> None:
        """An omitted row is the least diagnosable UI there is."""
        monkeypatch.setenv("PATH", str(tmp_path))
        rows = agent_cli.detect()
        assert [r.id for r in rows] == list(agent_cli.known_ids())
        assert all(not r.installed for r in rows)
        assert all(r.path == "" for r in rows)
        assert all(r.install_hint for r in rows)

    def test_reports_installed_with_absolute_path(self, monkeypatch, tmp_path) -> None:
        _make_exe(tmp_path, "codex")
        monkeypatch.setenv("PATH", str(tmp_path))
        rows = {r.id: r for r in agent_cli.detect()}
        assert rows["codex"].installed is True
        assert rows["codex"].path == str(tmp_path / "codex")
        assert rows["claude"].installed is False
        assert rows["claude"].path == ""

    def test_absent_cli_never_guesses_a_path(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PATH", str(tmp_path))
        for row in agent_cli.detect():
            if not row.installed:
                assert row.path == "", f"{row.id} guessed {row.path!r}"

    def test_detection_is_not_cached(self, monkeypatch, tmp_path) -> None:
        """Installing something must change the answer without a restart."""
        monkeypatch.setenv("PATH", str(tmp_path))
        assert {r.id for r in agent_cli.detect() if r.installed} == set()
        _make_exe(tmp_path, "claude")
        assert {r.id for r in agent_cli.detect() if r.installed} == {"claude"}
