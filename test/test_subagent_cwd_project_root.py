"""The parent session's project directory is an allowed sub-agent cwd."""

from __future__ import annotations

import os
from pathlib import Path

from kiro_crew.subagent import validate_cwd
from kiro_crew.subagent_manager.admission.gate import _GateMixin


class _Provider:
    def __init__(self, cwd: str) -> None:
        self.cwd = cwd


class _Sessions:
    def __init__(self, providers: dict[str, object]) -> None:
        self._providers = providers

    def get_provider(self, key: str):
        return self._providers.get(key)


class _Manager:
    def __init__(self, sessions: _Sessions) -> None:
        self._sessions = sessions


class _Gate(_GateMixin):
    """Only the seam under test; the mixin's own __init__ needs a whole manager."""

    def __init__(self, manager: _Manager) -> None:  # noqa: D107 - test double
        self._manager = manager


def _gate(providers: dict[str, object]) -> _Gate:
    return _Gate(_Manager(_Sessions(providers)))


def test_project_root_is_read_off_the_parents_live_provider(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    gate = _gate({"chat-1": _Provider(str(project))})

    assert gate._parent_project_root("chat-1") == os.path.realpath(project)


def test_a_parent_without_a_live_provider_contributes_no_root() -> None:
    """A cron or headless caller must not widen the allow-list by accident."""
    gate = _gate({})

    assert gate._parent_project_root("chat-1") == ""
    assert gate._parent_project_root("") == ""


def test_the_projects_own_tree_passes_validate_cwd_once_it_is_a_root(tmp_path: Path) -> None:
    """The rule's PAYOFF: a project outside the shipped work-tree roots works.

    Without the parent's project as a root, every spawn in such a project is
    refused and retried with no cwd -- which silently moves the child to an
    unrelated directory, so its file reads answer for the wrong tree.
    """
    project = tmp_path / "Downloads" / "two_projects"
    (project / "sub").mkdir(parents=True)
    shipped_roots = [str(tmp_path / "workspace")]

    refused, err = validate_cwd(str(project / "sub"), shipped_roots)
    assert refused == "" and "not under any allowed root" in err

    gate = _gate({"chat-1": _Provider(str(project))})
    widened = [*shipped_roots, gate._parent_project_root("chat-1")]
    resolved, err = validate_cwd(str(project / "sub"), widened)

    assert err == ""
    assert resolved == os.path.realpath(project / "sub")


def test_an_empty_allow_list_still_means_no_cwd_overrides(tmp_path: Path) -> None:
    """Disabling the feature outright must not be re-enabled by the project rule.

    The gate only widens a NON-empty list, so this asserts the contract the
    widening relies on: an empty list refuses every cwd.
    """
    project = tmp_path / "repo"
    project.mkdir()

    resolved, err = validate_cwd(str(project), [])

    assert resolved == ""
    assert "disabled" in err
