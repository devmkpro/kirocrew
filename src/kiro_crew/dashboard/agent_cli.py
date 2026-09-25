"""Which agent CLIs this host can launch in a terminal, and where they are.

What this is for: a pane that runs the VENDOR'S OWN CLI — ``claude``, ``codex`` —
directly on a PTY, so a Claude or Codex plan is used through the tool the vendor
authenticates, with no harness in between and no API key stored here. The
credential stays where that CLI already keeps it.

**The client never names the program.** ``dashboard/handlers/terminal.py`` states
the invariant this module is built to preserve: "client is never given a say in
WHICH shell is spawned". A terminal that accepted a path or an argv from the
browser would be a remote-execution surface, owner-gated or not. So a caller names
an ``id`` from :data:`AGENT_CLIS` — a table in this file, in the server — and the
program is resolved here. An unknown id is refused rather than passed through, and
there is deliberately no escape hatch that takes a path.

**Resolution mirrors ``_resolve_shell``** for one reason worth stating: the spawn
runs with the session's cwd, which is the chat's PROJECT directory. A bare name
would be resolved a second time there, so a project-planted ``codex`` on a
relative ``PATH`` entry could win a race the validation already decided. Every
path returned here is therefore the absolute path ``shutil.which`` resolved,
anchored in the gateway's cwd, exactly as the shell resolver pins its own.

This module only ANSWERS; it never launches, and it holds no state. Detection is a
filesystem question whose answer changes when the user installs something, so a
cache belongs to the caller that knows how long its own answer may be stale.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentCli:
    """One launchable agent CLI, as declared by the server.

    ``args`` is the argv the harness needs AFTER the program and is part of the
    declaration, never of the request: a caller that could append arguments could
    turn ``codex`` into ``codex --some-exec-flag``, which is the same
    remote-execution hole by a longer road.
    """

    id: str
    display: str
    binary: str
    args: tuple[str, ...] = ()
    #: What the user must do when the binary is absent. Shown verbatim, so it
    #: names the vendor's own installer rather than implying Kiro Crew ships it.
    install_hint: str = ""
    #: Where that CLI keeps its credential, for the UI to explain why no API key
    #: is asked for here. Not read by any code path -- it is operator-facing text.
    credential_note: str = ""


#: The launchable set. A table, not discovery: a host scan that offered whatever
#: it found on PATH would let an installed-anything become a spawnable program,
#: and the point of the allowlist is that the server decides what is offerable.
AGENT_CLIS: tuple[AgentCli, ...] = (
    AgentCli(
        id="claude",
        display="Claude Code",
        binary="claude",
        install_hint="Install Anthropic's Claude Code CLI, then run its own login.",
        credential_note="Claude Code holds the credential; nothing is stored here.",
    ),
    AgentCli(
        id="codex",
        display="Codex",
        binary="codex",
        install_hint="Install OpenAI's Codex CLI, then run its own login.",
        credential_note="Codex holds the credential; nothing is stored here.",
    ),
    AgentCli(
        id="gemini",
        display="Gemini",
        binary="gemini",
        install_hint="Install Google's Gemini CLI, then run its own login.",
        credential_note="The Gemini CLI holds the credential; nothing is stored here.",
    ),
    AgentCli(
        id="kiro",
        display="Kiro CLI",
        binary="kiro-cli",
        install_hint="Kiro Crew's own CLI; install it with the product installer.",
        credential_note="Signed in through the Kiro plan, not with a pasted key.",
    ),
)

_BY_ID: dict[str, AgentCli] = {entry.id: entry for entry in AGENT_CLIS}


def known_ids() -> tuple[str, ...]:
    """Every id a caller may name, in declaration order.

    Declaration order and not sorted: the table is written most-asked-for first,
    and a picker that reorders it alphabetically buries the common choice.
    """
    return tuple(entry.id for entry in AGENT_CLIS)


def get(agent_id: str) -> AgentCli | None:
    """The declaration for *agent_id*, or ``None`` when it is not in the table."""
    return _BY_ID.get(agent_id)


def resolve(agent_id: str) -> tuple[str, ...] | None:
    """The argv to launch *agent_id*, or ``None`` when it cannot be launched.

    Returns ``None`` for BOTH an unknown id and an id whose binary is absent, and
    that conflation is deliberate at this layer: a caller must not branch on the
    difference, because "not offerable" is the only distinction a spawn path needs
    and the two failures deserve the same refusal. A UI that wants to say which
    one it is calls :func:`detect` instead, which reports them separately.

    Blocking: ``shutil.which`` stats every ``PATH`` entry. Callers on the event
    loop must run this in an executor, never inline -- the same note
    ``_resolve_shell`` carries, for the same reason.
    """
    entry = _BY_ID.get(agent_id)
    if entry is None:
        return None
    resolved = shutil.which(entry.binary)
    if not resolved:
        return None
    # abspath: `which` joins the matching PATH entry verbatim, so a RELATIVE
    # entry yields a relative result the session's project cwd would re-resolve
    # -- the substitution this pinning exists to prevent.
    return (os.path.abspath(resolved), *entry.args)


@dataclass(frozen=True)
class AgentCliStatus:
    """One row of the picker: what it is, and whether it can run here."""

    id: str
    display: str
    binary: str
    installed: bool
    #: Absolute path when installed, else "" -- never a guess at where it would be.
    path: str
    install_hint: str
    credential_note: str


def detect() -> list[AgentCliStatus]:
    """Probe every declared CLI and report what this host can actually launch.

    The auto-detection a provider picker needs: one row per declared CLI, so an
    absent one is shown WITH its install hint rather than omitted. A picker that
    listed only what is installed cannot tell the user what they are missing, and
    "the option vanished" is the least diagnosable UI there is.

    No caching, by design -- see the module docstring. Blocking, for the reason
    :func:`resolve` states.
    """
    rows: list[AgentCliStatus] = []
    for entry in AGENT_CLIS:
        resolved = shutil.which(entry.binary)
        path = os.path.abspath(resolved) if resolved else ""
        rows.append(
            AgentCliStatus(
                id=entry.id,
                display=entry.display,
                binary=entry.binary,
                installed=bool(resolved),
                path=path,
                install_hint=entry.install_hint,
                credential_note=entry.credential_note,
            )
        )
    return rows
