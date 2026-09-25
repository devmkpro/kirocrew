"""Agent-to-agent inbox: a durable recado one session leaves for another.

What this is FOR, and why it is not ``session_send``. ``session_send`` injects text
into a target session as a user turn: the message IS the wake, so an idle target
starts a turn and a busy one queues the text in its slot. That is the right shape
for "do this now" and the wrong one for "here is something for when you get to it"
— the slot queue is capped (32 entries / 256 KB) and a body over the cap is
accepted but does not survive a restart, so a recado routed that way can be lost
silently.

The inbox splits the two halves that ``session_send`` fuses:

* the MESSAGE is committed to durable storage, addressed to a recipient, and
* the WAKE carries only a pointer — "you have mail" — never the body.

That ordering is the whole contract. :func:`enqueue` returns only after the row is
committed, so a wake that never lands, or a recipient that crashes before reading,
loses nothing: the row is still unread and the next read finds it. A design that
woke first and wrote second would have a window where the recipient is told to read
an inbox that is empty, and the message would be gone with no record it existed.

**At-least-once, never at-most-once.** :func:`read_unread` claims rows by stamping
``read_at`` inside the same transaction that selects them, so two concurrent
readers of one recipient cannot both receive a row. A reader that dies after the
claim and before acting has consumed the message — the alternative (claim on
acknowledgement) needs an ack channel this feature does not have, and a message
delivered twice is worse than one delivered once and dropped by a crashed reader,
because a recado is usually an instruction.

Storage is the task-queue database, reached through a connection the caller owns.
This module deliberately takes a :class:`sqlite3.Connection` rather than opening
its own: the queue's store already holds the lock, the ``BEGIN IMMEDIATE`` and the
busy timeout that make the commit-before-wake ordering atomic, and a second
connection with its own pragmas would quietly opt out of them.

What must NEVER reach this table, and why each one is named rather than left to
judgement: a PTY handle or anything else that ties a message to a live process (the
row outlives every process), a credential (the body is read back into a transcript),
a filesystem path into a worktree (the worktree may be gone when the row is read),
or a Python callable (a row must be meaningful to a build that has never imported
the sender's code).
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass

#: Hard ceiling on one body, in characters. Not a policy knob: the route layer
#: validates and sanitises the text long before this, and this exists so a direct
#: caller cannot put an unbounded blob in a table every future reader pages
#: through. Chosen to sit above the slot queue's per-entry budget so the inbox is
#: never the narrower path of the two.
MAX_BODY_CHARS = 64_000

#: Default page size for a read, and the ceiling on one. A reader that asks for
#: more gets this many; the rest stay unread for the next call, which is why the
#: claim is FIFO.
DEFAULT_READ_LIMIT = 20
MAX_READ_LIMIT = 100


@dataclass(frozen=True)
class InboxMessage:
    """One committed recado, as a reader receives it.

    Frozen because a claimed row is history: the reader may not edit what it was
    handed and hand a different thing onward.
    """

    id: str
    recipient_session_key: str
    sender_session_key: str
    body: str
    created_at: float
    read_at: float | None


def _row_to_message(row: tuple) -> InboxMessage:
    return InboxMessage(
        id=row[0],
        recipient_session_key=row[1],
        sender_session_key=row[2],
        body=row[3],
        created_at=row[4],
        read_at=row[5],
    )


def enqueue(
    conn: sqlite3.Connection,
    *,
    sender_session_key: str,
    recipient_session_key: str,
    body: str,
) -> InboxMessage:
    """Commit one unread row and return it. The caller wakes the recipient AFTER.

    Returns only once the row is durable, which is the ordering the module exists
    to guarantee — see the module docstring. The caller must not treat a wake
    failure as a send failure: the message is already stored, and the recipient
    will find it on its next read or its next wake.

    Every refusal is a :exc:`ValueError`, because the caller does one thing with
    all of them: reject the send rather than pretend a message was left. Identity
    is the caller's to establish; this function trusts the keys it is handed and
    only refuses shapes the table cannot hold meaningfully.
    """
    if not sender_session_key:
        raise ValueError("inbox send needs a sender session key")
    if not recipient_session_key:
        raise ValueError("inbox send needs a recipient session key")
    if sender_session_key == recipient_session_key:
        # A self-addressed recado would wake the sender with its own message and
        # is always a caller bug: a session that wants to remember something has
        # the session ledger, which is built for resumable state.
        raise ValueError(
            "inbox send addressed the sender itself; use the session ledger to "
            "carry state forward within one session"
        )
    if not body.strip():
        raise ValueError("inbox send needs a non-empty body")
    if len(body) > MAX_BODY_CHARS:
        raise ValueError(
            f"inbox body is {len(body)} characters, over the {MAX_BODY_CHARS} "
            "ceiling; send a pointer to an artifact instead of its contents"
        )

    message = InboxMessage(
        id=f"inbox_{uuid.uuid4().hex}",
        recipient_session_key=recipient_session_key,
        sender_session_key=sender_session_key,
        body=body,
        created_at=time.time(),
        read_at=None,
    )
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO agent_inbox_messages"
            "(id, recipient_session_key, sender_session_key, body, created_at, read_at) "
            "VALUES(?, ?, ?, ?, ?, NULL)",
            (
                message.id,
                message.recipient_session_key,
                message.sender_session_key,
                message.body,
                message.created_at,
            ),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return message


def read_unread(
    conn: sqlite3.Connection,
    *,
    recipient_session_key: str,
    limit: int = DEFAULT_READ_LIMIT,
) -> list[InboxMessage]:
    """Claim up to *limit* unread messages for one recipient, oldest first.

    The select and the ``read_at`` stamp share ONE transaction, so a row is
    claimed exactly once even with two readers on the same recipient — the
    property a plain "select, then update" would lose. The returned objects carry
    the ``read_at`` that was written, so a caller can record when it consumed
    them without a second query.

    The recipient is an argument here and NOT something this module infers,
    because inference belongs where identity is authenticated: the route layer
    derives it from the strict caller identity and never from a request field. A
    reader that could name its own recipient could read another session's mail.
    """
    if not recipient_session_key:
        raise ValueError("inbox read needs a recipient session key")
    bounded = max(1, min(int(limit), MAX_READ_LIMIT))

    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            "SELECT id, recipient_session_key, sender_session_key, body, created_at, read_at "
            "FROM agent_inbox_messages "
            "WHERE recipient_session_key = ? AND read_at IS NULL "
            "ORDER BY created_at, id LIMIT ?",
            (recipient_session_key, bounded),
        ).fetchall()
        if not rows:
            conn.execute("COMMIT")
            return []
        claimed_at = time.time()
        conn.executemany(
            "UPDATE agent_inbox_messages SET read_at = ? WHERE id = ?",
            [(claimed_at, row[0]) for row in rows],
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return [
        InboxMessage(
            id=row[0],
            recipient_session_key=row[1],
            sender_session_key=row[2],
            body=row[3],
            created_at=row[4],
            read_at=claimed_at,
        )
        for row in rows
    ]


def unread_count(conn: sqlite3.Connection, *, recipient_session_key: str) -> int:
    """How many messages wait for one recipient, without claiming any.

    What a wake payload carries instead of a body: it lets the trigger say "you
    have 3 messages" without the sender's text passing through a transcript twice.
    """
    if not recipient_session_key:
        return 0
    row = conn.execute(
        "SELECT COUNT(*) FROM agent_inbox_messages "
        "WHERE recipient_session_key = ? AND read_at IS NULL",
        (recipient_session_key,),
    ).fetchone()
    return int(row[0]) if row else 0
