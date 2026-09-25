"""``agent_inbox``: the commit-before-wake contract, pinned.

Every test here exists for one of the two properties the module is built on, and
neither is visible by reading the happy path:

* **Commit before wake.** :func:`enqueue` returns only once the row is durable, so
  a wake that never lands loses nothing. The tests prove the row is readable from
  a SECOND connection after enqueue returns — the only way to distinguish a real
  commit from an uncommitted transaction that merely looks fine to its own writer.
* **Claim, not peek.** :func:`read_unread` stamps ``read_at`` in the same
  transaction that selects, so one row reaches one reader. The tests prove a
  second read comes back empty and that two readers of one recipient partition
  the rows rather than duplicating them.

The storage is the task-queue database, so these tests build the real schema with
``migrate.apply_schema`` rather than a hand-written ``CREATE TABLE``: a test with
its own idea of the shape would keep passing after the shipped schema drifted.
"""

from __future__ import annotations

import sqlite3

import pytest

from kiro_crew.dashboard import agent_inbox
from kiro_crew.taskq import migrate

ALICE = "session:alice"
BOB = "session:bob"
CAROL = "session:carol"


@pytest.fixture
def db_path(tmp_path):
    """A real on-disk store: the commit tests need a second connection to it."""
    path = tmp_path / "tasks.db"
    conn = sqlite3.connect(path)
    try:
        migrate.apply_schema(conn)
    finally:
        conn.close()
    return path


@pytest.fixture
def conn(db_path):
    connection = sqlite3.connect(db_path)
    yield connection
    connection.close()


def _open(db_path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


class TestSchema:
    def test_table_and_index_exist(self, conn: sqlite3.Connection) -> None:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE name IN "
                "('agent_inbox_messages', 'agent_inbox_unread')"
            )
        }
        assert names == {"agent_inbox_messages", "agent_inbox_unread"}

    def test_apply_schema_is_idempotent(self, db_path) -> None:
        """A second open must do no work and must not lose rows."""
        first = _open(db_path)
        try:
            agent_inbox.enqueue(
                first, sender_session_key=ALICE, recipient_session_key=BOB, body="keep me"
            )
        finally:
            first.close()
        again = _open(db_path)
        try:
            migrate.apply_schema(again)
            assert agent_inbox.unread_count(again, recipient_session_key=BOB) == 1
        finally:
            again.close()

    def test_inbox_rows_are_not_tasks(self, conn: sqlite3.Connection) -> None:
        """A recado must never be leased, retried or counted against a lane."""
        agent_inbox.enqueue(
            conn, sender_session_key=ALICE, recipient_session_key=BOB, body="hello"
        )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


class TestCommitBeforeWake:
    def test_row_is_durable_when_enqueue_returns(self, db_path) -> None:
        writer = _open(db_path)
        try:
            message = agent_inbox.enqueue(
                writer, sender_session_key=ALICE, recipient_session_key=BOB, body="mail"
            )
        finally:
            writer.close()

        # A SEPARATE connection: an uncommitted write would be invisible here.
        reader = _open(db_path)
        try:
            claimed = agent_inbox.read_unread(reader, recipient_session_key=BOB)
        finally:
            reader.close()
        assert [m.id for m in claimed] == [message.id]
        assert claimed[0].body == "mail"
        assert claimed[0].sender_session_key == ALICE

    def test_unread_survives_reopening(self, db_path) -> None:
        """What a crash between commit and wake looks like: the mail is still there."""
        writer = _open(db_path)
        try:
            agent_inbox.enqueue(
                writer, sender_session_key=ALICE, recipient_session_key=BOB, body="survive"
            )
        finally:
            writer.close()
        reopened = _open(db_path)
        try:
            assert agent_inbox.unread_count(reopened, recipient_session_key=BOB) == 1
        finally:
            reopened.close()

    def test_enqueue_returns_unread(self, conn: sqlite3.Connection) -> None:
        message = agent_inbox.enqueue(
            conn, sender_session_key=ALICE, recipient_session_key=BOB, body="x"
        )
        assert message.read_at is None
        assert message.id.startswith("inbox_")


class TestClaimSemantics:
    def test_second_read_is_empty(self, conn: sqlite3.Connection) -> None:
        agent_inbox.enqueue(conn, sender_session_key=ALICE, recipient_session_key=BOB, body="one")
        assert len(agent_inbox.read_unread(conn, recipient_session_key=BOB)) == 1
        assert agent_inbox.read_unread(conn, recipient_session_key=BOB) == []

    def test_read_stamps_read_at(self, conn: sqlite3.Connection) -> None:
        agent_inbox.enqueue(conn, sender_session_key=ALICE, recipient_session_key=BOB, body="one")
        claimed = agent_inbox.read_unread(conn, recipient_session_key=BOB)
        assert claimed[0].read_at is not None
        stored = conn.execute(
            "SELECT read_at FROM agent_inbox_messages WHERE id = ?", (claimed[0].id,)
        ).fetchone()[0]
        assert stored == pytest.approx(claimed[0].read_at)

    def test_fifo_order(self, conn: sqlite3.Connection) -> None:
        for n in range(5):
            agent_inbox.enqueue(
                conn, sender_session_key=ALICE, recipient_session_key=BOB, body=f"m{n}"
            )
        assert [m.body for m in agent_inbox.read_unread(conn, recipient_session_key=BOB)] == [
            "m0",
            "m1",
            "m2",
            "m3",
            "m4",
        ]

    def test_limit_leaves_the_rest_unread(self, conn: sqlite3.Connection) -> None:
        for n in range(5):
            agent_inbox.enqueue(
                conn, sender_session_key=ALICE, recipient_session_key=BOB, body=f"m{n}"
            )
        first = agent_inbox.read_unread(conn, recipient_session_key=BOB, limit=2)
        assert [m.body for m in first] == ["m0", "m1"]
        assert agent_inbox.unread_count(conn, recipient_session_key=BOB) == 3
        assert [m.body for m in agent_inbox.read_unread(conn, recipient_session_key=BOB)] == [
            "m2",
            "m3",
            "m4",
        ]

    def test_two_readers_partition_rather_than_duplicate(self, db_path) -> None:
        """One row reaches one reader, which a select-then-update would lose."""
        writer = _open(db_path)
        try:
            for n in range(4):
                agent_inbox.enqueue(
                    writer, sender_session_key=ALICE, recipient_session_key=BOB, body=f"m{n}"
                )
        finally:
            writer.close()

        one, two = _open(db_path), _open(db_path)
        try:
            first = agent_inbox.read_unread(one, recipient_session_key=BOB, limit=2)
            second = agent_inbox.read_unread(two, recipient_session_key=BOB, limit=2)
        finally:
            one.close()
            two.close()
        ids = [m.id for m in first] + [m.id for m in second]
        assert len(ids) == 4
        assert len(set(ids)) == 4, "a row was handed to both readers"

    def test_limit_is_bounded_and_floored(self, conn: sqlite3.Connection) -> None:
        for n in range(3):
            agent_inbox.enqueue(
                conn, sender_session_key=ALICE, recipient_session_key=BOB, body=f"m{n}"
            )
        assert len(agent_inbox.read_unread(conn, recipient_session_key=BOB, limit=0)) == 1
        assert len(agent_inbox.read_unread(conn, recipient_session_key=BOB, limit=10_000)) == 2


class TestIsolation:
    def test_recipients_do_not_see_each_others_mail(self, conn: sqlite3.Connection) -> None:
        agent_inbox.enqueue(
            conn, sender_session_key=ALICE, recipient_session_key=BOB, body="for bob"
        )
        agent_inbox.enqueue(
            conn, sender_session_key=ALICE, recipient_session_key=CAROL, body="for carol"
        )
        assert [m.body for m in agent_inbox.read_unread(conn, recipient_session_key=BOB)] == [
            "for bob"
        ]
        assert [m.body for m in agent_inbox.read_unread(conn, recipient_session_key=CAROL)] == [
            "for carol"
        ]

    def test_claiming_one_recipient_leaves_the_other_unread(self, conn: sqlite3.Connection) -> None:
        agent_inbox.enqueue(conn, sender_session_key=ALICE, recipient_session_key=BOB, body="b")
        agent_inbox.enqueue(conn, sender_session_key=ALICE, recipient_session_key=CAROL, body="c")
        agent_inbox.read_unread(conn, recipient_session_key=BOB)
        assert agent_inbox.unread_count(conn, recipient_session_key=CAROL) == 1

    def test_unread_count_of_stranger_is_zero(self, conn: sqlite3.Connection) -> None:
        agent_inbox.enqueue(conn, sender_session_key=ALICE, recipient_session_key=BOB, body="b")
        assert agent_inbox.unread_count(conn, recipient_session_key="session:nobody") == 0


class TestRefusals:
    """Each refusal is a shape the table could hold but no reader could act on."""

    def test_empty_sender_refused(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="sender"):
            agent_inbox.enqueue(
                conn, sender_session_key="", recipient_session_key=BOB, body="x"
            )

    def test_empty_recipient_refused(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="recipient"):
            agent_inbox.enqueue(
                conn, sender_session_key=ALICE, recipient_session_key="", body="x"
            )

    def test_self_addressed_refused(self, conn: sqlite3.Connection) -> None:
        """It would wake the sender with its own message; the ledger is for that."""
        with pytest.raises(ValueError, match="ledger"):
            agent_inbox.enqueue(
                conn, sender_session_key=ALICE, recipient_session_key=ALICE, body="x"
            )

    @pytest.mark.parametrize("body", ["", "   ", "\n\t "])
    def test_blank_body_refused(self, conn: sqlite3.Connection, body: str) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            agent_inbox.enqueue(
                conn, sender_session_key=ALICE, recipient_session_key=BOB, body=body
            )

    def test_oversized_body_refused(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="ceiling"):
            agent_inbox.enqueue(
                conn,
                sender_session_key=ALICE,
                recipient_session_key=BOB,
                body="x" * (agent_inbox.MAX_BODY_CHARS + 1),
            )

    def test_body_at_the_ceiling_is_accepted(self, conn: sqlite3.Connection) -> None:
        agent_inbox.enqueue(
            conn,
            sender_session_key=ALICE,
            recipient_session_key=BOB,
            body="x" * agent_inbox.MAX_BODY_CHARS,
        )
        assert agent_inbox.unread_count(conn, recipient_session_key=BOB) == 1

    def test_a_refused_send_writes_nothing(self, conn: sqlite3.Connection) -> None:
        for kwargs in (
            {"sender_session_key": "", "recipient_session_key": BOB, "body": "x"},
            {"sender_session_key": ALICE, "recipient_session_key": BOB, "body": ""},
            {"sender_session_key": ALICE, "recipient_session_key": ALICE, "body": "x"},
        ):
            with pytest.raises(ValueError):
                agent_inbox.enqueue(conn, **kwargs)
        assert conn.execute("SELECT COUNT(*) FROM agent_inbox_messages").fetchone()[0] == 0

    def test_empty_recipient_read_refused(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="recipient"):
            agent_inbox.read_unread(conn, recipient_session_key="")
