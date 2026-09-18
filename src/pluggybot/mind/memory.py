"""The robot's memory as RECORDS (issue #221): one SQLite store per robot,
unbounded and append-only, from which every document the robot reads is
a VIEW.

The memory has four tiers (docs/Overseer.md §7): the constitution (a
human's file, not here), the core (`Goals.md`, `Top_of_mind.md` -- always
shown), the notes (topics the robot names; the index always shown, a body
by `recall`) and the history (what happened, the system's and the
senders'; the tail shown, the rest by `recall`). Every line in the last
three is a row here. `mind/thoughts.py` owns the views and the verbs and
asks this module for nothing but rows; the registry (`mind/text.py`) owns
the caps and the writers and is consulted BEFORE a row is added.

⚠ NOTHING IS DELETED. `retire` marks a row and `recall` can still find
it; a true death (`archive`) moves the robot on to a new GENERATION and
the old rows stay on the volume where the operator looks. Only views and
searches are scoped to the living generation, so the roll-vs-refuse
question is answered once: the store never rolls, the VIEW is capped.

`sqlite3` is stdlib and FTS5 is compiled into python:3.12-slim's build
(measured: 3.46.1 in the image, 3.53.1 here), so the serving image's
six-package rule holds. `:memory:` is the test double -- the same code
path, which is the point of not having two implementations.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

#: What a row IS. `core` is a line of an always-shown document, `note` a
#: titled line in a topic, `history` a line of the narrative record,
#: `think` the scratch the model wrote before a decision (issue #221;
#: a history line by any other name, kept apart so a tail of History is
#: not twelve thinks).
KINDS = ("core", "note", "history", "think")
ACTIVE, RETIRED = "active", "retired"
#: Most rows a search returns. Small: a recalled block is paid for on the
#: next turn (`thoughts.RECALLED_CHARS` is the byte cap), and eight good
#: hits beat forty.
FIND_LIMIT = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
  id INTEGER PRIMARY KEY,
  robot TEXT NOT NULL,
  generation INTEGER NOT NULL DEFAULT 1,
  t REAL NOT NULL DEFAULT 0,
  at TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL,
  writer TEXT NOT NULL,
  topic TEXT NOT NULL DEFAULT '',
  title TEXT NOT NULL DEFAULT '',
  text TEXT NOT NULL,
  fields TEXT NOT NULL DEFAULT '{}',
  cites TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'active',
  retired_t REAL
);
CREATE INDEX IF NOT EXISTS records_view ON records (robot, generation, kind, status, id);
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
  text, title, topic, content='records', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON records BEGIN
  INSERT INTO records_fts(rowid, text, title, topic)
  VALUES (new.id, new.text, new.title, new.topic);
END;
CREATE TABLE IF NOT EXISTS generations (
  robot TEXT PRIMARY KEY,
  generation INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class Record:
  id: int
  robot: str
  generation: int
  t: float
  at: str
  kind: str
  writer: str
  topic: str
  title: str
  text: str
  fields: dict
  cites: tuple[int, ...]
  status: str
  retired_t: float | None

  @property
  def active(self) -> bool:
    return self.status == ACTIVE

  def as_dict(self) -> dict:
    return {"id": self.id, "t": self.t, "kind": self.kind, "writer": self.writer,
            "topic": self.topic, "title": self.title, "text": self.text,
            "fields": dict(self.fields), "cites": list(self.cites),
            "status": self.status}


def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _row(r: sqlite3.Row) -> Record:
  return Record(id=r["id"], robot=r["robot"], generation=r["generation"],
                t=r["t"], at=r["at"], kind=r["kind"], writer=r["writer"],
                topic=r["topic"], title=r["title"], text=r["text"],
                fields=json.loads(r["fields"]),
                cites=tuple(json.loads(r["cites"])), status=r["status"],
                retired_t=r["retired_t"])


def fts_query(text: str) -> str:
  """A model's free text as an FTS5 query: every word quoted, OR-joined.

  Quoted, because FTS5's own syntax (`AND`, `NEAR`, a stray `"`) would
  turn an ordinary sentence into a syntax error; OR-joined, because a
  small model's query is a handful of words and demanding all of them
  returns nothing -- bm25 already ranks a row matching three of four above
  one matching one.
  """
  words = [w for w in re.findall(r"[\w']+", text or "") if w.strip("_'")]
  return " OR ".join(f'"{w}"' for w in words[:16])


class RecordStore:
  """One robot's rows -- or several robots' in one file, scoped by
  `robot` on every call (a pair shares nothing here and keeps one file
  each, but the schema does not care)."""

  def __init__(self, path: str = ":memory:", clock=_now) -> None:
    self.path = path
    self.clock = clock
    # ⚠ ONE THREAD. Writes come from the physics thread (`_reconsider`, the
    # lifecycle's `_remember`) and reads from the same; the overseer's
    # worker thread builds prompts from what `thoughts.volatile()` already
    # rendered, never from here.
    self.db = sqlite3.connect(path)
    self.db.row_factory = sqlite3.Row
    # WAL, and an fsync per CHECKPOINT rather than per commit. Measured on
    # this box: the default (`synchronous=FULL`, rollback journal) cost
    # ~100 ms a write -- 77 s for a test that writes 800 lines -- and the
    # robot writes on every decision. A crash can lose the last unsynced
    # commits and never the store's consistency, which is the trade the
    # boards and the ledger already make with an unsynced `os.replace`.
    if path != ":memory:":
      self.db.execute("PRAGMA journal_mode=WAL")
      self.db.execute("PRAGMA synchronous=NORMAL")
    self.db.executescript(_SCHEMA)

  def close(self) -> None:
    self.db.close()

  # ---- generations (a true death, issue #136) ---------------------------------

  def generation(self, robot: str) -> int:
    row = self.db.execute("SELECT generation FROM generations WHERE robot = ?",
                          (robot,)).fetchone()
    return int(row["generation"]) if row else 1

  def archive(self, robot: str, t: float = 0.0) -> int:
    """Move `robot` on to the next generation. The rows stay; views and
    searches no longer see them. Returns how many rows were put aside."""
    gen = self.generation(robot)
    n = self.db.execute(
      "SELECT COUNT(*) FROM records WHERE robot = ? AND generation = ?",
      (robot, gen)).fetchone()[0]
    self.db.execute(
      "INSERT INTO generations (robot, generation) VALUES (?, ?) "
      "ON CONFLICT(robot) DO UPDATE SET generation = excluded.generation",
      (robot, gen + 1))
    self.db.commit()
    return int(n)

  # ---- writing --------------------------------------------------------------

  def add(self, robot: str, kind: str, writer: str, text: str, t: float = 0.0,
          topic: str = "", title: str = "", fields: dict | None = None,
          cites=()) -> Record:
    assert kind in KINDS, kind
    cur = self.db.execute(
      "INSERT INTO records (robot, generation, t, at, kind, writer, topic, "
      "title, text, fields, cites) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
      (robot, self.generation(robot), float(t), self.clock(), kind, writer,
       topic, title, text, json.dumps(fields or {}, sort_keys=True),
       json.dumps([int(c) for c in cites])))
    self.db.commit()
    return self.get(int(cur.lastrowid))

  def retire(self, id: int, t: float = 0.0) -> Record | None:
    """Mark one row retired. Retiring what is retired, or absent, changes
    nothing and returns what is there."""
    self.db.execute(
      "UPDATE records SET status = ?, retired_t = ? WHERE id = ? AND status = ?",
      (RETIRED, float(t), int(id), ACTIVE))
    self.db.commit()
    return self.get(id)

  # ---- reading --------------------------------------------------------------

  def get(self, id: int) -> Record | None:
    row = self.db.execute("SELECT * FROM records WHERE id = ?", (int(id),)).fetchone()
    return _row(row) if row else None

  def active(self, robot: str, kind: str, topic: str | None = None,
             prefix: str | None = None) -> list[Record]:
    """The living generation's active rows of one kind, oldest first --
    what a view is rendered from. `topic` narrows to one topic, `prefix`
    to a family (`findings/`)."""
    sql = ("SELECT * FROM records WHERE robot = ? AND generation = ? "
           "AND kind = ? AND status = ?")
    args: list = [robot, self.generation(robot), kind, ACTIVE]
    if topic is not None:
      sql += " AND topic = ?"
      args.append(topic)
    if prefix is not None:
      sql += " AND topic LIKE ?"
      args.append(prefix.replace("%", "") + "%")
    sql += " ORDER BY id"
    return [_row(r) for r in self.db.execute(sql, args)]

  def tail(self, robot: str, kind: str, n: int) -> list[Record]:
    """The newest `n` active rows of a kind, oldest first."""
    rows = self.db.execute(
      "SELECT * FROM records WHERE robot = ? AND generation = ? AND kind = ? "
      "AND status = ? ORDER BY id DESC LIMIT ?",
      (robot, self.generation(robot), kind, ACTIVE, int(n))).fetchall()
    return [_row(r) for r in reversed(rows)]

  def topics(self, robot: str, kind: str = "note") -> dict[str, list[Record]]:
    """The index: every topic of the living generation with its active rows."""
    out: dict[str, list[Record]] = {}
    for rec in self.active(robot, kind):
      out.setdefault(rec.topic, []).append(rec)
    return out

  def find(self, robot: str, query: str, kind: str | None = None,
           since: float | None = None, writer: str | None = None,
           limit: int = FIND_LIMIT) -> list[Record]:
    """Full-text search over the living generation, retired rows included
    (a retired row is what `recall` is FOR: the robot took it off the page,
    not out of the world). Best match first; ties to the newest."""
    q = fts_query(query)
    if not q:
      return []
    sql = ("SELECT r.* FROM records_fts f JOIN records r ON r.id = f.rowid "
           "WHERE records_fts MATCH ? AND r.robot = ? AND r.generation = ?")
    args: list = [q, robot, self.generation(robot)]
    if kind is not None:
      sql += " AND r.kind = ?"
      args.append(kind)
    if since is not None:
      sql += " AND r.t >= ?"
      args.append(float(since))
    if writer is not None:
      sql += " AND r.writer = ?"
      args.append(writer)
    sql += " ORDER BY bm25(records_fts), r.id DESC LIMIT ?"
    args.append(int(limit))
    return [_row(r) for r in self.db.execute(sql, args)]

  def count(self, robot: str, kind: str | None = None,
            status: str | None = None) -> int:
    sql = "SELECT COUNT(*) FROM records WHERE robot = ? AND generation = ?"
    args: list = [robot, self.generation(robot)]
    if kind is not None:
      sql += " AND kind = ?"
      args.append(kind)
    if status is not None:
      sql += " AND status = ?"
      args.append(status)
    return int(self.db.execute(sql, args).fetchone()[0])
