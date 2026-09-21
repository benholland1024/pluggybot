"""Support tickets (issue #284): the robot tells the people who run its
world what is wrong with it, what it would like, what it does not
understand -- and they answer.

A ticket is a DOCUMENT in `mind/text.py`'s registry (the `tickets` row):
one JSON record per ticket under `$PLUGGY_THOUGHTS/tickets/`, written
through the store like every other document, the robot's own words in it
and the thread it grew. The robot opens one with the `ticket` decision
field (a `kind` off `TICKET_KINDS`, a title, a report) and answers on a
thread with `ticket_reply`; both are paperwork on `pin`'s terms and cost
no turn. What happens to a ticket after that is the OPERATOR's: a reply
lands on the thread (`ticket_reply`, inbound), a close ends it with a
message (`ticket_close`) and PAYS -- the reward table's `ticket` row,
banked once, through `scoring.evaluate` and `Ledger.award` like every
other payout -- and a delete erases it (`ticket_delete`). The robot can
neither close nor withdraw one, so a full desk is the operator's to
clear, and the refusal says so.

Rules, each pinned in `tests/test_tickets.py`:

  THE CAP COUNTS OPEN TICKETS (`MAX_OPEN_TICKETS`, the registry row's, at
  `text.admit`): a closed ticket stays in the record for the robot to
  read what came of it and costs no slot; a deleted one is gone.

  IDS NEVER REPEAT. The counter is its own record (`counter.json`), not
  the highest id in the directory: a deleted ticket would otherwise hand
  its id to the next one, and the website keys a ticket by (robot, id).
  The desk survives a restart and a true death both -- a ticket is a
  report about the WORLD, and the next robot inherits the world.

  A REPLAYED CLOSE PAYS NOTHING. The website re-sends a close it never
  saw acknowledged; `close` on a closed ticket answers the same ticket,
  `paid` False, and the lifecycle re-emits the acknowledgement with the
  points that were banked the first time.

  NOTHING HERE MOVES A BALANCE. The desk records what the ledger paid
  (`pay`), it does not pay: the payout is `scoring.evaluate("ticket")`'s
  in the lifecycle, off the table, like every other.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from pluggybot.mind import text as registry
from pluggybot.mind.store import FileStore, MemoryStore, Store
from pluggybot.telemetry.protocol import TICKET_KINDS, TICKET_SENDERS

ROW = registry.BY_NAME["tickets"]
SUFFIX = ROW.suffix
#: Open tickets at once: the registry row's cap (`text.MAX_OPEN_TICKETS`).
MAX_OPEN = ROW.cap
#: A report's length (`text.MAX_TICKET_CHARS`); a title's; a line of the
#: thread's, in either direction -- a message's cap, the operator row's.
MAX_TEXT = registry.MAX_TICKET_CHARS
MAX_TITLE = 80
MAX_LINE = registry.BY_NAME["operator"].cap
#: Lines kept on a thread (the newest), and how many of them the context
#: shows. A thread is a conversation about one thing; eight lines of it
#: is the record and four is what the robot needs in front of it.
MAX_THREAD = 8
THREAD_SHOWN = 4
#: Closed tickets kept in the record (the newest), and how many the
#: context shows with their closing message -- what came of asking is
#: the one thing that teaches what is worth asking.
MAX_CLOSED = 20
CLOSED_SHOWN = 3
#: The counter's record: the one key under the store that is not a ticket.
COUNTER = "counter.json"

ROBOT, OPERATOR = TICKET_SENDERS
STATES = ("open", "closed")


class DeskRefused(ValueError):
  """The desk would not take it, and says why -- narrated, on the wire as
  a `ticket` event with outcome `refused`, never swallowed."""


def _clean(value: object, limit: int) -> str:
  """One line, capped -- the inbox's normaliser, so a ticket cannot forge a
  narration line either. Not a security boundary (mind/inbox.py)."""
  from pluggybot.mind.inbox import clean
  return clean(value, limit)


@dataclass
class Line:
  sender: str                        # `robot` / `operator`
  who: str                           # the display name
  text: str
  t: float

  def as_dict(self) -> dict:
    return {"sender": self.sender, "from": self.who, "text": self.text,
            "t": round(float(self.t), 3)}


@dataclass
class Ticket:
  id: str
  kind: str
  title: str
  text: str
  t: float
  state: str = "open"
  thread: list[Line] = field(default_factory=list)
  closed_by: str = ""
  closed_text: str = ""
  closed_t: float | None = None
  #: What the ledger paid at the close, and the entry's `seq` -- recorded
  #: by `pay`, so a replayed close can answer with the same figure.
  points: int | None = None
  seq: int | None = None

  @property
  def open(self) -> bool:
    return self.state == "open"

  def as_record(self) -> dict:
    return {"id": self.id, "kind": self.kind, "title": self.title,
            "text": self.text, "t": round(float(self.t), 3), "state": self.state,
            "thread": [line.as_dict() for line in self.thread],
            "closedBy": self.closed_by, "closedText": self.closed_text,
            "closedT": self.closed_t, "points": self.points, "seq": self.seq}

  @classmethod
  def from_record(cls, rec: dict) -> "Ticket":
    thread = [Line(str(x.get("sender", OPERATOR)), str(x.get("from", "")),
                   str(x.get("text", "")), float(x.get("t", 0.0)))
              for x in rec.get("thread", ()) if isinstance(x, dict)]
    return cls(id=str(rec["id"]), kind=str(rec.get("kind", "")),
               title=str(rec.get("title", "")), text=str(rec.get("text", "")),
               t=float(rec.get("t", 0.0)),
               state=str(rec.get("state", "open")), thread=thread,
               closed_by=str(rec.get("closedBy", "")),
               closed_text=str(rec.get("closedText", "")),
               closed_t=rec.get("closedT"),
               points=rec.get("points"), seq=rec.get("seq"))

  def as_dict(self) -> dict:
    """The wire's shape: the record, whole (the `tickets` message)."""
    return self.as_record()

  def as_context(self) -> dict:
    """How the robot is shown it: its own report, the thread's newest
    `THREAD_SHOWN` lines each labelled with who wrote it (a report of
    what somebody said, never a turn), and -- closed -- who closed it,
    with what words, and what it paid."""
    out = {"id": self.id, "kind": self.kind, "title": self.title,
           "text": self.text, "openedAtS": round(float(self.t), 1),
           "thread": [line.as_dict() for line in self.thread[-THREAD_SHOWN:]]}
    if not self.open:
      out.update({"closedBy": self.closed_by, "closedWith": self.closed_text,
                  "points": self.points})
    return out


class Desk:
  """The robot's tickets, open and recently closed, and the counter."""

  def __init__(self, root: str | os.PathLike | None = None) -> None:
    #: The one path to the disk (issue #217): a `Store`, the volume's or
    #: memory's.
    self.store: Store = FileStore(root) if root is not None else MemoryStore()
    self.tickets: dict[str, Ticket] = {}
    self.seq = 0
    self.refusals: list[dict] = []
    self._load()

  @property
  def root(self):
    return self.store.root if isinstance(self.store, FileStore) else None

  # ---- the record ---------------------------------------------------------------

  def _load(self) -> None:
    for key in self.store.keys(suffix=SUFFIX):
      try:
        rec = json.loads(self.store.read(key) or "")
        ticket = Ticket.from_record(rec)
      except (ValueError, KeyError, TypeError):
        continue
      if ticket.kind in TICKET_KINDS and ticket.state in STATES:
        self.tickets[ticket.id] = ticket
    try:
      self.seq = int(json.loads(self.store.read(COUNTER) or "{}").get("seq", 0))
    except (ValueError, TypeError):
      self.seq = 0
    # A record older than its counter (a counter file lost, a directory
    # copied by hand): never hand out an id that is already taken.
    for ticket in self.tickets.values():
      try:
        self.seq = max(self.seq, int(ticket.id.split("_")[-1]))
      except ValueError:
        pass

  def _save(self, ticket: Ticket) -> None:
    self.store.write(f"{ticket.id}{SUFFIX}",
                     json.dumps(ticket.as_record(), indent=1, sort_keys=True) + "\n")

  def _save_counter(self) -> None:
    self.store.write(COUNTER, json.dumps({"seq": self.seq}) + "\n")

  def get(self, ticket_id: str) -> Ticket | None:
    return self.tickets.get(str(ticket_id or ""))

  def open_tickets(self) -> list[Ticket]:
    """Open, oldest first."""
    return sorted((t for t in self.tickets.values() if t.open), key=lambda t: t.t)

  def closed_tickets(self) -> list[Ticket]:
    """Closed, newest close first."""
    return sorted((t for t in self.tickets.values() if not t.open),
                  key=lambda t: (t.closed_t or 0.0), reverse=True)

  def open_ids(self) -> tuple[str, ...]:
    return tuple(t.id for t in self.open_tickets())

  # ---- the robot's two verbs ------------------------------------------------------

  def open(self, kind: str, title: str, text: str, t: float) -> Ticket:
    """File a ticket, or `DeskRefused`. The cap is the registry's, through
    the one gate every document write passes."""
    kind = str(kind or "").strip()
    title = _clean(title, MAX_TITLE)
    text = _clean(text, MAX_TEXT)
    reasons = []
    if kind not in TICKET_KINDS:
      reasons.append(f"a ticket's kind is one of {', '.join(TICKET_KINDS)}, "
                     f"not {kind!r}")
    if not text:
      reasons.append("a ticket says something: `text` is empty")
    if reasons:
      raise self._refuse(reasons)
    try:
      registry.admit(ROW, registry.ROBOT, len(self.open_tickets()) + 1)
    except registry.Refused:
      raise self._refuse([f"you have {MAX_OPEN} tickets open, which is the "
                          "most at once; one has to be closed or deleted "
                          "by the people who run the world before another "
                          "will be taken"]) from None
    self.seq += 1
    ticket = Ticket(id=f"tk_{self.seq:04d}", kind=kind,
                    title=title or text[:MAX_TITLE], text=text, t=float(t))
    self.tickets[ticket.id] = ticket
    self._save_counter()
    self._save(ticket)
    return ticket

  def reply(self, ticket_id: str, text: str, t: float,
            sender: str = ROBOT, who: str = "") -> Ticket:
    """A line on an OPEN ticket's thread, from the robot or the operator
    (`sender` is stated by the caller, never read off the wire). The
    thread keeps its newest `MAX_THREAD` lines."""
    ticket = self.get(ticket_id)
    text = _clean(text, MAX_LINE)
    if ticket is None:
      raise self._refuse([f"no ticket {ticket_id!r} on the desk"])
    if not ticket.open:
      raise self._refuse([f"ticket {ticket.id} is closed; open a new one"])
    if not text:
      raise self._refuse(["a reply says something: `text` is empty"])
    if sender not in TICKET_SENDERS:
      raise ValueError(f"a thread line's sender is one of {TICKET_SENDERS}")
    ticket.thread.append(Line(sender, _clean(who, 40), text, float(t)))
    del ticket.thread[:-MAX_THREAD]
    self._save(ticket)
    return ticket

  # ---- the operator's two -------------------------------------------------------------

  def close(self, ticket_id: str, by: str, text: str,
            t: float) -> tuple[Ticket | None, bool]:
    """End a ticket with a message. `(ticket, changed)`: None for a ticket
    the desk does not hold; `changed` False for one already closed -- a
    replayed close, answered with the record and paid nothing."""
    ticket = self.get(ticket_id)
    if ticket is None:
      return None, False
    if not ticket.open:
      return ticket, False
    ticket.state = "closed"
    ticket.closed_by = _clean(by, 40)
    ticket.closed_text = _clean(text, MAX_LINE)
    ticket.closed_t = round(float(t), 3)
    self._save(ticket)
    self._trim_closed()
    return ticket, True

  def pay(self, ticket_id: str, points: int, seq: int) -> None:
    """Record what the ledger paid at the close (the lifecycle's figure)."""
    ticket = self.get(ticket_id)
    if ticket is None:
      return
    ticket.points, ticket.seq = int(points), int(seq)
    self._save(ticket)

  def delete(self, ticket_id: str) -> Ticket | None:
    """Erase a ticket, open or closed. None for one the desk does not hold."""
    ticket = self.tickets.pop(str(ticket_id or ""), None)
    if ticket is not None:
      self.store.remove(f"{ticket.id}{SUFFIX}")
    return ticket

  def _trim_closed(self) -> None:
    for old in self.closed_tickets()[MAX_CLOSED:]:
      self.tickets.pop(old.id, None)
      self.store.remove(f"{old.id}{SUFFIX}")

  def _refuse(self, reasons: list[str]) -> DeskRefused:
    # The newest few, for `stats()`: the record of a refusal is the
    # `ticket` event the lifecycle emits, not this list.
    self.refusals.append({"reasons": list(reasons)})
    del self.refusals[:-50]
    return DeskRefused("; ".join(reasons))

  # ---- what the robot and the wire are shown -------------------------------------------

  def as_context(self) -> dict:
    """The `tickets` block of the user turn: every open ticket with its
    thread, the newest closed ones with what came of them, and how many
    more may be opened."""
    opened = self.open_tickets()
    return {"open": [t.as_context() for t in opened],
            "closed": [t.as_context() for t in self.closed_tickets()[:CLOSED_SHOWN]],
            "slotsLeft": max(0, MAX_OPEN - len(opened))}

  def snapshot(self, t: float, robot: str) -> dict:
    """The `tickets` message a stream opens with: the open tickets, whole."""
    return {"type": "tickets", "t": round(float(t), 3), "robot": robot,
            "tickets": [ticket.as_dict() for ticket in self.open_tickets()]}

  def stats(self) -> dict:
    return {"open": len(self.open_tickets()),
            "closed": len(self.closed_tickets()),
            "refused": len(self.refusals), "seq": self.seq}
