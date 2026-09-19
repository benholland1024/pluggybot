"""The library: the robot reads Wikipedia, for ideas rather than answers
(issue #216).

EXPOSURE, not a test. A test of "find an obscure fact" measures the model
and the search tool, not the embodied agent. This is a way for the robot
to meet ideas from outside its world -- something to think about, to talk
about with visitors and the other robot, to draw, to turn into a goal --
and the metric is whether an idea can be TRACED from a lookup into any of
those (`evaluation/qualities.py`, `ideas_traced`). It feeds quality five
(goal creation) and quality four (creativity), and it is offered on the
`autonomous` arm alone, so `guarded`'s prefix is unchanged.

Four rules, and they are the whole design:

  CODE DOES THE LOOKUP. The robot sets one decision field, `lookup` (a
  topic or a question), and this module fetches ONE page's summary from
  Wikipedia's REST API -- never the open web, never anything the robot
  names as a URL. `read` is `recall`'s key and was taken by the time this
  landed, which is why the field is `lookup`.

  THE RESULT IS A MESSAGE. What comes back is delivered on the visitor
  channel's terms (`mind/text.py`'s `library` row, `mind/inbox.py`'s
  framing): a labelled block, sender "the library", information and never
  an instruction. Wikipedia is editable by anyone, so the framing is what
  protects the body -- exactly as it does for a visitor who pretends to be
  the operator -- and the model's only output stays the fixed vocabulary.

  RATIONED LIKE ESCALATION. Each read is a turn's worth of tokens on the
  next call, and a robot that could read freely would spend its day
  reading. An interval and a share of decisions (`why_not`, the escalation
  gate's shape), and points pay the throttle off; there is no money here
  to keep points away from.

  DETERMINISM IS NOT A GOAL. A page changes over time and that is fine --
  this is not a graded measurement, and an idea is an idea whichever
  revision it came from. The revision id is recorded so a traced idea has
  a source.

Stdlib `urllib`, for the reason `mind/llm.py` gives: the serving image
installs six pinned packages and this is not one more.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from typing import Callable

from pluggybot.mind import text as registry
from pluggybot.mind.inbox import VisitorMessage, clean
from pluggybot.telemetry.protocol import READ_OUTCOMES

#: Who the page arrives from: the display name in the block the model is
#: shown, the registry row's sender, and what the narration says.
SENDER = "the library"
#: The summary endpoint: one page, its first paragraph, its revision id.
#: `{title}` is URL-quoted with spaces as underscores, and the endpoint
#: follows redirects itself ("durian" -> "Durian").
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
#: ...and the one fallback when a query is not a title ("why is the sky
#: blue"): a title search, then the summary of its first hit that is a
#: page rather than a list of pages. MEASURED: "Mercury" is a
#: disambiguation page and the search's first hit is that same page, so
#: the first hit alone handed back a list of titles and no idea; three
#: hits are asked for and a disambiguation is skipped. One page delivered.
SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page?q={q}&limit=3"
#: Wikimedia's API policy asks for a descriptive User-Agent with a contact;
#: a generic one is throttled or refused.
USER_AGENT = "PluggyBot/1.0 (https://rooftop-media.org/pluggyworld) python-urllib"
#: Wall seconds one request may take. Inside the decision's own deadline
#: (`overseer.CALL_TIMEOUT_S` 90), because the read runs on the decision's
#: worker thread after the answer parsed: a slow Wikipedia costs the robot
#: a longer pause, never the world a freeze.
TIMEOUT_S = 10.0
#: Longest query kept, in characters. A topic or a question, one line.
MAX_QUERY_CHARS = 200
#: Longest extract delivered: the registry row's cap (`mind/text.py`). A
#: summary's first paragraph is typically 300-900 characters; the cap is
#: the prompt's, not Wikipedia's.
MAX_PAGE_CHARS = registry.BY_NAME["library"].cap

#: THE THROTTLE, on the escalation's shape exactly (`overseer.ESCALATE_
#: SHARE` / `ESCALATE_MIN_INTERVAL_S`): about one read in ten decisions,
#: never two inside ten minutes. Sized against a day of ~100 decisions
#: (Evaluation.md §3's baseline), so a robot reads a handful of pages a day
#: rather than a shelf. The share warms up as escalation's does: the FIRST
#: read is always allowed.
LOOKUP_SHARE = 0.10
LOOKUP_MIN_INTERVAL_S = 600.0
#: ...and what paying the throttle off costs, in points. Below the
#: escalation's 15 because a read costs nobody money; above nothing
#: because a read that cost nothing would not be rationed.
LOOKUP_POINTS = 10

#: Why a read did not happen, when it did not. `too-soon` and `share` are
#: the throttle; `timeout` / `offline` / `garbled` are the transport, on
#: the fallback reasons' names so a reader of both sees one vocabulary.
REFUSALS = ("too-soon", "share")
FAILURES = ("timeout", "offline", "garbled")


class Page:
  """One page's summary, as fetched. `revision` is the source a traced idea
  is attributed to; `url` is where a person can read what the robot read."""

  __slots__ = ("query", "title", "extract", "revision", "url", "cut")

  def __init__(self, query: str, title: str, extract: str, revision: str,
               url: str) -> None:
    self.query = query
    self.title = title
    text = clean(extract, MAX_PAGE_CHARS + 1)
    self.cut = len(text) > MAX_PAGE_CHARS
    self.extract = text[:MAX_PAGE_CHARS]
    self.revision = str(revision)
    self.url = url


def fetch_json(url: str, timeout: float = TIMEOUT_S) -> dict | None:
  """GET one JSON document. None on a 404 (no such page); raises on any
  other failure, which `lookup` maps to a `FAILURES` word."""
  req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                             "Accept": "application/json"})
  try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      return json.loads(resp.read().decode("utf-8"))
  except urllib.error.HTTPError as e:
    if e.code == 404:
      return None
    raise


def _title_url(query: str) -> str:
  return SUMMARY_URL.format(title=urllib.parse.quote(query.replace(" ", "_"), safe=""))


def _is_page(doc: dict | None) -> bool:
  """A summary with something to read: not a 404, not a disambiguation (a
  list of titles and no idea), not an empty extract."""
  return bool(doc) and doc.get("type") != "disambiguation" and bool(doc.get("extract"))


def lookup(query: str, fetch: Callable[[str], dict | None] = fetch_json) -> Page | None:
  """The query as a title first, then as a search; None when neither names
  a page. `fetch` is the injection seam -- a test hands in a dict of URLs.
  """
  doc = fetch(_title_url(query))
  if not _is_page(doc):
    found = fetch(SEARCH_URL.format(q=urllib.parse.quote(query, safe="")))
    doc = None
    for hit in (found or {}).get("pages") or []:
      if not hit.get("key"):
        continue
      candidate = fetch(_title_url(str(hit["key"]).replace("_", " ")))
      if _is_page(candidate):
        doc = candidate
        break
    if doc is None:
      return None
  return Page(query=query, title=str(doc.get("title") or query),
              extract=str(doc["extract"]),
              revision=str(doc.get("revision") or ""),
              url=str(((doc.get("content_urls") or {}).get("desktop") or {})
                      .get("page") or ""))


def failure(e: BaseException) -> str:
  """A transport exception -> one of `FAILURES`."""
  if isinstance(e, (socket.timeout, TimeoutError)):
    return "timeout"
  if isinstance(e, (json.JSONDecodeError, KeyError, ValueError, TypeError)):
    return "garbled"
  return "offline"


class Wiki:
  """The library's desk: the throttle, the fetch, and the record of every
  read. One per mind, held by the `Overseer` on `autonomous` alone.

  `ledger` is what pays the throttle off; None means a refusal stands.
  `clock` is injected as `Overseer.clock` is, so a test need not wait ten
  minutes to see the interval bite.
  """

  def __init__(self, fetch: Callable[[str], dict | None] = fetch_json,
               ledger=None, clock: Callable[[], float] = time.monotonic,
               share: float = LOOKUP_SHARE,
               interval_s: float = LOOKUP_MIN_INTERVAL_S) -> None:
    self.fetch = fetch
    self.ledger = ledger
    self.clock = clock
    self.share = float(share)
    self.interval_s = float(interval_s)
    #: Every read asked for, in order -- delivered, missing, failed or
    #: refused -- as the row the wire and the run record carry.
    self.reads: list[dict] = []
    self.refused: Counter = Counter()
    self.paid = 0
    # ⚠ None, not 0.0, for `_last_escalation`'s reason: "never read" and
    # "read at clock zero" are different facts.
    self._last_read: float | None = None
    self._delivered = 0

  # ---- the throttle ----------------------------------------------------------

  def why_not(self, decisions: int) -> str:
    """"" if a read may happen now, else the reason it may not.

    ⚠ EVERY GATE HERE IS CODE THE MODEL CANNOT REACH -- the escalation
    gate's rule. `decisions` is how many the mind has made so far; the
    share warms up from one. Where a gate bites and the wallet can pay,
    the points are spent HERE, once per read, and never banked as a
    standing exemption: a robot that wants to read out of turn twice pays
    twice.
    """
    bites = []
    if (self._last_read is not None
        and self.clock() - self._last_read < self.interval_s):
      bites.append("too-soon")
    allowed = max(1, int(self.share * int(decisions)))
    if self._delivered + 1 > allowed:
      bites.append("share")
    if not bites:
      return ""
    if self.ledger is not None and self.ledger.balance() >= LOOKUP_POINTS:
      self.ledger.spend(LOOKUP_POINTS, why="a read out of turn")
      self.paid += 1
      return ""
    return bites[0]

  # ---- the read --------------------------------------------------------------

  def read(self, query: str, decisions: int) -> dict:
    """One lookup, gated, fetched and recorded. Returns the row.

    Never raises: a transport failure is a `failed` row, because a slow
    or absent Wikipedia must cost the robot a page and nothing else --
    the decision it rode on stands whatever the library said.
    """
    query = clean(query, MAX_QUERY_CHARS)
    row = {"query": query, "outcome": "", "page": "", "revision": "",
           "url": "", "chars": 0, "why": ""}
    why = self.why_not(decisions) if query else "garbled"
    if why:
      row.update(outcome="refused" if why in REFUSALS else "failed", why=why)
      self.refused[why] += 1
    else:
      try:
        page = lookup(query, self.fetch)
      except Exception as e:                # noqa: BLE001 -- see docstring
        row.update(outcome="failed", why=failure(e))
      else:
        if page is None:
          row.update(outcome="missing")
        else:
          self._delivered += 1
          self._last_read = self.clock()
          row.update(outcome="read", page=page.title, revision=page.revision,
                     url=page.url, chars=len(page.extract), text=page.extract,
                     cut=page.cut, id=f"library:{self._delivered}")
    assert row["outcome"] in READ_OUTCOMES
    self.reads.append(row)
    return row

  def stats(self) -> dict:
    return {"reads": sum(1 for r in self.reads if r["outcome"] == "read"),
            "missing": sum(1 for r in self.reads if r["outcome"] == "missing"),
            "failed": sum(1 for r in self.reads if r["outcome"] == "failed"),
            "refused": dict(self.refused), "paid": self.paid}


def as_context(row: dict) -> dict:
  """How the model is shown a page: the visitor channel's block -- an id,
  a `from` and a `text` -- plus the page's title, its revision and what
  was asked for. Built through `VisitorMessage` so the framing IS the
  visitor channel's rather than resembling it."""
  msg = VisitorMessage(id=str(row.get("id") or ""), kind="message", who=SENDER,
                       text=str(row.get("text") or ""))
  return {**msg.as_context(), "page": row.get("page", ""),
          "revision": row.get("revision", ""), "for": row.get("query", ""),
          **({"cut": True} if row.get("cut") else {})}
