"""The changelog (issue #617): what the people who run the world have written
for the robots about how their world and their bodies changed, one dated
entry at a time, newest first (docs/Overseer.md §7b).

A robot reads it through `recall` (`read` set to `changelog`, then
`changelog 2` for the page after; `ThoughtFiles.recall`), whenever it
chooses: nothing makes it. Every turn's context carries the newest entry's
date beside today's (`context`), in the per-turn state and never the cached
prefix, so a release moves no prompt sha. The site shows the same entries,
streamed once per open where a mind reads them (`message`,
`telemetry.protocol.CHANGELOG_MESSAGE`) and kept by its observatory.

⚠ IT IS A HUMAN'S TEXT, WRITTEN FOR THE ROBOTS: facts about their world and
their body, never what we measure or why, how anything is graded, a hidden
truth or what we hope they will do, and never advice. Ben approves every
entry, the release skill drafts them (`.claude/skills/pw-release`), and
`tests/test_changelog.py` reads every one for what it may not carry. The
robots have no verb that writes it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from pluggybot.telemetry.protocol import CHANGELOG_MESSAGE

#: The file, beside this module: shipped with the package, as the
#: constitutions are.
PATH = Path(__file__).with_name("changelog.md")
#: The `recall` key. A key like `chat`, ahead of the notes: a note topic
#: called `changelog` is read by `changelog/<title>`.
KEY = "changelog"
#: `changelog` is the newest page; `changelog 2` (or `changelog page 2`)
#: the one after it.
_KEY = re.compile(r"changelog(?:\s+(?:page\s+)?#?(\d+))?", re.I)
#: The longest item, in characters: a page of a recall block holds several
#: (`tests/test_changelog.py` reads every item against it).
MAX_ITEM_CHARS = 500

_TITLE = re.compile(r"# \S.*")
_HEADING = re.compile(r"## (\d{4}-\d{2}-\d{2})")
_ITEM = re.compile(r"- (\S.*)")
_MORE = re.compile(r"  +(\S.*)")


class Malformed(ValueError):
  """The file is not a changelog: refused with the line, never read past.
  A test loads the committed file, so a deployed image cannot hold one."""


@dataclass(frozen=True)
class Entry:
  #: An ISO date, UTC: the day the change reached the robots.
  date: str
  #: One fact each, in the file's order.
  items: tuple[str, ...]

  def as_dict(self) -> dict:
    return {"date": self.date, "items": list(self.items)}


@dataclass(frozen=True)
class Changelog:
  #: Newest first, as the file holds them.
  entries: tuple[Entry, ...] = ()

  @classmethod
  def parse(cls, text: str) -> "Changelog":
    """The file's one shape: a `# ` title, `<!-- -->` comments and blank
    lines, read past; a `## YYYY-MM-DD` heading per entry, newest first,
    each date once; and under it `- ` items, an item's further lines
    indented. Anything else is `Malformed`."""
    entries: list[tuple[str, list[str]]] = []
    comment = False
    for n, raw in enumerate(text.splitlines(), 1):
      line = raw.rstrip()
      if comment or line.lstrip().startswith("<!--"):
        comment = "-->" not in line
        continue
      if not line:
        continue
      if (m := _HEADING.fullmatch(line)) is not None:
        date = m.group(1)
        try:
          datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
          raise Malformed(f"line {n}: {date!r} is not a date") from None
        if entries and date >= entries[-1][0]:
          raise Malformed(f"line {n}: {date} is not older than {entries[-1][0]}"
                          " -- newest first, each date once")
        entries.append((date, []))
      elif (m := _ITEM.fullmatch(line)) is not None and entries:
        entries[-1][1].append(m.group(1))
      elif (m := _MORE.fullmatch(line)) is not None and entries and entries[-1][1]:
        entries[-1][1][-1] += " " + m.group(1)
      elif _TITLE.fullmatch(line) is not None and not entries:
        continue
      else:
        raise Malformed(f"line {n}: {line[:60]!r} is not a title, a dated "
                        "heading or an item")
    for date, items in entries:
      if not items:
        raise Malformed(f"{date}: an entry with nothing in it")
    return cls(tuple(Entry(date, tuple(items)) for date, items in entries))

  @classmethod
  def load(cls, path: Path = PATH) -> "Changelog":
    return cls.parse(path.read_text())

  @property
  def newest(self) -> str | None:
    """The newest entry's date: when the world last changed."""
    return self.entries[0].date if self.entries else None

  @property
  def sha(self) -> str:
    """sha256 of the entries as the robots read them: the version, so a
    comment or a re-wrapped line is not a new one."""
    body = json.dumps([e.as_dict() for e in self.entries], separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()

  def lines(self) -> list[str]:
    """Every item as `recall` shows it, its date in front, newest first."""
    return [f"[{e.date}] {item}" for e in self.entries for item in e.items]

  def pages(self, cap: int) -> list[list[str]]:
    """The lines in pages of at most `cap` characters, newest first; a
    page holds whole items, and one past the cap has a page of its own."""
    out: list[list[str]] = []
    size = 0
    for line in self.lines():
      if not out or size + len(line) > cap:
        out.append([])
        size = 0
      out[-1].append(line)
      size += len(line) + 1
    return out

  def page(self, n: int, cap: int) -> tuple[list[str], int]:
    """Page `n` (from 1) and how many there are: no lines past the last."""
    pages = self.pages(cap)
    return (pages[n - 1] if 1 <= n <= len(pages) else []), len(pages)

  def context(self, today: str) -> dict | None:
    """What every turn carries (Ben, 2026-11-06): the newest entry's date
    beside today's, so the date means something. None with no entries."""
    if not self.entries:
      return None
    return {"newest": self.newest, "today": today}

  def message(self, t: float) -> dict:
    """The entries on the wire, once per stream open where a mind reads
    them (`telemetry.protocol.CHANGELOG_MESSAGE`): the world's, so no
    `robot`; byte-stable for a run."""
    return {"type": CHANGELOG_MESSAGE, "t": round(float(t), 3), "sha": self.sha,
            "newest": self.newest, "entries": [e.as_dict() for e in self.entries]}


def page_of(key: str) -> int | None:
  """The page a `recall` key names, from 1, or None when it is not the
  changelog's."""
  m = _KEY.fullmatch(" ".join(str(key or "").split()))
  if m is None:
    return None
  return max(1, int(m.group(1))) if m.group(1) else 1


@lru_cache(maxsize=1)
def current() -> Changelog:
  """The changelog this image carries, read once. A test swaps it with
  `monkeypatch.setattr(changelog, "current", ...)`."""
  return Changelog.load()


def today(now: datetime | None = None) -> str:
  """Today's date, UTC, as the entries are dated: the robots' own, beside
  the newest entry's."""
  return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()
