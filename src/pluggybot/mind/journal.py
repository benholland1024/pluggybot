"""The overseer's memory: long-term goals in, notes-to-self out (issue #15).

Local files beside the sim, deliberately, and not a round trip to the website.
The design doc's reasoning is worth repeating because it is the whole choice:
the overseer runs INSIDE the sim process and is called between lifecycle
phases, so a read against the website would put an HTTP failure mode on the
path that decides what the robot does next -- and the website is the one
component whose absence the sim is otherwise completely indifferent to
(telemetry is fire-and-forget, and the publisher's whole design is "if nobody
is listening the robot keeps living"). Memory that only works when the site is
up is memory the robot loses in exactly the situation it most needs it.

Two files, and the asymmetry between them is the point:

  GOALS are the ROBOT's since issue #154: it writes them itself through
  `thoughts.intend` / `drop_goal`, and `read_goals` below still reads them.
  A plain text file on the state volume (`$PLUGGY_GOALS`,
  `/var/lib/pluggybot/goals.md` in the deploy), so they survive a restart the
  way the ledger and the boards do. What a HUMAN writes is the CONSTITUTION
  (`Main.md`) -- edit that file and the next decision reflects it, no
  redeploy, no API, no code, which is how goals themselves used to work.
  Because the robot writes this one it rides the VOLATILE half of the prompt
  rather than the cached prefix; `thoughts.py` has the argument.

  The JOURNAL is written and never edited. Append-only notes the overseer
  writes to itself, most-recent-N of which come back in the volatile half of
  the next prompt. Bounded on disk, because an unbounded file that is read into
  every prompt is a slow-motion context leak.

Neither is scoring, and neither can become scoring: nothing here has a path to
`Ledger.award`, and the journal is `narrative` tier -- the one tier in
economy/scoring.py with no evaluator and none coming. A robot writing "I did great
today" earns nothing by writing it.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pluggybot.telemetry.protocol import ROBOT_ROOT

STATE_VERSION = 1

#: Notes kept on disk. The journal is a rolling memory, not an archive -- the
#: site is where the full history is meant to live (rooftop-media-2026 #30),
#: and it gets every entry as it happens rather than by reading this file.
MAX_NOTES = 200
#: ...and how many of them the overseer is shown. Small on purpose: the last
#: few decisions are context, the last two hundred are a distraction the model
#: pays input tokens for on every call.
RECENT_NOTES = 10
#: Longest note kept, in characters. A note is a sentence to itself, and an
#: LLM handed an unbounded text field will eventually write an essay into the
#: thing that gets replayed into all of its future prompts.
MAX_NOTE_CHARS = 400
#: Longest goals file, in characters. Since issue #154 the ROBOT writes this
#: one, so the cap is a real bound on an author rather than a guard against a
#: mounted file being something nobody intended: a full file REFUSES, the way
#: `Knowledge_and_Opinions.md` does, and `drop_goal` is the remedy. Roomier
#: than the opinions file (3000) because a goal is a commitment and a robot
#: that has to abandon one to think of another is being rushed by an
#: implementation detail.
MAX_GOALS_CHARS = 8000



def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_goals(path: str | os.PathLike | None = None) -> str:
  """The robot's long-term goals, as prose. Missing file -> the defaults.

  Missing rather than empty is deliberately not an error: the very first run
  of a fresh deploy has no goals file, and a robot that refuses to start
  because nobody has written its purpose down yet is worse than one that has
  a sensible default purpose and says so.

  ⚠ Since issue #38 this file is `Goals.md`, one of the four thought files,
  and `mind/thoughts.py` owns the reading of it. This DELEGATES rather than
  keeping a second copy of the rule: two implementations of "read it, cap
  it, fall back to the defaults" is exactly the drift that ends with
  different prose in the model's prompt and on the wire. Kept under the name
  every caller before #38 used. The import is local because `thoughts`
  imports the defaults and the cap from this module.
  """
  from pluggybot.mind.thoughts import GOALS, ThoughtFiles
  return ThoughtFiles(goals_path=path).read(GOALS)


class Journal:
  """Append-only notes the overseer writes to itself.

  `on_event` receives each note as it is written, so the publisher and the
  recorder can stream it exactly as they stream a stroke or an award. That is
  the acceptance criterion "journal entries stream as events and appear on the
  site" -- the site is not expected to read this file, and cannot, since it
  runs on a different box (docs/pluggyworld.md, "Plan B").
  """

  def __init__(self, path: str | os.PathLike | None = None,
               clock: Callable[[], str] = _now) -> None:
    self.path = Path(path) if path is not None else None
    self.clock = clock
    self.on_event: list[Callable[[dict], None]] = []
    self.notes: list[dict] = []
    self.dropped = 0
    if self.path is not None and self.path.exists():
      self.load()

  def __len__(self) -> int:
    return len(self.notes)

  def note(self, text: str, t: float = 0.0, why: str = "") -> dict | None:
    """Write one note. Empty text writes nothing and returns None.

    An LLM asked for a note will sometimes return "" -- an empty entry in a
    memory that is replayed into every future prompt is pure cost, so it is
    dropped here rather than at the reader.
    """
    text = " ".join(str(text).split())[:MAX_NOTE_CHARS]
    if not text:
      return None
    # `type` is part of the entry rather than added by the publisher, so the
    # thing stored on disk and the thing on the wire are the same object
    # (protocol 0.7.0). The same shape the boards and the ledger emit.
    entry = {"type": "journal", "t": round(float(t), 3), "at": self.clock(),
             "robot": ROBOT_ROOT, "text": text}
    if why:
      entry["why"] = " ".join(str(why).split())[:MAX_NOTE_CHARS]
    self.notes.append(entry)
    if len(self.notes) > MAX_NOTES:
      self.dropped += len(self.notes) - MAX_NOTES
      del self.notes[:len(self.notes) - MAX_NOTES]
    for hook in self.on_event:
      hook(dict(entry))
    self.save()
    return entry

  def recent(self, n: int = RECENT_NOTES) -> list[dict]:
    return list(self.notes[-n:])

  # ---- persistence ----------------------------------------------------------
  # Same shape as economy/ledger.py and tools/boards.py, and for the same reason:
  # /var/lib/pluggybot outlives the image, so this must survive the restart
  # that ends every mission.

  def save(self, path: str | os.PathLike | None = None) -> Path | None:
    target = Path(path) if path is not None else self.path
    if target is None:
      return None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps({"version": STATE_VERSION, "dropped": self.dropped,
                               "notes": self.notes}, indent=1) + "\n")
    os.replace(tmp, target)          # a crash mid-write keeps the old journal
    return target

  def load(self, path: str | os.PathLike | None = None) -> "Journal":
    target = Path(path) if path is not None else self.path
    if target is None or not target.exists():
      return self
    doc = json.loads(target.read_text())
    version = int(doc.get("version", 0))
    if not 1 <= version <= STATE_VERSION:
      raise ValueError(
        f"{target}: journal state version {doc.get('version')!r}, expected "
        f"1..{STATE_VERSION} -- delete the file to start with no memory")
    self.notes = list(doc.get("notes", []))
    self.dropped = int(doc.get("dropped", 0))
    return self
