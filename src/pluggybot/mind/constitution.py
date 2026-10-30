"""The library of constitutions (issue #263): what a robot here is told it
IS, named, versioned by content, and chosen per robot.

`Main.md` used to be copied to the volume from a default in code on a
fresh root and was the human's from then on, with no write API -- the right
rule for the ROBOT, and the reason a change to the default never reached a
deployed robot (Luca's went stale after the last edit), and the reason two
robots in one world could not be given different constitutions. Two robots
on the same model in the same world with different dispositions is the
cleanest experiment this project can run: the world is the fixed
instrument, the model is held, and the only variable is what each robot is
told to care about.

So the constitution is RENDERED, not copied: the library is
`constitutions/<name>.md` beside this module (data, shipped with the
package like the reward table), each robot reads the file its environment
names on every run, and the on-volume `Main.md` is a view of it -- an
operator can read it there, and a hand edit is set aside out loud
(`ThoughtFiles`), never honoured, because the header names the constitution
in force and a file nobody can name would make that a lie.

Every file carries the same essential information -- the body, the manner,
what the person who looks after it hopes for it -- differing in EMPHASIS.
A constitution may shape what the robot VALUES; it may never hand it an
answer: no file names a threshold, a survival tactic, or a specific act
(`tests/test_constitution.py` reads every file), on the rule the event-map
and acts examples obey. And no file names the robot (issue #39): the name
is per instance, `robot_display_name`, and would freeze in a file.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

#: The library: one `.md` per constitution, the file's stem its name.
LIBRARY = Path(__file__).with_name("constitutions")
#: What a robot lives by when nothing names one: byte for byte what the
#: served quadruped reads, and the persona the fixture recording carries
#: (`tests/test_telemetry.py`). Every file opens with the robot's body, in
#: the quadruped's words since #427, which wrote it into the files rather
#: than swapping it in at load.
DEFAULT_NAME = "default"
#: ...and a humanoid's, in its own words (#559): the files' sentences about
#: the quadruped's body and world swapped for the humanoid's at load
#: (`for_body`), as the quadruped's were swapped into the rover's before
#: #427 -- so the served quadruped reads the files byte for byte, and the
#: files take the humanoid's words when the quadruped goes (#559, stage 5).
#: Asserted, each one: a file whose sentence moved fails, never quietly
#: tells a humanoid it lies on a dock.
BODY_SWAPS = {"humanoid": (
  ("You are a four-legged robot about the size of a small dog, living in a simulated "
   "house with a garden. You walk, you lie down to rest, and you charge by lying down "
   "on your dock. You have an arm on your back that folds away while you walk, and it "
   "takes the tools on the rack beside your dock, one at a time.",
   "You are a humanoid robot about as tall as a child, living in a simulated house "
   "with a garden. You walk on two legs, you rest standing still, and you charge "
   "standing at your charging station, plugging its cable into the port on your "
   "chest. You have two arms, each ending in a gripper with two jaws, and your hands "
   "are how you do your work."),
  ("A tool you fetched belongs back in its bay",
   "Leave what you worked on as you would want to find it"),
)}
#: Which constitution each robot reads, by environment, the way the display
#: name is (`PLUGGY_ROBOT_NAME` / `PLUGGY_ROBOT_NAME_2`, issue #39).
NAME_ENV = "PLUGGY_CONSTITUTION"
SECOND_NAME_ENV = "PLUGGY_CONSTITUTION_2"
#: The name a constitution handed in as TEXT reports (a test's `texts=`, a
#: demo without a library): not in the library, so its hash is the only
#: thing that identifies it.
INLINE = "inline"


class UnknownConstitution(KeyError):
  """A name the library does not hold. Refused at build, out loud, listing
  what it does hold: a misspelt `$PLUGGY_CONSTITUTION` that quietly fell
  back to the default would put the wrong name on every header of a
  period."""


@dataclass(frozen=True)
class Constitution:
  #: The library file's stem, or `INLINE`.
  name: str
  #: The text as the robot reads it: stripped, one string.
  text: str
  #: sha256 of `text` -- the VERSION. Two deployments with the same name
  #: and different hashes are two periods (docs/Observatory.md).
  sha: str

  @classmethod
  def of(cls, name: str, text: str) -> "Constitution":
    text = text.strip()
    return cls(name, text, sha_of(text))

  def as_dict(self) -> dict:
    """What the build identity and the run record carry: the name and the
    hash, never the text -- the text rides the `thought` message as
    `Main.md`, as it always did."""
    return {"name": self.name, "sha": self.sha}

  @property
  def short(self) -> str:
    return f"{self.name} ({self.sha[:8]})"


def sha_of(text: str) -> str:
  """The content hash, over the stripped text: a trailing newline on the
  volume or in a file is not a different constitution."""
  return hashlib.sha256(text.strip().encode()).hexdigest()


def names(library: Path = LIBRARY) -> list[str]:
  """Every constitution the library holds, sorted."""
  return sorted(p.stem for p in library.glob("*.md"))


def load(name: str = DEFAULT_NAME, library: Path = LIBRARY) -> Constitution:
  path = library / f"{name}.md"
  if not name or "/" in name or "\\" in name or not path.is_file():
    raise UnknownConstitution(
      f"no constitution {name!r} in the library; it holds {names(library)}")
  return Constitution.of(name, path.read_text())


def for_body(constitution: Constitution, body: str = "quadruped") -> Constitution:
  """The constitution as a robot with this body reads it: the file's own
  text, or its sentences swapped for the body's (`BODY_SWAPS`) -- the same
  name, and the hash of the text read. A text handed in (`INLINE`) is its
  writer's and is never swapped."""
  if body not in BODY_SWAPS or constitution.name == INLINE:
    return constitution
  text = constitution.text
  for old, new in BODY_SWAPS[body]:
    if old not in text:
      raise ValueError(f"constitution {constitution.name!r} no longer says {old[:50]!r}...: "
                       f"a {body} would read the quadruped's words")
    text = text.replace(old, new)
  return Constitution.of(constitution.name, text)


def resolve(name: str | None = None, env: str = NAME_ENV,
            library: Path = LIBRARY, body: str = "quadruped") -> Constitution:
  """The deploy shape: an explicit name, else the environment, else the
  default, in this `body`'s words (`for_body`). `env` is the variable this
  ROBOT reads (`NAME_ENV` for the first, `SECOND_NAME_ENV` for the second
  of a pair)."""
  name = (name or os.environ.get(env, "")).strip() or DEFAULT_NAME
  return for_body(load(name, library), body)
