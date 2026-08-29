"""Operator mode: the switch the robot cannot reach (issue #37).

Three states, and they are a SAFETY feature at least as much as a cost one --
a kill switch for a robot behaving badly, and a way to stop spending inside a
second:

  llm       normal. The overseer decides, spending against the allowance.
  scripted  free mode. The scripted rotation decides and NO API call is made
            at all. The world keeps running and looks alive, which is the
            point: a world that goes dark to save money looks broken.
  paused    physics stops stepping and the socket STAYS OPEN, with a heartbeat
            saying `paused`, so the site shows a paused robot rather than a
            dead one.

A FILE, POLLED, AND DELIBERATELY NOT A MESSAGE. The inbound channel (#16) was
the alternative and is rejected for the reason `reset_tool` was allowed onto
it and this is not: a tool reset is recoverable and idempotent, while mode is
the control that turns the robot off. A file on the mounted volume has no
auth surface to get wrong, survives the restart that ends every mission, and
is readable by a person with `cat` when the website is the thing that is
broken. The website's admin page writes it; the sim only ever reads it.

⚠ NOTHING IN THE DECISION PATH CAN WRITE IT. There is no writer here at all
-- not a private one, not a "for tests" one -- because the file is operator
input and the overseer's action vocabulary must not gain a verb that reaches
it. `tests/test_mode.py` asserts the module exposes no way to set the mode,
which is a test that would fail the moment somebody adds a convenience.

⚠ AN UNREADABLE FILE MEANS `llm`, NOT `paused`. Failing safe here means
failing OPEN: a typo, a half-written file or a missing volume must not
silently stop a robot that nobody meant to stop, because a paused world is
indistinguishable from a broken one to everybody except the person who
paused it. A malformed value is reported and ignored.
"""

import json
import os
from pathlib import Path

from pluggybot.telemetry.protocol import MODES

#: The vocabulary lives in `telemetry.protocol` with the other two-repo
#: contracts (FACE_STATES, TASK_STATES) rather than here, because the
#: website writes these strings and the wire carries them back -- one home
#: for the words both repos have to agree on.
DEFAULT_MODE = "llm"

MODE_ENV = "PLUGGY_MODE_FILE"
DEFAULT_PATH = "/var/lib/pluggybot/mode.json"

#: Wall seconds between reads of the file. The check is a stat() at this
#: cadence rather than a read: a pause should take effect in about a second,
#: and an unchanged file should cost nothing to notice.
POLL_S = 1.0


class ModeSwitch:
  """Reads the operator's mode file. Never writes it.

  `on_change` hooks are called with (previous, current) whenever the file
  says something new -- how the lifecycle narrates a mode change into
  History and how the publisher tells the site.
  """

  def __init__(self, path: str | os.PathLike | None = None,
               clock=None) -> None:
    import time
    self.path = Path(path) if path else None
    self.clock = clock or time.monotonic
    self.mode = DEFAULT_MODE
    self.on_change: list = []
    self.errors: list[str] = []
    self._next_poll = 0.0
    self._stamp: tuple | None = None
    self.poll(force=True)

  # ---- reading -------------------------------------------------------------

  def poll(self, force: bool = False) -> str:
    """The current mode, re-read at most every `POLL_S`.

    Cheap enough for the physics seam: the fast path is one float compare,
    then a stat(), and the file is only parsed when its mtime or size moved.
    """
    if not force:
      now = self.clock()
      if now < self._next_poll:
        return self.mode
      self._next_poll = now + POLL_S
    if self.path is None:
      return self.mode
    try:
      st = self.path.stat()
      stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
      # No file is not an error: an unconfigured deployment has never had
      # one, and that is exactly the `llm` default.
      stamp = None
    if stamp == self._stamp and stamp is not None:
      return self.mode
    self._stamp = stamp
    self._apply(self._read())
    return self.mode

  def _read(self) -> str:
    if self.path is None or not self.path.exists():
      return DEFAULT_MODE
    try:
      raw = json.loads(self.path.read_text())
    except (json.JSONDecodeError, OSError) as e:
      # Half-written, hand-edited badly, or gone between the stat and the
      # read. Say so once and keep running in whatever mode was last valid.
      self._note(f"mode file unreadable ({type(e).__name__}) -- staying "
                 f"{self.mode}")
      return self.mode
    mode = str((raw or {}).get("mode", "")).strip().lower()
    if mode not in MODES:
      self._note(f"mode file says {mode!r}, which is not one of "
                 f"{', '.join(MODES)} -- staying {self.mode}")
      return self.mode
    return mode

  def _apply(self, mode: str) -> None:
    if mode == self.mode:
      return
    was, self.mode = self.mode, mode
    for hook in list(self.on_change):
      hook(was, mode)

  def _note(self, line: str) -> None:
    if line not in self.errors:
      self.errors.append(line)
    del self.errors[:-5]

  # ---- what it means -------------------------------------------------------

  @property
  def paused(self) -> bool:
    return self.mode == "paused"

  @property
  def thinking(self) -> bool:
    """May an LLM be asked? False in `scripted`, which is the whole of free
    mode: the loop still runs and still decides, from the rotation."""
    return self.mode == "llm"

  def snapshot(self) -> dict:
    return {"mode": self.mode, "notes": list(self.errors[-2:])}


def open_switch(path: str | os.PathLike | None = None) -> ModeSwitch:
  """The deployment's switch: `$PLUGGY_MODE_FILE`, then the volume's default
  location, then nothing at all (which is `llm`, forever, for a demo run)."""
  configured = path or os.environ.get(MODE_ENV) or None
  if configured is None and Path(DEFAULT_PATH).parent.is_dir():
    # Only when the state volume is actually there: a laptop running a demo
    # should not start reading /var/lib.
    configured = DEFAULT_PATH
  return ModeSwitch(configured)
