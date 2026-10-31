"""A tag WATCHED while the robot moves (issue #554's stage 2; docs/Hands.md
§3): one tag's pictures read at the camera's rate, each reading the robot's
a detection's latency after its picture was taken -- no oftener and no
sooner than a head camera and the computer decoding it give one. What a
picture shows is the camera's own decode (`spot`'s, `perception.tags.TagDetector`);
where it puts the tag is the body's own model of itself (its `look`). A
picture that does not show the tag says so, and where the tag was last seen
is kept, its age growing.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Callable

#: How often a watch takes a picture, Hz: the D435 colour imager's 15 fps
#: mode at 1280 x 720 -- half the rate a Raspberry Pi 5 decodes AprilTags
#: at (PhotonVision's docs: two streams at 1280 x 800, 30 fps). MEASURED
#: (#554, the counter world): a picture costs a flight 14 ms on EGL (the
#: render 1.5, the decode 12.8) and 25 under osmesa, beside the humanoid's
#: own 0.26 s a sim-second; at 15 a second a watch near doubles a flight.
WATCH_HZ = 15.0
#: How long after its picture a reading is the robot's, s: the picture's
#: capture at that rate (1/15 s) and its decode on the Pi 5 (under 1/30 s,
#: the rate it sustains). No Pi 5 latency is published; this is the sum.
LATENCY_S = 0.1


class TagWatch:
  """One tag watched (the module docstring): `start` it, `step` it after
  every physics step, and `newest` is the newest reading the robot has --
  {tag, seen: whether its picture shows the tag, at: when that picture was
  taken (sim s), last: where the newest picture that showed it put it
  (`look`'s place and `at`) or None} -- or None before its first reading.

  `look(tag)` is the body's: one picture's decode of `tag` in the believed
  world frame -- {x, y, z, range, normal, edge} -- or None where the
  picture does not show it."""

  def __init__(self, timestep: float, look: Callable[[int], dict | None]) -> None:
    #: physics steps a picture, and from a picture to its reading -- each
    #: rounded UP: never oftener nor sooner than the part
    self.every = max(1, math.ceil(1.0 / WATCH_HZ / timestep - 1e-6))
    self.lag = max(1, math.ceil(LATENCY_S / timestep - 1e-6))
    self._look = look
    self.tag: int | None = None
    self.newest: dict | None = None
    self._last: dict | None = None
    self._due: deque = deque()
    #: ...and the camera's own clock, physics steps, running whether or not
    #: anything is watched: a watch begun takes its first picture at the
    #: camera's next frame -- restarted each `start`, a watch begun just
    #: after another took a picture 21 steps after the last (#554's review)
    self._n = 0

  def start(self, tag: int) -> None:
    """`tag` watched from the camera's next frame on, what was read of any
    tag before forgotten."""
    self.stop()
    self.tag = int(tag)

  def stop(self) -> None:
    """Nothing watched, and nothing of it kept."""
    self.tag, self.newest, self._last = None, None, None
    self._due.clear()

  def step(self, t: float, sees: bool = True) -> None:
    """One physics step on, at sim time `t`: a picture where one is due --
    of nothing where the body cannot see (`sees`: a get-up's) -- and the
    reading of each picture whose latency has passed."""
    n, self._n = self._n, self._n + 1
    if self.tag is None:
      return
    if n % self.every == 0:
      self._due.append((n + self.lag, t, self._look(self.tag) if sees else None))
    while self._due and self._due[0][0] <= n:
      _, at, got = self._due.popleft()
      if got is not None:
        self._last = {**got, "at": at}
      self.newest = {"tag": self.tag, "seen": got is not None, "at": at, "last": self._last}
