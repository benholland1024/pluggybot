"""The places a robot has found (issue #419): a task area's tag, where the
robot saw it in its own map, which way its face points, and when. Its own
knowledge, kept with its map across a restart and forgotten only at a true
death; nothing here is handed to it.

A place is a TAG, merged by its decoded identity and never by distance
(`rack.localize.RackFinder.look`'s rule: gated by distance after drift, a
fresh sighting made a second landmark and the belief never moved), its
position blended with a floor on the newest look's weight (`RECENCY`, the
rack's), so a robot whose belief has drifted re-learns the place in its
current frame from a handful of looks. The map, the belief and the places
are laid in one frame, which is why a place found by sight carries none of
the error a coordinate handed over in the world's frame does (#414: 0.45 m
of belief drift put a foot on the shock plate).

WHICH WAY A PLACE FACES matters to the approach, never to finding it, and
is known three ways, best first (`facing`):

  the FIXTURE  the tags of one fixture -- the lab's row of plate signs --
               fitted to its drawing together: the facing off their
               baseline, the dock's rule (#88);
  the TAG      one tag's own rotation, from looks close and oblique enough
               that its pose is not a coin flip (`OBLIQUE_RAD`, `CLOSE_M`);
  the VIEW     where it was last seen from: a sign is read only from in
               front of it, so its face points somewhere this side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: A new look's weight never falls below this: `RackFinder`'s recency.
RECENCY = 0.25
#: One tag's own rotation is believed only off a look at least this far
#: off its face and no further than `CLOSE_M` away. MEASURED on the lab's
#: 120 mm signs from the nose camera: square-on the solver's pick swings
#: +-10 deg; 20 deg off it came out mirrored in 2 of 12; 40 deg and more
#: off, within 0.5 deg at every range to 5 m; and one look 27 deg off at
#: 4.1 m read 38.
OBLIQUE_RAD = math.radians(35.0)
CLOSE_M = 3.0
#: A fixture's facing needs its known tags this far apart, m.
MIN_BASELINE_M = 0.5


def _wrap(a: float) -> float:
  return math.atan2(math.sin(a), math.cos(a))


def _blend_angle(old: float, new: float, w: float) -> float:
  return _wrap(old + w * _wrap(new - old))


@dataclass
class Place:
  """One tag the robot has seen: where (its map), when, how often, and
  what its looks said of which way it faces."""

  tag: int
  x: float
  y: float
  seen_t: float
  first_t: float
  n: int = 1
  #: The direction it was last seen FROM (tag -> camera), rad.
  view: float = 0.0
  #: Its face's direction off its own trusted looks (`OBLIQUE_RAD`), and
  #: how many there were.
  tag_facing: float | None = None
  facings: int = 0

  def as_state(self) -> dict:
    return {"tag": self.tag, "x": self.x, "y": self.y, "seenT": self.seen_t,
            "firstT": self.first_t, "n": self.n, "view": self.view,
            "tagFacing": self.tag_facing, "facings": self.facings}

  @classmethod
  def from_state(cls, s: dict) -> "Place":
    return cls(tag=int(s["tag"]), x=float(s["x"]), y=float(s["y"]),
               seen_t=float(s["seenT"]), first_t=float(s["firstT"]),
               n=int(s.get("n", 1)), view=float(s.get("view", 0.0)),
               tag_facing=(None if s.get("tagFacing") is None
                           else float(s["tagFacing"])),
               facings=int(s.get("facings", 0)))


@dataclass(frozen=True)
class Fixture:
  """Tags whose layout the robot knows before it sees them, as it knows its
  dock's board: each tag's (x, y) in the fixture's frame (`layout`), and the
  direction their faces point in that frame, rad."""

  layout: dict
  facing: float


class Places:
  """Every place this robot has found (the module docstring). `ids` are
  the tags that ARE places -- a task area's; a look reports every tag it
  decoded and the rest are the dock's, the rack's, a cube's."""

  def __init__(self, ids=(), fixtures: tuple[Fixture, ...] = ()) -> None:
    self.ids = frozenset(int(t) for t in ids)
    self.fixtures = tuple(fixtures)
    self._places: dict[int, Place] = {}
    #: Bumped on every look that changed anything: a reader keeps the last
    #: it saw (the keep-out layer, `legs.places`).
    self.version = 0

  def __len__(self) -> int:
    return len(self._places)

  def __iter__(self):
    return iter(sorted(self._places.values(), key=lambda p: p.tag))

  def get(self, tag: int) -> Place | None:
    return self._places.get(int(tag))

  def see(self, tag: int, x: float, y: float, t: float, view: float,
          tag_facing: float | None = None) -> Place | None:
    """One sighting, in the robot's map frame: the tag at (x, y), seen from
    the direction `view` (tag -> camera), and its face's direction off this
    look where the look was one to trust it from (the caller's to say, on
    `OBLIQUE_RAD` and `CLOSE_M`). None for a tag that is not a place."""
    tag = int(tag)
    if tag not in self.ids:
      return None
    p = self._places.get(tag)
    if p is None:
      p = self._places[tag] = Place(tag=tag, x=float(x), y=float(y),
                                    seen_t=float(t), first_t=float(t), view=float(view))
    else:
      p.n += 1
      w = max(1.0 / p.n, RECENCY)
      p.x += w * (float(x) - p.x)
      p.y += w * (float(y) - p.y)
      p.seen_t = float(t)
      p.view = float(view)
    if tag_facing is not None:
      p.facings += 1
      p.tag_facing = (float(tag_facing) if p.tag_facing is None else
                      _blend_angle(p.tag_facing, float(tag_facing),
                                   max(1.0 / p.facings, RECENCY)))
    self.version += 1
    return p

  def facing(self, tag: int) -> tuple[float, str] | None:
    """Which way the tag's face points in the map, and how that is known:
    "fixture", "tag" or "view" (the module docstring); None if unseen."""
    p = self._places.get(int(tag))
    if p is None:
      return None
    for fx in self.fixtures:
      if p.tag in fx.layout:
        fit = self._fit(fx)
        if fit is not None:
          return _wrap(fit + fx.facing), "fixture"
    if p.tag_facing is not None:
      return p.tag_facing, "tag"
    return p.view, "view"

  def expected(self, tag: int) -> tuple[float, float, float] | None:
    """Where the tag is and which way it faces, (x, y, facing): where it was
    found, or -- not found yet -- where its fixture's drawing puts it off
    the fixture's tags that were (one, turned by its facing; two or more,
    fitted). None with nothing to go on. How a robot that has read one sign
    of the row walks to the next without searching the room for it."""
    tag = int(tag)
    p = self._places.get(tag)
    if p is not None:
      return p.x, p.y, self.facing(tag)[0]
    for fx in self.fixtures:
      if tag not in fx.layout:
        continue
      ids = [t for t in fx.layout if t in self._places]
      if not ids:
        continue
      rot = self._fit(fx)
      if rot is None:
        rot = _wrap(self.facing(ids[0])[0] - fx.facing)
      c, s = math.cos(rot), math.sin(rot)
      # the fixture's origin in the map off every known tag, averaged
      ox = oy = 0.0
      for t in ids:
        lx, ly = fx.layout[t]
        ox += self._places[t].x - (c * lx - s * ly)
        oy += self._places[t].y - (s * lx + c * ly)
      ox, oy = ox / len(ids), oy / len(ids)
      lx, ly = fx.layout[tag]
      return ox + c * lx - s * ly, oy + s * lx + c * ly, _wrap(rot + fx.facing)
    return None

  def _fit(self, fx: Fixture) -> float | None:
    """The fixture's rotation into the map off its known tags (a 2D rigid
    fit, `legs.dock.fit_dock`'s), or None without a baseline."""
    ids = [t for t in fx.layout if t in self._places]
    if len(ids) < 2:
      return None
    p = np.array([fx.layout[t] for t in ids], dtype=float)
    q = np.array([(self._places[t].x, self._places[t].y) for t in ids], dtype=float)
    if np.linalg.norm(p - p[0], axis=1).max() < MIN_BASELINE_M:
      return None
    h = (p - p.mean(axis=0)).T @ (q - q.mean(axis=0))
    return math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])

  def forget(self) -> None:
    """A true death (#419): the new robot has found nothing."""
    self._places.clear()
    self.version += 1

  def kept_state(self) -> list[dict]:
    return [p.as_state() for p in self]

  def restore_kept(self, state) -> None:
    self._places = {}
    for s in state or ():
      p = Place.from_state(s)
      if p.tag in self.ids:
        self._places[p.tag] = p
    self.version += 1
