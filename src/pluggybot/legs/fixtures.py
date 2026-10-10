"""Where the commissioned fixtures put the robot (issue #476): the dock's
board and the tool rack's tags are where they were installed (#378, #405;
both commissioned, Ben 2026-09-29), so one look at either says where the
robot stands in the WORLD's frame -- what no map it laid itself can say.
A mixin: `QuadMission` is the rest of the body.

A FIX (`fixture_fix`) is a fit of a fixture's tags turned into the robot's
pose off the fixture's commissioned one, taken only where one look was
MEASURED to be good enough for it: from within `NEAR_FIX_M`, at an
approach's END -- lying on the dock (`anchor_at_dock`, the board's own
anchor) and lined up at a rack's bay (`_lined_up_routine`) -- and, from
within `FAR_FIX_M`, on any look a lost robot takes, which only has to bring
it back to an approach whose end fixes again. Never at an approach's start
nor on a walk-in's looks, which steer by what they see: from there one look
was up to 0.26 m and 7 deg off. It moves the belief only past `FIX_TOL_*`,
and the map is laid again round it (`ScanMatcher.anchored`); past `ASKEW_*`
the robot had been lost past its matcher's own search, and the map it laid
while lost is DROPPED with everything laid in it
(`QuadMission.forget_world`).

LOST (`lost_routine`): a fixture looked for from where the belief puts it
in plain sight -- the dock's board from its standoff, the rack's tags from a
bay's approach -- and not seen, with no other robot near enough it to hide
it. The map goes, and the robot looks for either fixture round where it
stands (`find_fixture_routine`): a look all round, then viewpoints outward
as a find walks them (`legs.places.next_viewpoint`), every look trying both
fixtures and one sighted too far off to fix from walked toward, until a
look is a fix.

Each fix, loss and search is a record on `belief_events`
(`Navigator.belief_event`), which the lifecycle drains into History and onto
the wire (`drift`). SimNotes, "Lost on its own map, and found by its
fixtures", has what was measured.
"""

from __future__ import annotations

import math

from pluggybot.legs import dock as dk
from pluggybot.legs import rack as rk
from pluggybot.legs.places import LOOKED_M, VIEWPOINT_PATIENCE_S, next_viewpoint
from pluggybot.legs.swap import APPROACH_STANDOFF_M
from pluggybot.mapping.scan_match import SEARCH_M, SEARCH_RAD
from pluggybot.telemetry.protocol import DRIFT_FIXTURES
from pluggybot.tick import Routine

#: The two commissioned fixtures, by the name a record carries.
FIXTURES = DRIFT_FIXTURES
#: A look is a fix only where it was MEASURED to be one (#476,
#: `scripts/drift_spike.py --looks`, the served body stood and nothing
#: stepped): within `NEAR_FIX_M` off `NEAR_FIX_TAGS` tags or more -- an
#: approach's end, lined up at a bay, where the camera reads that bay's own
#: pair -- within 0.3 cm and 0.22 deg of the truth, every look (270: every
#: bay, the line-up's whole spread); within `FAR_FIX_M` off `FAR_FIX_TAGS`
#: -- a lost robot's look -- within 15 cm and 3.6 deg off the board and 8
#: cm and 2.5 deg off the rack (392 poses round each), and one look on a
#: walk 0.26 m and 6.6 deg; three tags of the rack at 1.5-2 m, 17 cm and 5.3
#: deg. Further off, or off fewer, a look is a sighting to walk toward.
NEAR_FIX_M, NEAR_FIX_TAGS = 0.8, 2
FAR_FIX_M, FAR_FIX_TAGS = 2.5, 4
#: ...and a fix moves the belief only past this, m / rad: the matched
#: belief in the house is good to a few centimetres, and a window of scans
#: laid unmatched (`ScanMatcher.anchored`) for less is churn.
FIX_TOL_M = 0.10
FIX_TOL_RAD = math.radians(2.0)
#: ⚠ A FIX THIS FAR FROM THE BELIEF DROPS THE MAP: the robot had been lost
#: past its matcher's own search (`scan_match.SEARCH_M` / `SEARCH_RAD`), and
#: the walls it laid while lost are copies the matcher holds a corrected
#: belief to. MEASURED (#476): put back at the truth on the deployed pair's
#: own maps, a walk to the lab was 1.3 and 1.9 m out within 20 s -- each map
#: refused the truth's scans until the wide search relocated the belief onto
#: a copy (0.9 m and 3 deg; 3.1 m and 17 deg) -- and on an empty map 0.23 m
#: out at worst.
ASKEW_M = SEARCH_M
ASKEW_RAD = SEARCH_RAD
#: Another robot's reported pose this near a fixture can hide its tags, m:
#: a miss there is no evidence (`lost_routine`).
HIDDEN_BY_PEER_M = 1.5
#: How long a lost robot looks for a fixture, s.
LOST_SEARCH_S = 300.0
#: Where a fixture is looked at from, in its own frame: the dock's charge
#: standoff, facing the board; the rack's middle bay's approach start,
#: facing the rack.
LOOK_FROM = {"dock": (-dk.STANDOFF_M, 0.0, 0.0),
             "rack": (rk.WORK_X + APPROACH_STANDOFF_M, 0.0, math.pi)}
#: Each fixture's tags, by id: the drawings the fits read.
FIXTURE_TAGS = {"dock": frozenset(dk.tag_layout()),
                "rack": frozenset(i for s in rk.SPECS for i in rk.tag_layout(s))}


class Fixtures:
  """The fixtures' fix and the lost robot's search (the module docstring),
  on `QuadMission`."""

  def _init_fixtures(self) -> None:
    #: How many looks have fixed the belief: a search ends when it moves.
    self.fixes = 0
    #: A lost robot is looking for a fixture: every decode tries both
    #: (`watch_fixtures`), and one sighted too far off to fix from is walked
    #: toward, from here (x, y, heading), or None.
    self._seeking = False
    self._sighted: tuple[float, float, float] | None = None
    #: ...and how many fixes there had been when it began to look.
    self._seek_fixes = 0
    #: The last fix's record (`fixture_fix`), or None.
    self.last_fix: dict | None = None

  # ---- a fix -------------------------------------------------------------------

  def fixture_fit(self, fixture: str, seen: dict):
    """One decode's fit of a fixture, `seen` as `dock.seen_from` gives it:
    `(fit, range)` -- the range its tags' mean, m -- or None."""
    fit = dk.fit_dock(seen) if fixture == "dock" else rk.fit_rack(seen, rk.SPECS)
    if fit is None:
      return None
    near = [math.hypot(*seen[i]) for i in seen if i in FIXTURE_TAGS[fixture]]
    return fit, sum(near) / len(near)

  def fixture_pose(self, fixture: str, fit) -> tuple[float, float, float]:
    """Where a fit puts the robot in the world: the fixture's commissioned
    pose, and the robot in the fixture's frame (the fit inverted: the fit
    is the fixture seen from the robot)."""
    prior = self.dock_prior if fixture == "dock" else self.tool_rack_prior
    return dk.compose(prior, dk.relative((0.0, 0.0, 0.0), (fit.x, fit.y, fit.yaw)))

  def fixture_fix(self, fixture: str, fit, range_m: float) -> dict | None:
    """A look's fit of a fixture as a fix (the module docstring): the belief
    moved where it puts the robot past `FIX_TOL_*`, and the map laid again
    round it, or past `ASKEW_*` dropped. Its record, or None where the look
    is no fix -- too few tags or too far for one (`NEAR_FIX_*`; a lost
    robot's, `FAR_FIX_*`) -- or, unless the robot is lost, agrees with the
    belief within `FIX_TOL_*`."""
    reach, tags = (FAR_FIX_M, FAR_FIX_TAGS) if self._seeking else (NEAR_FIX_M, NEAR_FIX_TAGS)
    if range_m > reach or fit.n < (NEAR_FIX_TAGS if range_m <= NEAR_FIX_M else tags):
      return None
    before = self.pose
    wx, wy, wyaw = self.fixture_pose(fixture, fit)
    off, turned = math.hypot(wx - before[0], wy - before[1]), dk._wrap(wyaw - before[2])
    moved = off > FIX_TOL_M or abs(turned) > FIX_TOL_RAD
    if not (moved or self._seeking):
      return None
    dropped = moved and (off > ASKEW_M or abs(turned) > ASKEW_RAD)
    if moved:
      self.odo.correct(wx, wy, before[2] + turned)
      if dropped:
        self.forget_world()
      elif self.matcher is not None:
        self.matcher.anchored()
      # ...and what it believed of the fixtures was believed in the old frame
      self.dock_seen = self.tool_rack_seen = None
    self.fixes += 1
    self.last_fix = self.belief_event("fixed", before=before, fixture=fixture, tags=fit.n,
                                      atM=round(range_m, 2), moved=moved, dropped=dropped)
    return self.last_fix

  def watch_fixtures(self, dets: dict) -> None:
    """A lost robot's look (`_seeking`), every decode it takes: a fix off
    either fixture where one fits, else -- sighted too far off, or off too
    few tags -- where to look at it from instead (`LOOK_FROM`), in the map."""
    from pluggybot.legs.body import NAV_EYE
    if self.fixes > self._seek_fixes:
      return                              # ...found: the search ends at its next ask
    seen = dk.seen_from(self.model, self.data, dets, self.handle.el(NAV_EYE), self.root)
    for fixture in FIXTURES:
      got = self.fixture_fit(fixture, seen)
      if got is None:
        continue
      fit, range_m = got
      if self.fixture_fix(fixture, fit, range_m) is not None:
        self._sighted = None
        return
      there = dk.compose(self.pose, (fit.x, fit.y, fit.yaw))
      self._sighted = dk.compose(there, LOOK_FROM[fixture])

  # ---- lost --------------------------------------------------------------------

  def fixture_face(self, fixture: str) -> tuple[float, float]:
    """Where a fixture's tags are, commissioned: the board's middle, the
    rack's rail of tags' middle."""
    if fixture == "dock":
      x, y, _ = dk.compose(self.dock_prior, (dk.DEFAULT.board_x, 0.0, 0.0))
      return x, y
    ys = [y for s in rk.SPECS for y in s.tag_ys]
    x, y, _ = dk.compose(self.tool_rack_prior,
                         (rk.DEFAULT.back_x, (min(ys) + max(ys)) / 2.0, 0.0))
    return x, y

  def fixture_hidden(self, fixture: str) -> bool:
    """Another robot reported near enough the fixture to hide its tags
    (`HIDDEN_BY_PEER_M`): not seeing it there says nothing of the belief."""
    fx, fy = self.fixture_face(fixture)
    return any(math.hypot(b.x - fx, b.y - fy) < HIDDEN_BY_PEER_M for b in self._bodies())

  def lost_routine(self, fixture: str, stop=None) -> Routine:
    """`fixture` was looked for where the belief puts it in plain sight and
    is not there (the module docstring): unless another robot is near
    enough it to hide it, the robot is LOST. Its map goes, and it looks for
    either fixture round where it stands (`find_fixture_routine`); `stop`,
    asked as a walk asks it, ends that. True once a look there fixed the
    belief."""
    if self.fixture_hidden(fixture):
      return False
    before = self.pose
    self.forget_world()
    self.belief_event("lost", before=before, fixture=fixture)
    rec = yield from self.find_fixture_routine(self.pose_xy(), LOST_SEARCH_S, stop)
    return bool(rec["found"])

  def find_fixture_routine(self, near: tuple[float, float], patience: float,
                           stop=None) -> Routine:
    """Look for either fixture round `near` until a look is a fix, `patience`
    s run out or `stop` says so: a look all round where it stands, one
    sighted too far off walked toward and looked at, and viewpoints outward
    from `near` as a find walks them, a look all round at each. Returns its
    record -- `found`, `why` ("found", "not found": every viewpoint walked
    to, "out of time", "interrupted"), `seconds`, `viewpoints`, `arounds`
    -- and says it (`belief_event`, "searched", with the fix's `fixture`
    and whether it `dropped` the map, where one was found)."""
    t0 = float(self.data.time)
    until = t0 + float(patience)
    fixes0 = self.fixes
    rec = {"near": [round(near[0], 2), round(near[1], 2)], "viewpoints": 0, "arounds": 0}

    def found() -> bool:
      return self.fixes > fixes0

    def left() -> float:
      return until - float(self.data.time)

    def halt() -> bool:
      return found() or left() <= 0.0 or (stop is not None and bool(stop()))

    looked: list[tuple[float, float]] = []      # where a look all round was taken
    tried: list = []                            # ...and the viewpoints walked to
    why = ""
    self._seeking, self._sighted, self._seek_fixes = True, None, fixes0
    try:
      yield from self.stand_routine()
      self.look_for_places()                    # a decode: `watch_fixtures`
      while not halt():
        if self._sighted is not None:
          sx, sy, sh = self._sighted
          self._sighted = None
          yield from self.drive_to_routine(sx, sy, timeout=min(left(), VIEWPOINT_PATIENCE_S),
                                           stop=halt)
          if halt():
            break
          yield from self.face_routine(sh)
          self.look_for_places()
          continue
        here = self.pose_xy()
        if not any(math.hypot(here[0] - lx, here[1] - ly) < LOOKED_M for lx, ly in looked):
          rec["arounds"] += 1
          looked.append(here)
          yield from self._look_around_routine(halt)
          continue
        wall, seen_over, frontier, walk, known = self._search_map(looked, near)
        vp = next_viewpoint(near, here, looked, tried, wall, frontier, seen_over, walk, known)
        if vp is None:
          why = "not found"
          break
        tried.append(vp)
        rec["viewpoints"] += 1
        yield from self.drive_to_routine(vp[0], vp[1], timeout=min(left(), VIEWPOINT_PATIENCE_S),
                                         stop=halt)
    finally:
      self._seeking, self._sighted = False, None
    if found():
      why = "found"
    elif not why:
      why = "interrupted" if stop is not None and stop() else "out of time"
    rec.update(found=found(), why=why, seconds=round(float(self.data.time) - t0, 1))
    fix = self.last_fix if found() else None
    self.belief_event("searched", found=rec["found"], ended=why, seconds=rec["seconds"],
                      viewpoints=rec["viewpoints"], arounds=rec["arounds"],
                      **({} if fix is None else {"fixture": fix["fixture"],
                                                 "dropped": fix["dropped"]}))
    return rec
