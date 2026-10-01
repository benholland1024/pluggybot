"""Finding a place and pressing a plate, on legs (issue #419): the place
memory the nose camera feeds, the search a job's errand walks when the
robot has not found the place yet, and the approach measured off the
place's own tag. A mixin: `QuadMission` is the rest of the body.

LOOKING. Every decode of the nose camera (`QuadMission.detect_board`: the
dock's walk-in, the rack's and this module's) goes through `see_places`,
and the place tags in it go into `places` in the map frame, through the
belief, as the map is laid. Walking, the body also looks every
`LOOK_MOVED_M` walked or `LOOK_TURNED_RAD` turned, at most every
`LOOK_EVERY_S`: what it finds in its free time it remembers, and a robot
standing still spends no render.

FINDING (`find_routine`). Where it should be first: where it was found,
or where the row's drawing puts it off a sign already read
(`Places.expected`) -- facing it from its standoff, a look and a look
round. Else to `near` -- a job's address, in the map -- over #399's
planner, and a search round it: a look-around (`LOOK_AROUND_DEG` turns, a
look after each) from each viewpoint `next_viewpoint` picks, the frontiers
of its own map nearest `near` and then a lattice round it, until the tag is
in view or its patience runs out. Every walk stops the moment the tag is
seen, on or off an errand.

PRESSING (`press_routine`). From the plate's standoff, facing its sign, a
look; then the walk in, steered by the sign as the rack's bays are
(`rack.walk_in_twist`), re-reading it every `dock.LOOK_EVERY_S`, to where
the front feet stand inside the pad (`cage.PRESS_BACK_M`); a hold, and
straight back out -- three stretches that run to their end, so a press
starts them only with `FINAL_S` of its patience left. The map's drift does
not matter: the last step is measured off the tag.

KEEPING OFF (`keep_out`). Every pad the robot knows is a wall to the
planner, so no walk crosses a plate it has seen, and a press is the only
way onto one.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from pluggybot.activity import cage
from pluggybot.activity.plate import PLATE_HALF
from pluggybot.legs import dock as dk
from pluggybot.legs import rack as rk
from pluggybot.mapping.places import CLOSE_M, OBLIQUE_RAD, Places
from pluggybot.navigator import DRIVE_STOPPED
from pluggybot.rack.coupling import touching
from pluggybot.tick import Routine

#: A walking body looks for places every this far walked or this far
#: turned, and at most this often (s): the dock's cadence would be 4 looks
#: a second, 32 ms each on the served box.
LOOK_MOVED_M = 1.0
LOOK_TURNED_RAD = math.radians(45.0)
LOOK_EVERY_S = 0.5
#: The search: a look-around is this many degrees a turn (the nose
#: camera's field is 67 deg across), and the viewpoints a lattice this far
#: apart within `SEARCH_RADIUS_M` of the address; a look-around within
#: `LOOKED_M` of one covers it. MEASURED on the lab's signs: square-on
#: past 5 m, 55 deg off to 4 m, so 2.5 m apart leaves no floor unread.
LOOK_AROUND_DEG = 60.0
SEARCH_STEP_M = 2.5
SEARCH_RADIUS_M = 7.5
LOOKED_M = 1.25
#: ...and a lattice point on floor the map knows, this near a look-around,
#: has been looked over, m: the signs read 55 deg off their face to 4 m.
LOOKED_OVER_M = 3.5
#: Where no walk from the address reaches a viewpoint, it ranks this far
#: behind every one a walk does, m: after them all, never out of the search.
UNWALKED_M = 1000.0
#: A walk to one viewpoint gives up after this long, s.
VIEWPOINT_PATIENCE_S = 90.0
#: The press: its standoff from the sign, m -- the neighbours' signs 1 m
#: either side are then inside the camera's 33.5 deg half-field, and one
#: look there fits the row -- its tries, the walk in's budget, the hold on
#: the pad and the walk back out (s, m).
STANDOFF_M = 1.8
PRESS_TRIES = 2
WALK_IN_S = 25.0
PRESS_HOLD_S = 1.0
BACK_OUT_M, BACK_OUT_S = 1.0, 8.0
#: A look at the standoff that moves it this far sends the press there
#: before its walk in (m): a facing 10 deg out puts it 0.31 m off the axis.
STANDOFF_MOVED_M = 0.25
#: The walk in, the hold and the walk back out run to their end, so a press
#: starts them only with this long left of its patience, s: their budgets.
FINAL_S = WALK_IN_S + PRESS_HOLD_S + BACK_OUT_S + dk.BACK_OUT_SETTLE_S
#: A pad it knows is kept out to its corners' circle: the planner's
#: inflation (0.35 m) holds the torso that far off it, and the feet, 0.26 m
#: from the torso, 0.09 m off.
PAD_KEEP_OUT_M = PLATE_HALF * math.sqrt(2.0)


def next_viewpoint(near: tuple[float, float], here: tuple[float, float],
                   looked, tried, wall, frontier=(), seen=None, walk=None,
                   known=None) -> tuple[float, float] | None:
  """The next viewpoint of a search round `near`, within `SEARCH_RADIUS_M`
  of it. First the robot's OWN FLOOR: a FRONTIER of its map -- known floor
  meeting the unknown, the way into a room it has not seen -- or a point of
  a lattice `SEARCH_STEP_M` apart round `near` on floor the map knows
  (`known(x, y)`) that no look-around has looked over (`seen(x, y)`),
  whichever is the shorter WALK from `near` (`walk(x, y)`, over the map;
  the straight line where none is given). With none left, a lattice point
  in space it has not mapped, by the same walk. Never one within `LOOKED_M`
  of a look-around (`looked`) or of a viewpoint `tried`, nor on a
  `wall(x, y)` it knows; one no walk reaches comes after every one a walk
  does, by the straight line (a map drifted far enough can cut the walk's
  field short, and the search must not end on it); of those as near
  `near`, the nearest `here`.

  ⚠ ROUND THE ADDRESS, NEVER ROUND THE ROBOT: nearest the robot first
  walked #419's first flight out of the house, round the building and into
  the street (600 s, not found). ⚠ AND THE MAP BEFORE THE LATTICE IN THE
  UNKNOWN: from the facility's address, which is in its storeroom, the
  lattice's first nine viewpoints were the storeroom's and its walls' (600
  s, not found, twice); the storeroom's frontier is its door, and the
  lobby's the lab's door. ⚠ AND BY THE WALK, WITH THE FLOOR IT HAS NOT
  LOOKED AT BESIDE ITS FRONTIERS (#403): after an explore the only frontiers
  near the address were outside the building -- metres off in a straight
  line, a long walk by the way in -- and the lab's floor, mapped through its
  door and never looked at, came after all of them: three finds of three
  failed."""
  def dist(x: float, y: float) -> float:
    straight = math.hypot(x - near[0], y - near[1])
    if walk is None:
      return straight
    w = walk(x, y)
    return w if math.isfinite(w) else UNWALKED_M + straight

  def ok(x: float, y: float, lattice: bool) -> bool:
    if math.hypot(x - near[0], y - near[1]) > SEARCH_RADIUS_M + 1e-9:
      return False
    if any(math.hypot(x - lx, y - ly) < LOOKED_M for lx, ly in (*looked, *tried)):
      return False
    return not wall(x, y) and not (lattice and seen is not None and seen(x, y))

  def pick(points) -> tuple[float, float] | None:
    best, best_d = None, (math.inf, math.inf)
    for x, y, lattice in points:
      d = (round(dist(x, y), 3), math.hypot(x - here[0], y - here[1]))
      if d < best_d and ok(x, y, lattice):
        best, best_d = (round(x, 3), round(y, 3)), d
    return best

  n = int(SEARCH_RADIUS_M // SEARCH_STEP_M)
  lattice = [(near[0] + i * SEARCH_STEP_M, near[1] + j * SEARCH_STEP_M)
             for i in range(-n, n + 1) for j in range(-n, n + 1)]
  mapped = [(x, y) for x, y in lattice if known is not None and known(x, y)]
  found = pick([(float(x), float(y), False) for x, y in frontier]
               + [(x, y, True) for x, y in mapped])
  if found is not None:
    return found
  return pick((x, y, True) for x, y in lattice if (x, y) not in mapped)


class PlaceWalk:
  """The places' routines (the module docstring), on `QuadMission`."""

  def _init_places(self, model) -> None:
    signs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lab_feed_sign") >= 0
    #: What it has found (`mapping.places`): the lab's plate signs, where
    #: the world has them.
    self.places = Places(ids=cage.PLATE_TAGS.values() if signs else (),
                         fixtures=(cage.sign_row(),) if signs else ())
    #: Each plate's pad, by its sign's tag: what a press is felt on.
    self._pads = ({cage.PLATE_TAGS[n]: model.geom(f"lab_{n}_plate_pad").id
                   for n in cage.PLATE_NAMES} if signs else {})
    from pluggybot.legs.model import LEGS
    self._foot_gids = [model.geom(self.handle.el(f"{leg}_foot")).id for leg in LEGS]
    #: When and from where it last looked for places (the walking look).
    self._place_look: tuple[float, tuple[float, float, float] | None] = (-math.inf, None)
    self._keep_out: tuple[int, np.ndarray | None] | None = None
    #: The last find's and the last press's records.
    self.last_find: dict | None = None
    self.last_press: dict | None = None

  # ---- looking ------------------------------------------------------------------

  def see_places(self, dets: dict) -> list[int]:
    """Every place tag in one decode into the memory, in the map frame:
    where it is, where it was seen from, and -- off a look close and
    oblique enough (`places.OBLIQUE_RAD`) -- which way its face points."""
    ids = [i for i in dets if i in self.places.ids]
    if not ids:
      return []
    from pluggybot.legs.body import NAV_EYE
    level, r_mount, p_mount = dk.camera_mount(self.model, self.data,
                                              self.handle.el(NAV_EYE), self.root)
    p = level @ p_mount                          # the camera, heading frame
    x, y, th = self.pose
    c, s = math.cos(th), math.sin(th)
    cam = (x + c * p[0] - s * p[1], y + s * p[0] + c * p[1])
    t = float(self.data.time)
    for i in ids:
      d = dets[i]
      tx, ty, tz = d["t"]
      h = level @ (r_mount @ np.array([tx, -ty, -tz]) + p_mount)   # `dock.seen_from`
      wx, wy = x + c * h[0] - s * h[1], y + s * h[0] + c * h[1]
      facing = None
      if abs(d["yaw"]) >= OBLIQUE_RAD and math.hypot(h[0] - p[0], h[1] - p[1]) <= CLOSE_M:
        nx, ny, nz = d["normal"]
        n = level @ (r_mount @ np.array([nx, -ny, -nz]))   # into the face, heading frame
        facing = th + math.atan2(-n[1], -n[0])             # ...and out of it, in the map
      self.places.see(i, wx, wy, t, math.atan2(cam[1] - wy, cam[0] - wx), facing)
    return ids

  def look_for_places(self) -> list[int]:
    """One look now: the place tags it decoded."""
    self._place_look = (float(self.data.time), self.pose)
    dets = self.detect_board()
    return [i for i in dets if i in self.places.ids]

  def _place_step(self) -> None:
    """The walking look (the module docstring), on the physics seam."""
    if not self.places.ids:
      return
    from pluggybot.legs.body import STANDING
    if self.posture != STANDING:
      return
    t = float(self.data.time)
    last_t, last = self._place_look
    if t - last_t < LOOK_EVERY_S:
      return
    x, y, th = self.pose
    if last is not None and math.hypot(x - last[0], y - last[1]) < LOOK_MOVED_M \
        and abs(dk._wrap(th - last[2])) < LOOK_TURNED_RAD:
      return
    self.look_for_places()

  def _look_around_routine(self, stop) -> Routine:
    """Turn a full circle in `LOOK_AROUND_DEG` steps, a look after each;
    True as soon as `stop()` is."""
    for _ in range(int(round(360.0 / LOOK_AROUND_DEG))):
      self.look_for_places()
      if stop():
        return True
      yield from self._turn_by_routine(LOOK_AROUND_DEG)
    self.look_for_places()
    return bool(stop())

  # ---- where a place is ---------------------------------------------------------

  def place_standoff(self, tag: int, dist: float = STANDOFF_M) -> tuple[float, float, float] | None:
    """`dist` out of the tag's face, facing it: (x, y, heading), or None
    for a tag it has not found."""
    p = self.places.get(tag)
    if p is None:
      return None
    f, _ = self.places.facing(tag)
    return (p.x + dist * math.cos(f), p.y + dist * math.sin(f), dk._wrap(f + math.pi))

  def press_pose(self, tag: int) -> tuple[float, float, float] | None:
    """Where the torso stands to press the plate `tag` marks, facing its
    sign (`cage.PRESS_BACK_M` short of the pad's centre), or None."""
    p = self.places.get(tag)
    if p is None:
      return None
    f, _ = self.places.facing(tag)
    px, py = cage.pad_from_sign(p.x, p.y, f)
    return (px + cage.PRESS_BACK_M * math.cos(f), py + cage.PRESS_BACK_M * math.sin(f),
            dk._wrap(f + math.pi))

  def keep_out(self) -> np.ndarray | None:
    """The cells of every pad it knows (`PAD_KEEP_OUT_M` round each), or
    None: what the planner treats as wall (`QuadMission._planning_grid`).
    ⚠ A sign whose face is known only by where it was seen from ("view",
    up to 70 deg off) has its pad anywhere `cage.SIGN_BEHIND_M` round it,
    so all of that is kept out. Built again only when a look changed what
    it knows."""
    v = self.places.version
    if self._keep_out is not None and self._keep_out[0] == v:
      return self._keep_out[1]
    g = self.grid
    mask = None
    for p in self.places:
      if p.tag not in self._pads:
        continue
      f, how = self.places.facing(p.tag)
      if how == "view":
        (px, py), radius = (p.x, p.y), cage.SIGN_BEHIND_M + PAD_KEEP_OUT_M
      else:
        (px, py), radius = cage.pad_from_sign(p.x, p.y, f), PAD_KEEP_OUT_M
      if mask is None:
        mask = np.zeros_like(g.grid, dtype=bool)
      r = int(math.ceil(radius / g.resolution))
      cx, cy = g.world_to_cell(px, py)
      rows, cols = mask.shape
      y0, y1 = max(cy - r, 0), min(cy + r + 1, rows)
      x0, x1 = max(cx - r, 0), min(cx + r + 1, cols)
      if y0 >= y1 or x0 >= x1:
        continue
      yy, xx = np.mgrid[y0:y1, x0:x1]
      wx = g.x_min + (xx + 0.5) * g.resolution
      wy = g.y_min + (yy + 0.5) * g.resolution
      mask[y0:y1, x0:x1] |= (wx - px) ** 2 + (wy - py) ** 2 <= radius ** 2
    self._keep_out = (v, mask)
    return mask

  def _search_map(self, looked, near: tuple[float, float]):
    """What a search's next viewpoint is picked over, off the planning map:
    `wall(x, y)` -- a known wall or its inflation --, `seen(x, y)` -- known
    floor within `LOOKED_OVER_M` of a look-around WITH NO WALL BETWEEN --,
    `known(x, y)` -- floor the map knows --, `walk(x, y)` -- how far it is
    to walk from `near` over the map, the planner's costs (`optimistic`),
    inf where no walk gets there -- and the frontier cells a body could
    stand on, as world points.

    ⚠ IN SIGHT, NOT IN REACH (#403): a look-around in the lobby had marked
    the lab's floor, 3 m off through a wall, looked over."""
    from scipy.ndimage import distance_transform_edt

    from pluggybot.mapping import optimistic
    from pluggybot.mapping.astar import nearest_traversable
    from pluggybot.mapping.frontier import FREE_THRESH, OCC_THRESH, find_frontiers, inflated
    g = self.grid
    grid = self._planning_grid()
    rows, cols = grid.shape
    walls = inflated(grid, self.INFLATION_CELLS)
    cells = find_frontiers(grid, traversable=(grid < FREE_THRESH) & ~walls)
    frontier = np.column_stack([g.x_min + (cells[:, 0] + 0.5) * g.resolution,
                                g.y_min + (cells[:, 1] + 0.5) * g.resolution])
    # the LIDAR's walls are what the nose camera cannot see through; the
    # layer under its plane (the cage, the signs) is below the camera's eye
    blocks = g.grid > OCC_THRESH

    def cell(x: float, y: float) -> tuple[int, int] | None:
      cx, cy = g.world_to_cell(x, y)
      return (cx, cy) if 0 <= cx < cols and 0 <= cy < rows else None

    def wall(x: float, y: float) -> bool:
      c = cell(x, y)
      return c is None or bool(walls[c[1], c[0]])

    def known(x: float, y: float) -> bool:
      c = cell(x, y)
      return c is not None and bool(grid[c[1], c[0]] < FREE_THRESH)

    def in_sight(ax: float, ay: float, bx: float, by: float) -> bool:
      n = int(math.hypot(bx - ax, by - ay) / (0.5 * g.resolution)) + 1
      t = np.linspace(0.0, 1.0, n + 1)
      ix = np.floor((ax + (bx - ax) * t - g.x_min) / g.resolution).astype(np.int64)
      iy = np.floor((ay + (by - ay) * t - g.y_min) / g.resolution).astype(np.int64)
      ok = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
      return not blocks[iy[ok], ix[ok]].any()

    def seen(x: float, y: float) -> bool:
      return known(x, y) and any(
        math.hypot(x - lx, y - ly) <= LOOKED_OVER_M and in_sight(lx, ly, x, y)
        for lx, ly in looked)

    # THE WALK FROM THE ADDRESS: one Dijkstra over the lattice the walks plan
    # on, from the lattice cell nearest the address a body could stand on
    b = optimistic.BLOCK
    cost = optimistic.coarsen(optimistic.map_costs(grid, self.INFLATION_CELLS,
                                                   self.UNKNOWN_COST), b)
    lat = optimistic.lattice_for(cost.shape, g.resolution * b)
    nx, ny = g.world_to_cell(near[0], near[1])
    src = nearest_traversable(np.isfinite(cost), (nx // b, ny // b),
                              radius=optimistic.ESCAPE_CELLS)
    if src is None:
      return wall, seen, frontier, None, known
    lat.weigh(cost)
    dist, _ = lat.search(src[1] * cost.shape[1] + src[0])
    field = dist.reshape(cost.shape)
    # a point whose own lattice cell does not pass (a frontier at the edge
    # of the floor) is read off the nearest one that does
    finite = np.isfinite(field)
    _, (ny_, nx_) = distance_transform_edt(~finite, return_indices=True)

    def walk(x: float, y: float) -> float:
      c = cell(x, y)
      if c is None:
        return math.inf
      ly, lx = c[1] // b, c[0] // b
      jy, jx = int(ny_[ly, lx]), int(nx_[ly, lx])
      if math.hypot(jy - ly, jx - lx) > 2:
        return math.inf
      return float(field[jy, jx])

    return wall, seen, frontier, walk, known

  # ---- finding --------------------------------------------------------------------

  def find_routine(self, tag: int, near: tuple[float, float] | None,
                   patience: float, stop=None) -> Routine:
    """Find the place `tag` marks (the module docstring). Returns its
    record: `found`, `why` ("found", "not found" -- every viewpoint looked
    from --, "out of time" after `patience` s, or "interrupted" by `stop`,
    the robot's own interrupt, asked as every walk is), the `seconds` it
    took, whether it was `remembered`, how many `guesses`, `viewpoints` and
    look-`arounds`, and where the place is (`at`) once found. Out of time,
    a walk ends where it stands and a look-around at its next look."""
    tag = int(tag)
    t0 = float(self.data.time)
    until = t0 + float(patience)
    rec = {"tag": tag, "remembered": self.places.get(tag) is not None,
           "near": None if near is None else [round(near[0], 2), round(near[1], 2)],
           "guesses": 0, "viewpoints": 0, "arounds": 0}
    self.last_find = rec

    def seen() -> bool:
      p = self.places.get(tag)
      return p is not None and p.seen_t >= t0

    def left() -> float:
      return until - float(self.data.time)

    def halt() -> bool:
      return seen() or left() <= 0.0 or (stop is not None and bool(stop()))

    def ended() -> str:
      """Why a walk or a look-around ended the find: "" if it goes on."""
      if seen():
        return "found"
      if stop is not None and stop():
        return "interrupted"
      return "out of time" if left() <= 0.0 else ""

    def done(why: str) -> dict:
      p = self.places.get(tag)
      rec.update(found=why == "found", why=why,
                 seconds=round(float(self.data.time) - t0, 1),
                 **({"at": [round(p.x, 3), round(p.y, 3)]} if p is not None and why == "found"
                    else {}))
      return rec

    if tag not in self.places.ids:
      return done("not a place")
    yield from self.stand_routine()
    self.look_for_places()
    if seen():
      return done("found")
    looked: list[tuple[float, float]] = []      # where a look-around was done
    guessed: list[tuple[float, float]] = []     # ...and where it was expected
    tried: list = []                            # viewpoints walked to
    went_near = near is None
    while not ended():
      guess = self.places.expected(tag)
      if guess is not None and not any(math.hypot(guess[0] - gx, guess[1] - gy) < LOOKED_M
                                       for gx, gy in guessed):
        # WHERE IT SHOULD BE: where it was found, or where the fixture's
        # drawing puts it off a sign already read -- facing it from its
        # standoff, a look, and a look round
        guessed.append(guess[:2])
        rec["guesses"] += 1
        gx, gy, gf = guess
        sx, sy = gx + STANDOFF_M * math.cos(gf), gy + STANDOFF_M * math.sin(gf)
        yield from self.drive_to_routine(sx, sy, timeout=min(left(), VIEWPOINT_PATIENCE_S),
                                         stop=halt)
        if ended():
          break
        yield from self.face_routine(dk._wrap(gf + math.pi))
        self.look_for_places()
        if ended():
          break
        rec["arounds"] += 1
        if (yield from self._look_around_routine(halt)):
          break
        looked.append(self.pose_xy())
        continue
      if not went_near:
        # ...else THE ADDRESS, looking as it walks
        went_near = True
        yield from self.drive_to_routine(near[0], near[1], timeout=left(), stop=halt)
        continue
      here = self.pose_xy()
      if not any(math.hypot(here[0] - lx, here[1] - ly) < LOOKED_M for lx, ly in looked):
        rec["arounds"] += 1
        if (yield from self._look_around_routine(halt)):
          break
        looked.append(here)
        continue
      centre = near if near is not None else here
      wall, seen_over, frontier, walk, known = self._search_map(looked, centre)
      vp = next_viewpoint(centre, here, looked, tried, wall, frontier, seen_over,
                          walk, known)
      if vp is None:
        return done("not found")
      tried.append(vp)
      rec["viewpoints"] += 1
      yield from self.drive_to_routine(vp[0], vp[1], timeout=min(left(), VIEWPOINT_PATIENCE_S),
                                       stop=halt)
    return done(ended())

  # ---- pressing ---------------------------------------------------------------------

  def _feet_on(self, pad: int) -> bool:
    return any(touching(self.data, f, (pad,)) for f in self._foot_gids)

  def press_routine(self, tag: int, patience: float, stop=None) -> Routine:
    """Walk onto the plate `tag` marks and back off it (the module
    docstring). Returns its record: `pressed` (a foot of this robot on the
    pad during the hold), `why` ("pressed", "not a plate", "not found" --
    it has not found the plate, or forgot it on the way, a true death --,
    "gave up" on the walk to its standoff, "lost" -- its sign not in view
    there --, "not pressed", "out of time", or "interrupted" by `stop` on
    the walk to the standoff), the `seconds` it took, and its `attempts`:
    each one's standoff, a walk that did not arrive (`walk`, its
    `last_drive`), its look round, where it stopped against the press pose,
    and how a failed one ended -- `why`, where the belief stood (`at`) and
    its error against the truth (`err`, a trace's), `s` into the press --
    and `leftS` where too little was left for the next. The walk in, the
    hold and the walk back out run to their end, so they start only with
    `FINAL_S` of the `patience` left."""
    tag = int(tag)
    t0 = float(self.data.time)
    until = t0 + float(patience)
    rec = {"tag": tag, "attempts": []}
    self.last_press = rec
    pad = self._pads.get(tag)

    def left() -> float:
      return until - float(self.data.time)

    def done(why: str) -> dict:
      rec.update(pressed=why == "pressed", why=why,
                 seconds=round(float(self.data.time) - t0, 1))
      return rec

    def failed(att: dict, why: str) -> str:
      att.update(why=why, at=[round(self.pose[0], 3), round(self.pose[1], 3)],
                 err=self.truth_error(), s=round(float(self.data.time) - t0, 1))
      return why

    if pad is None:
      return done("not a plate")
    if self.places.get(tag) is None:
      return done("not found")
    yield from self.stand_routine()
    # ⚠ A TRY THAT NEVER BEGAN IS NOT WHAT FAILED (#439): the first walk is
    # handed all but `FINAL_S` of the patience, so a walk that used it left
    # the second try nothing, and that try's "out of time" overwrote the
    # walk's own cause -- 24 of 24 failed live presses, each after 85 s
    why = "out of time"
    for _ in range(PRESS_TRIES):
      standoff = self.place_standoff(tag)
      if standoff is None:
        return done("not found")
      if left() < FINAL_S:
        if rec["attempts"]:
          rec["leftS"] = round(left(), 1)
        break
      sx, sy, _ = standoff
      att: dict = {"standoff": [round(sx, 3), round(sy, 3)]}
      rec["attempts"].append(att)
      arrived = yield from self.drive_to_routine(
        sx, sy, timeout=min(VIEWPOINT_PATIENCE_S, left() - FINAL_S),
        **({"stop": stop} if stop is not None else {}))
      if stop is not None and (self.last_drive or {}).get("why") == DRIVE_STOPPED:
        return done("interrupted")
      if not arrived:
        att["walk"] = self.last_drive
        if math.hypot(sx - self.pose[0], sy - self.pose[1]) > 0.5:
          why = failed(att, "gave up")
          continue
      standoff = self.place_standoff(tag)      # as the looks on the way left it
      if standoff is None:
        return done("not found")
      yield from self.face_routine(standoff[2])
      if tag not in self.look_for_places():
        cut = yield from self._look_around_routine(
          lambda: tag in self._last_seen() or left() < FINAL_S)
        if tag not in self._last_seen():
          att["looked"] = "cut short" if cut else "all round"
          why = failed(att, "lost")
          continue
        standoff = self.place_standoff(tag)
        if standoff is None:
          return done("not found")
        yield from self.face_routine(standoff[2])
      now = self.place_standoff(tag)           # as the look here left it
      if (now is not None and math.hypot(now[0] - sx, now[1] - sy) > STANDOFF_MOVED_M
          and left() >= FINAL_S):
        standoff = now
        # ...ON ITS AXIS BEFORE THE WALK IN (#403): the look here fitted a
        # facing the walk there did not have -- one read off where the sign
        # was first seen, up to 70 deg out -- and the walk in, steered
        # straight at the press pose with no planner under it, crossed the
        # next plate from the old standoff: 4 shock presses in one flight.
        # So to the new one, over the planner and its kept-out pads, first.
        att["reaim"] = [round(standoff[0] - sx, 3), round(standoff[1] - sy, 3)]
        if not (yield from self.drive_to_routine(
            standoff[0], standoff[1], timeout=min(VIEWPOINT_PATIENCE_S, left() - FINAL_S),
            **({"stop": stop} if stop is not None else {}))):
          if stop is not None and (self.last_drive or {}).get("why") == DRIVE_STOPPED:
            return done("interrupted")
          att["walk"] = self.last_drive
        yield from self.face_routine(standoff[2])
        self.look_for_places()
      if left() < FINAL_S:
        why = failed(att, "out of time")
        break
      att["walkIn"] = yield from self._press_walk_in_routine(tag)
      if att["walkIn"] == "not found":
        yield from self._press_back_out_routine()
        return done("not found")
      ex, ey, eth = dk.relative(self.pose, self.press_pose(tag))
      att["stop"] = [round(ex, 3), round(ey, 3), round(math.degrees(eth), 1)]
      pressed = yield from self._hold_on_routine(pad)
      yield from self._press_back_out_routine()
      if pressed:
        return done("pressed")
      why = failed(att, "not pressed")
    return done(why)

  def _last_seen(self) -> list[int]:
    """The place tags the last look decoded, off its time."""
    t = self._place_look[0]
    return [p.tag for p in self.places if p.seen_t >= t]

  def _press_walk_in_routine(self, tag: int) -> Routine:
    """The walk from the standoff to the press pose, steered by the sign
    (`rack.walk_in_twist`), re-read every `dock.LOOK_EVERY_S`: "stopped",
    "budget", or "not found" -- the place forgotten under it."""
    from pluggybot.legs.policy import Twist
    t0 = last = float(self.data.time)
    while self.data.time - t0 < WALK_IN_S:
      pose = self.press_pose(tag)
      if pose is None:
        return "not found"
      tw = rk.walk_in_twist(*dk.relative(self.pose, pose))
      if tw == Twist():
        return "stopped"
      yield from self._twist_routine(tw.vx, tw.vy, tw.yaw_rate)
      if self.data.time - last >= dk.LOOK_EVERY_S:
        self.look_for_places()
        last = float(self.data.time)
    return "budget"

  def _hold_on_routine(self, pad: int) -> Routine:
    """Stand `PRESS_HOLD_S` on the plate: True if a foot was on its pad."""
    felt = False
    for _ in range(round(PRESS_HOLD_S / self.model.opt.timestep)):
      yield from self._twist_routine(0.0, 0.0, 0.0)
      felt = felt or self._feet_on(pad)
    return felt

  def _press_back_out_routine(self) -> Routine:
    return (yield from self._back_out_by_routine(BACK_OUT_M, BACK_OUT_S, rk.APPROACH_V,
                                                 dk.BACK_OUT_SETTLE_S))
