"""Finding a place (issue #419; on the humanoid, #559's stage 3): the place
memory its navigation camera feeds, and the search a job's `find` walks when
the robot has not found the place yet. A mixin: `HumanoidMission` is the
rest of the body.

LOOKING. One decode of its NAVIGATION camera, the head's Camera Module 3
pitched for the tags on the walls (`humanoid.model`'s `nav_pitch_deg`), goes
through `see_places`: each place tag in it is placed through its model of
itself as its odometry last posed it (its legs' encoders and its IMU, every
tick), and through its stance into its map. Walking, the body also looks
every `LOOK_MOVED_M` walked or `LOOK_TURNED_RAD` turned, at most every
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
seen, on or off an errand. A search whose walks to its viewpoints took it
nowhere new to look from gave up, in its last walk's words, and was never
"not found".

⚠ NO PRESS, AND NO PLATE KEPT OFF: no verb presses a plate, and its walks
keep off none -- one crossing a plate presses it (#559, stage 4: a humanoid
on a pad when it learned the sign planned no walk off it).

MEASURED (`home.storeroom`'s row from the storeroom's door, 5.5 m off, and
from in front of the counter, 0.8 m): the 120 mm tags decode across the
room and from where a robot works at the counter.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from pluggybot.activity import cage
from pluggybot.control import wrap_angle
from pluggybot.mapping.places import CLOSE_M, OBLIQUE_RAD, Places
from pluggybot.navigator import DRIVE_STOPPED
from pluggybot.perception.tags import STATION_TAG_IDS
from pluggybot.tick import Routine

#: The camera a place is looked for by: the head's, pitched for the walls.
NAV_CAMERA = "nav_eye"

#: A walking body looks for places every this far walked or this far
#: turned, and at most this often (s): 4 looks a second would be 32 ms
#: each on the served box.
LOOK_MOVED_M = 1.0
LOOK_TURNED_RAD = math.radians(45.0)
LOOK_EVERY_S = 0.5
#: The search: a look-around is this many degrees a turn, and the
#: viewpoints a lattice this far apart within `SEARCH_RADIUS_M` of the
#: address; a look-around within `LOOKED_M` of one covers it. MEASURED on
#: the lab's signs by the quadrupeds' nose camera (#419): square-on past
#: 5 m, 55 deg off to 4 m, so 2.5 m apart leaves no floor unread.
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
#: Where a place it expects is looked at from, m out of its tag's face:
#: the quadrupeds' (#419), where the neighbours' signs 1 m either side
#: were inside their camera's half-field and one look there fit the row.
STANDOFF_M = 1.8


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


def world_places(model) -> tuple[list[int], tuple]:
  """The place tags the house puts up, and their fixtures: the lab's plate
  signs, the whiteboards' tags (#406) and the task areas' (#407), where the
  world has them -- each where its body is."""
  from pluggybot.home import areas
  from pluggybot.perception.tags import BOARD_TAG_IDS
  from pluggybot.home.boards import board_fixture
  signs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lab_feed_sign") >= 0
  boards = [name for name, (tag, _) in BOARD_TAG_IDS.items()
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_tag{tag}") >= 0]
  found = {t for area, tags in areas.area_tags().items() for t, *_ in tags
           if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{area}_tag{t}") >= 0}
  ids = [*(cage.PLATE_TAGS.values() if signs else ()),
         *(t for name in boards for t in BOARD_TAG_IDS[name]), *sorted(found)]
  fixtures = (((cage.sign_row(),) if signs else ())
              + tuple(board_fixture(name) for name in boards)
              + tuple(f for f in areas.area_fixtures() if set(f.layout) <= found))
  return ids, fixtures


def place_tags(model) -> tuple[list[int], tuple]:
  """The place tags this world puts up, and their fixtures: the house's
  (`world_places`) and the storeroom's counter's row (#559)."""
  from pluggybot.home import storeroom
  ids, fixtures = world_places(model)
  if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                       f"store_tag{storeroom.TAGS[storeroom.PROPS[0]]}") >= 0:
    ids, fixtures = [*ids, *storeroom.TAGS.values()], (*fixtures, storeroom.fixture())
  return ids, fixtures


class HumanoidPlaces:
  """The places' search on `HumanoidMission` (the module docstring)."""

  def _init_places(self, model) -> None:
    ids, fixtures = place_tags(model)
    #: What it has found (`mapping.places`), of the places its world puts up.
    self.places = Places(ids=ids, fixtures=fixtures)
    #: When and from where it last looked for places (the walking look).
    self._place_look: tuple[float, tuple[float, float, float] | None] = (-math.inf, None)
    #: The last find's record.
    self.last_find: dict | None = None
    #: The newest look of its head's camera that saw its station (#572,
    #: `see_station`), or None.
    self.station_sighting: dict | None = None
    self._nav_tags = None

  def _place_detector(self):
    if self._nav_tags is None:
      from pluggybot.perception.tags import TagDetector
      self._nav_tags = TagDetector(self.model, self.handle.el(NAV_CAMERA))
    return self._nav_tags

  def close_places(self) -> None:
    if self._nav_tags is not None:
      self._nav_tags.close()
      self._nav_tags = None

  # ---- looking ------------------------------------------------------------------

  def _stance_frame(self) -> tuple[float, float, float]:
    """Its stance in the map, (x, y, heading): its pelvis where it believes
    it, less where its odometry's model of itself puts the pelvis over its
    feet (`HumanoidBody.stance`'s reading)."""
    x, y, yaw = self.pose
    sm = self.odo.sm
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, sm.d.qpos[3:7])
    r = rot.reshape(3, 3)
    syaw = yaw - math.atan2(r[1, 0], r[0, 0])
    c, s = math.cos(syaw), math.sin(syaw)
    px, py = float(sm.d.qpos[0]), float(sm.d.qpos[1])
    return x - (c * px - s * py), y - (s * px + c * py), syaw

  def _placed(self, dets: dict, ids) -> tuple[np.ndarray, dict]:
    """Each of `ids` one decode holds, in the map frame: the camera placed
    through its model of itself as its odometry last posed it (the stance
    frame), and the stance into the map. (Its eye, {id: (where the tag is,
    which way its face points out of it, rad)})."""
    sm = self.odo.sm
    mujoco.mj_camlight(sm.m, sm.d)
    cam = sm.m.camera(NAV_CAMERA).id
    origin, rot = sm.d.cam_xpos[cam].copy(), sm.d.cam_xmat[cam].reshape(3, 3).copy()
    sx, sy, syaw = self._stance_frame()
    c, s = math.cos(syaw), math.sin(syaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = {}
    for i in ids:
      d = dets[i]
      tx, ty, tz = d["t"]
      # the detector's camera is x right, y down, z ahead; MuJoCo's looks
      # along its -z with y up
      p = rz @ (origin + rot @ np.array([tx, -ty, -tz])) + [sx, sy, 0.0]
      nx, ny, nz = d["normal"]
      n = rz @ (rot @ np.array([nx, -ny, -nz]))             # into the face
      out[i] = (p, math.atan2(-n[1], -n[0]))               # ...and out of it
    return rz @ origin + [sx, sy, 0.0], out

  def see_places(self, dets: dict) -> list[int]:
    """Every place tag in one decode into the memory, in the map frame
    (`_placed`): where the tag is, where it was seen from and, off a look
    close and oblique enough (`places.OBLIQUE_RAD`), which way its face
    points."""
    ids = [i for i in dets if i in self.places.ids]
    if not ids:
      return []
    eye, placed = self._placed(dets, ids)
    t = float(self.data.time)
    for i in ids:
      p, out = placed[i]
      facing = None
      if abs(dets[i]["yaw"]) >= OBLIQUE_RAD and math.hypot(p[0] - eye[0], p[1] - eye[1]) <= CLOSE_M:
        facing = out
      self.places.see(i, float(p[0]), float(p[1]), t,
                      math.atan2(eye[1] - p[1], eye[0] - p[0]), facing)
    return ids

  def see_station(self, dets: dict) -> None:
    """Its charging station's tags in one decode of the head's camera
    (#572), in the map frame (`_placed`): `station_sighting`, the newest such
    look -- when, from where, and each tag seen, where it is and which way
    its face points by its own rotation. The station is commissioned, never
    a place: what it learns here is where its belief puts what it saw."""
    ids = [i for i in STATION_TAG_IDS if i in dets]
    if not ids:
      return
    eye, placed = self._placed(dets, ids)
    self.station_sighting = {
      "t": float(self.data.time), "eye": (float(eye[0]), float(eye[1])),
      "tags": {i: (tuple(float(v) for v in p), out) for i, (p, out) in placed.items()}}

  def look_for_places(self) -> list[int]:
    """One look now: the place tags it decoded -- and its station's, if in
    view (`see_station`)."""
    self._place_look = (float(self.data.time), self.pose)
    dets = self._place_detector().detect(self.data)
    self.see_station(dets)
    return self.see_places(dets)

  def _place_step(self) -> None:
    """The walking look (the module docstring) on its feet and level, never
    while the get-up policy has the body."""
    if not self.places.ids or self.legs == "getup" or not self.level():
      return
    t = float(self.data.time)
    last_t, last = self._place_look
    if t - last_t < LOOK_EVERY_S:
      return
    x, y, th = self.pose
    if last is not None and math.hypot(x - last[0], y - last[1]) < LOOK_MOVED_M \
        and abs(math.remainder(th - last[2], 2 * math.pi)) < LOOK_TURNED_RAD:
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

  # ---- finding --------------------------------------------------------------------

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

  def find_routine(self, tag: int, near: tuple[float, float] | None,
                   patience: float, stop=None) -> Routine:
    """Find the place `tag` marks (the module docstring). Returns its
    record: `found`, `why` ("found", "not found" -- every viewpoint looked
    from --, "gave up" -- every viewpoint walked for, and no walk to one
    took it anywhere new to look from, the last its `walk` --, "out of
    time" after `patience` s, or "interrupted" by `stop`, the robot's own
    interrupt, asked as every walk is), the `seconds` it took, whether it
    was `remembered`, how many `guesses`, `viewpoints` and look-`arounds`,
    where the place is (`at`) once found, and its `trail`, in order: each
    walk -- what to (`to`: "guess", where the place should be; "address";
    "viewpoint") and how it ended (its `last_drive`) -- and each look round
    (`lookedAt`); then where the belief stood (`err` against the truth, a
    trace's). Out of time, a walk ends where it stands and a look-around at
    its next look."""
    tag = int(tag)
    t0 = float(self.data.time)
    until = t0 + float(patience)
    rec = {"tag": tag, "remembered": self.places.get(tag) is not None,
           "near": None if near is None else [round(near[0], 2), round(near[1], 2)],
           "guesses": 0, "viewpoints": 0, "arounds": 0, "trail": []}
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
                 seconds=round(float(self.data.time) - t0, 1), err=self.truth_error(),
                 **({"at": [round(p.x, 3), round(p.y, 3)]} if p is not None and why == "found"
                    else {}))
      return rec

    def walked(to: str, x: float, y: float) -> None:
      """The walk just taken, on the trail: the body's record of it, where
      that record is this walk's, and one the find stopped saying what
      stopped it -- "found", "out of time" or "interrupted"."""
      d = self.last_drive
      mine = d is not None and math.hypot(d["goal"][0] - x, d["goal"][1] - y) < 1e-6
      entry = {"to": to, **(d if mine else {"goal": (float(x), float(y))})}
      if entry.get("why") == DRIVE_STOPPED:
        entry["why"] = ended() or DRIVE_STOPPED
      rec["trail"].append(entry)

    def looked_round(at: tuple[float, float]) -> None:
      looked.append(at)
      rec["trail"].append({"lookedAt": [round(at[0], 2), round(at[1], 2)]})

    if tag not in self.places.ids:
      return done("not a place")
    yield from self.stand_routine()
    self.look_for_places()
    if seen():
      return done("found")
    looked: list[tuple[float, float]] = []      # where a look-around was done
    guessed: list[tuple[float, float]] = []     # ...and where it was expected
    tried: list = []                            # viewpoints walked to
    began = 0                                   # ...and the looks before the first
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
        walked("guess", sx, sy)
        if ended():
          break
        yield from self.face_routine(wrap_angle(gf + math.pi))
        self.look_for_places()
        if ended():
          break
        rec["arounds"] += 1
        if (yield from self._look_around_routine(halt)):
          break
        looked_round(self.pose_xy())
        continue
      if not went_near:
        # ...else THE ADDRESS, looking as it walks
        went_near = True
        yield from self.drive_to_routine(near[0], near[1], timeout=left(), stop=halt)
        walked("address", *near)
        continue
      here = self.pose_xy()
      if not any(math.hypot(here[0] - lx, here[1] - ly) < LOOKED_M for lx, ly in looked):
        rec["arounds"] += 1
        if (yield from self._look_around_routine(halt)):
          break
        looked_round(here)
        continue
      centre = near if near is not None else here
      wall, seen_over, frontier, walk, known = self._search_map(looked, centre)
      vp = next_viewpoint(centre, here, looked, tried, wall, frontier, seen_over,
                          walk, known)
      if vp is None:
        # ⚠ NOT FOUND MEANS LOOKED (#439): the other robot resting in front
        # of the dock with nowhere to step, every walk gave up where it
        # stood, the lattice ran out in 23 s, and the find said it had
        # looked from every place near there -- Rowan took the board's tags
        # for moved. A search none of whose viewpoints' walks took it
        # anywhere new to look from gave up, in the last walk's words.
        if tried and len(looked) == began:
          rec["walk"] = next((w for w in reversed(rec["trail"]) if "to" in w), None)
          return done("gave up")
        return done("not found")
      if not tried:
        began = len(looked)
      tried.append(vp)
      rec["viewpoints"] += 1
      yield from self.drive_to_routine(vp[0], vp[1], timeout=min(left(), VIEWPOINT_PATIENCE_S),
                                       stop=halt)
      walked("viewpoint", *vp)
    return done(ended())

  # ---- what the search asks of its body ------------------------------------------

  def stand_routine(self):
    """On its feet already: it rests standing."""
    return
    yield

  def _turn_by_routine(self, degrees: float):
    """A turn where it stands by `degrees`, the look-around's step."""
    yield from self.face_routine(math.remainder(self.pose[2] + math.radians(degrees),
                                                2 * math.pi))
