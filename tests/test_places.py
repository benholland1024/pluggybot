"""Places, not coordinates (issue #419): a job says which house a task area
is in and how to find it, and the robot finds the area's tag, remembers it,
and approaches it by sight. Each rule pinned as cheaply as it fails:

  1. THE MEMORY (`mapping/places.py`): merged by tag, following new looks;
     its facing off the row, else the tag, else the view; the row's drawing
     puts a sign not yet seen; forgotten at a true death, kept by a restart.
  2. THE SIGNS: in the quadruped's house where the drawing says, and not in
     the rover's (its pass through a pad runs where a sign stands).
  3. THE BODY: a quadruped standing in the lab and looking once remembers
     the three signs where they are; the planner keeps off every pad it
     knows; the walking look is paid for by walking; the press stands the
     front feet on the pad and the fork off the sign; the places ride a
     restart with the map and go with it at a true death.
  4. THE SEARCH: where the row says first, then the address, then outward
     from the address -- never outward from the robot.
  5. THE VERBS, THE OFFER, THE MIND'S VIEW, THE TRUE DEATH.

The walk that finds a plate and presses it is flown by
`scripts/places_spike.py` (SimNotes, "Places, not coordinates"); no flight
of it is in the suite -- #403's paid feed on legs is the one that will fly
`find` and `press` end to end.
"""

import json
import math
import re

import mujoco
import pytest

from pluggybot import tick
from pluggybot.activity import cage
from pluggybot.activity.plate import PLATE_HALF
from pluggybot.economy.ledger import Ledger
from pluggybot.home import places as addresses
from pluggybot.home import world as home
from pluggybot.legs import body as qb
from pluggybot.legs import places as lp
from pluggybot.legs import world as lw
from pluggybot.legs.model import CHOSEN
from pluggybot.lifecycle import (QUAD_HOME, overseer_context, places_context,
                                 world_config, world_facts)
from pluggybot.mapping.frontier import OCC_THRESH
from pluggybot.mapping.places import Fixture, Places
from pluggybot.mind import overseer as ov
from pluggybot.procedure import steps as st
from pluggybot.rack.tags import PLATE_TAG_IDS
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

SHOCK, FEED, TOY = (cage.PLATE_TAGS[n] for n in ("shock", "feed", "toy"))
SOUTH = -math.pi / 2


def _row() -> Places:
  return Places(ids=PLATE_TAG_IDS, fixtures=(cage.sign_row(),))


def _sign(name: str) -> tuple[float, float]:
  return cage.sign_xy(home.LAB_CAGE_XY, name)


# ---- 1. the memory ----------------------------------------------------------------


def test_a_place_is_one_tag_merged_by_identity_and_follows_new_looks():
  """RackFinder's rule: a decoded id is the same place wherever the frame
  put it -- gated by distance, a drifted look made a second landmark -- and
  a new look carries a floored weight, so a drifted belief re-learns it."""
  p = _row()
  assert p.see(99, 1.0, 1.0, 0.0, SOUTH) is None and len(p) == 0, "not a place"
  p.see(FEED, 1.0, 2.0, 0.0, 0.0)
  p.see(FEED, 3.0, 2.0, 1.0, 0.0)              # 2 m off: the same place, blended
  assert len(p) == 1 and p.get(FEED).n == 2 and p.get(FEED).x == pytest.approx(2.0)
  for t in range(2, 40):
    p.see(FEED, 3.0, 2.0, float(t), 0.0)
  assert p.get(FEED).x == pytest.approx(3.0, abs=1e-3), "stuck in the old frame"
  assert p.get(FEED).seen_t == 39.0 and p.get(FEED).first_t == 0.0


def test_a_facing_comes_off_the_row_then_the_tag_then_the_view():
  p = _row()
  sx, sy = _sign("feed")
  p.see(FEED, sx, sy, 0.0, view=-1.2)
  assert p.facing(FEED) == (-1.2, "view")
  p.see(FEED, sx, sy, 1.0, view=-1.2, tag_facing=SOUTH + 0.01)
  f, src = p.facing(FEED)
  assert src == "tag" and f == pytest.approx(SOUTH + 0.01)
  # a second sign of the row: the facing off their baseline, whatever the
  # tag's own rotation said
  tx, ty = _sign("toy")
  p.see(TOY, tx, ty, 2.0, view=-2.5)
  f, src = p.facing(FEED)
  assert src == "fixture" and f == pytest.approx(SOUTH, abs=1e-6)
  assert p.facing(TOY) == (pytest.approx(SOUTH, abs=1e-6), "fixture")


def test_a_sign_not_yet_seen_is_where_the_rows_drawing_puts_it():
  """How a robot that has read one sign walks to the next rather than
  searching the room for it -- the drawing is the directions' ("a metre
  apart, from left to right...")."""
  p = _row()
  assert p.expected(FEED) is None, "nothing to go on"
  sx, sy = _sign("shock")
  p.see(SHOCK, sx, sy, 0.0, view=SOUTH + 0.3, tag_facing=SOUTH)
  fx, fy, ff = p.expected(FEED)
  assert (fx, fy) == pytest.approx(_sign("feed"), abs=1e-6) and ff == pytest.approx(SOUTH)
  # ...off two, fitted: a rotated row stays a row
  q = Places(ids=PLATE_TAG_IDS, fixtures=(Fixture(layout={35: (-1.0, 0.0), 36: (0.0, 0.0),
                                                           37: (1.0, 0.0)}, facing=SOUTH),))
  c, s = math.cos(0.4), math.sin(0.4)
  q.see(35, -c, -s, 0.0, view=0.0)
  q.see(37, c, s, 0.0, view=0.0)
  x, y, f = q.expected(36)
  assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9) and f == pytest.approx(SOUTH + 0.4)


def test_a_true_death_forgets_the_places_and_a_restart_keeps_them():
  p = _row()
  sx, sy = _sign("feed")
  p.see(FEED, sx, sy, 5.0, view=SOUTH, tag_facing=SOUTH)
  state = json.loads(json.dumps(p.kept_state()))           # rides a JSON file
  q = _row()
  q.restore_kept(state + [{"tag": 99, "x": 0, "y": 0, "seenT": 0, "firstT": 0}])
  assert [(r.tag, r.x, r.y, r.tag_facing) for r in q] == [(FEED, sx, sy, SOUTH)]
  p.forget()
  assert len(p) == 0 and p.expected(FEED) is None


# ---- 2. the signs ------------------------------------------------------------------


@pytest.fixture(scope="module")
def quad_world():
  return lw.home_spec().compile()


def test_the_signs_stand_where_the_rows_drawing_says_and_only_on_legs(quad_world):
  m = quad_world
  for name in cage.PLATE_NAMES:
    b = m.body(f"lab_{name}_sign").id
    assert tuple(m.body_pos[b][:2]) == pytest.approx(_sign(name))
    tag = m.geom(f"lab_{name}_sign_tag").id
    assert m.mat(int(m.geom_matid[tag])).name == f"tagmat{cage.PLATE_TAGS[name]}"
    # ...the pad the drawing puts in front of it is the plate
    pad = m.body_pos[m.body(f"lab_{name}_plate").id][:2]
    assert cage.pad_from_sign(*_sign(name), SOUTH) == pytest.approx(tuple(pad))
  layout = cage.sign_row().layout
  for name in cage.PLATE_NAMES:
    lx, ly = layout[cage.PLATE_TAGS[name]]
    assert (home.LAB_CAGE_XY[0] + lx, home.LAB_CAGE_XY[1] + ly) == pytest.approx(_sign(name))
  rover = mujoco.MjModel.from_xml_path("models/home_world.xml")
  assert mujoco.mj_name2id(rover, mujoco.mjtObj.mjOBJ_BODY, "lab_feed_sign") < 0, \
    "the rover's pass through a pad runs where a sign stands"


# ---- 3. the body -------------------------------------------------------------------


def _quad(model, x=25.0, y=2.0, yaw=math.pi / 2):
  cfg = world_config(QUAD_HOME)
  body = qb.QuadBody(model, mujoco.MjData(model), realtime=False,
                     grid_bounds=cfg["grid_bounds"])
  body.mission.start_at(x, y, yaw)
  return body


def test_one_look_from_the_lab_remembers_the_signs_where_they_are(quad_world):
  body = _quad(quad_world)
  m = body.mission
  try:
    assert sorted(m.look_for_places()) == [SHOCK, FEED, TOY]
    for name in cage.PLATE_NAMES:
      p = m.places.get(cage.PLATE_TAGS[name])
      sx, sy = _sign(name)
      assert math.hypot(p.x - sx, p.y - sy) < 0.02, (name, p)
      f, src = m.places.facing(p.tag)
      assert src == "fixture" and abs(f - SOUTH) < math.radians(1.0)
    # ...and the press is measured off them: the torso short of the pad's
    # centre, facing the sign
    px, py = cage.pad_from_sign(*_sign("feed"), SOUTH)
    x, y, h = m.press_pose(FEED)
    assert (x, y) == pytest.approx((px, py - cage.PRESS_BACK_M), abs=0.03)
    assert abs(math.remainder(h - math.pi / 2, 2 * math.pi)) < math.radians(1.0)
  finally:
    body.close()


def _near_pad(points, pads) -> float:
  """The nearest any segment of a path comes to a pad's centre, m."""
  best = math.inf
  for (ax, ay), (bx, by) in zip(points, points[1:]):
    for px, py in pads:
      dx, dy = bx - ax, by - ay
      k = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy or 1.0)))
      best = min(best, math.hypot(ax + k * dx - px, ay + k * dy - py))
  return best


def test_a_sign_known_only_by_where_it_was_seen_from_keeps_every_pad_it_could_mean_out(
    quad_world):
  """Its face known only by the direction it was seen from -- up to 70 deg
  off -- a sign's pad is anywhere `SIGN_BEHIND_M` round it, and all of that
  is kept out; once its face is known, only its pad. Premise: off the view,
  the pad's own disc misses the pad."""
  body = _quad(quad_world, 20.0, 0.0, 0.0)
  m = body.mission
  try:
    sx, sy = _sign("feed")
    view = SOUTH + math.radians(60.0)
    m.places.see(FEED, sx, sy, 0.0, view=view)
    assert m.places.facing(FEED)[1] == "view"
    pad = cage.pad_from_sign(sx, sy, SOUTH)
    assert math.dist(pad, cage.pad_from_sign(sx, sy, view)) > lp.PAD_KEEP_OUT_M, "premise"
    mask = m.keep_out()
    for deg in range(0, 360, 15):
      cx, cy = m.grid.world_to_cell(*cage.pad_from_sign(sx, sy, math.radians(deg)))
      assert mask[cy, cx], deg
    m.places.see(FEED, sx, sy, 1.0, view=view, tag_facing=SOUTH)
    mask = m.keep_out()
    cx, cy = m.grid.world_to_cell(*pad)
    bx, by = m.grid.world_to_cell(sx, sy + cage.SIGN_BEHIND_M)
    assert mask[cy, cx] and not mask[by, bx]
  finally:
    body.close()


def test_the_planner_keeps_off_every_pad_it_knows(quad_world, monkeypatch):
  """The other plates avoided because the robot has seen their signs, not by
  lanes laid from given coordinates: across the row it plans round its
  ends. Premise: without the pads kept out, the same plan walks over the
  feed plate."""
  body = _quad(quad_world, 25.0, 3.0)
  m = body.mission
  try:
    m.look_for_places()
    pads = [cage.pad_from_sign(p.x, p.y, m.places.facing(p.tag)[0]) for p in m.places]
    assert len(pads) == 3
    mask = m.keep_out()
    grid = m._planning_grid()
    for px, py in pads:
      cx, cy = m.grid.world_to_cell(px, py)
      assert mask[cy, cx] and grid[cy, cx] > OCC_THRESH
    route = [m.pose_xy(), *m._plan_to(25.0, 5.1)]
    assert _near_pad(route, pads) > 0.45, "a route over a known plate"
    m._plan_memo = None
    monkeypatch.setattr(type(m), "keep_out", lambda self: None)
    straight = [m.pose_xy(), *m._plan_to(25.0, 5.1)]
    assert _near_pad(straight, pads) < 0.2, "premise: the plan ignores the pads"
  finally:
    body.close()


def test_the_walking_look_is_paid_for_by_walking(quad_world, monkeypatch):
  """A look is a render (32 ms on the served box): a robot standing still
  spends none, and walking or turning buys one at most every
  `LOOK_EVERY_S`."""
  body = _quad(quad_world)
  m = body.mission
  looks = []
  monkeypatch.setattr(m, "detect_board", lambda: looks.append(float(m.data.time)) or {})
  try:
    m._place_step()
    assert len(looks) == 1, "the first step looks"
    m.data.time += 30.0
    m._place_step()
    assert len(looks) == 1, "standing still bought a render"
    m.odo.x += lp.LOOK_MOVED_M + 0.05
    m._place_step()
    assert len(looks) == 2
    m.odo.yaw += lp.LOOK_TURNED_RAD + 0.05
    m._place_step()
    assert len(looks) == 2, "sooner than LOOK_EVERY_S"
    m.data.time += lp.LOOK_EVERY_S
    m._place_step()
    assert len(looks) == 3
    m.data.time += 10.0
    m.odo.x += 2.0
    m.posture = qb.LYING
    m._place_step()
    assert len(looks) == 3, "lying down it does not look"
  finally:
    body.close()


def test_a_press_stands_the_front_feet_on_the_pad_and_the_fork_off_the_sign(quad_world):
  """The press pose's arithmetic against the body as built: its front feet
  inside the pad, its hind feet off it, and its folded fork -- the body's
  front -- short of the sign."""
  body = _quad(quad_world, 0.0, 0.0, 0.0)
  m = body.mission
  try:
    d, root = m.data, m.root
    mujoco.mj_forward(m.model, d)
    rot = d.xmat[root].reshape(3, 3)
    feet = [float((rot.T @ (d.site_xpos[s] - d.xpos[root]))[0]) for s in m.feet]
    arm = [g for g in range(m.model.ngeom)
           if m.model.body(int(m.model.geom_bodyid[g])).name.startswith("arm_")
           and m.model.geom_contype[g]]
    front = max(float((rot.T @ (d.geom_xpos[g] - d.xpos[root]))[0])
                + float(m.model.geom_rbound[g]) for g in arm)
  finally:
    body.close()
  # in the pad's frame, x toward its sign: the torso at -PRESS_BACK_M
  torso = -cage.PRESS_BACK_M
  fore = [torso + f for f in feet if f > 0]
  hind = [torso + f for f in feet if f < 0]
  assert all(-PLATE_HALF < x + CHOSEN.foot_r and x + CHOSEN.foot_r < PLATE_HALF for x in fore)
  assert all(x + CHOSEN.foot_r < -PLATE_HALF for x in hind), hind
  assert torso + front < cage.SIGN_BEHIND_M - 0.1, (torso + front, "the fork meets the sign")


def test_places_ride_a_restart_with_the_map_and_go_with_it_at_a_true_death(quad_world):
  body = _quad(quad_world)
  m = body.mission
  other = None
  try:
    m.look_for_places()
    m.grid.grid[10:20, 10:20] = 3.0
    m.low[5, 5] = 3.0
    state, arrays = m.kept_state()
    json.dumps(state)
    other = _quad(quad_world)
    assert other.mission.restore_kept(state, arrays)
    assert [(p.tag, p.x, p.y) for p in other.mission.places] == \
        [(p.tag, p.x, p.y) for p in m.places]
    # ...and a map that does not come back brings no places
    third = qb.QuadBody(quad_world, mujoco.MjData(quad_world), realtime=False,
                        grid_bounds=(-3, -3, 7, 7))
    assert not third.mission.restore_kept(state, arrays) and len(third.places) == 0
    third.close()
    body.forget_world()
    assert not m.grid.grid.any() and not m.low.any() and len(m.places) == 0
    assert m.keep_out() is None
  finally:
    body.close()
    if other is not None:
      other.close()


# ---- 4. the search ------------------------------------------------------------------


def test_the_search_goes_out_from_the_address_not_from_the_robot():
  """Nearest the ROBOT first walked the first flight out of the house and
  into the street (600 s, not found); round the address, the house is."""
  near, never = (10.0, 0.0), lambda x, y: False
  first = lp.next_viewpoint(near, (-50.0, 0.0), [], set(), never)
  assert first == near
  second = lp.next_viewpoint(near, (-50.0, 0.0), [], {first}, never)
  assert math.dist(second, near) == pytest.approx(lp.SEARCH_STEP_M)
  assert second == (near[0] - lp.SEARCH_STEP_M, near[1]), "a tie goes to the robot's side"
  # ...covered, tried and walled viewpoints are skipped, and the lattice ends
  def walls(x, y):
    return x > near[0]
  seen = [near]
  tried = set()
  while (vp := lp.next_viewpoint(near, (0.0, 0.0), seen, tried, walls)) is not None:
    assert vp[0] <= near[0] and math.dist(vp, near) <= lp.SEARCH_RADIUS_M + 1e-9
    assert math.dist(vp, near) >= max((math.dist(v, near) for v in tried), default=0.0) - 1e-9
    tried.add(vp)
  assert near not in tried and len(tried) > 5
  # ...and the MAP BEFORE THE LATTICE: a frontier -- a door into a room not
  # yet seen -- beats every lattice point, the nearest the address first;
  # a lattice point already looked over does not count
  door, far_door = (near[0] + 4.0, near[1]), (near[0] - 6.0, near[1])
  assert lp.next_viewpoint(near, near, [], [], never, [far_door, door]) == door
  assert lp.next_viewpoint(near, near, [], [door], never, [far_door, door]) == far_door
  assert lp.next_viewpoint(near, near, [], [], never, [(near[0] + 9.0, near[1])]) == near, \
      "a frontier beyond the radius"
  over = lp.next_viewpoint(near, near, [], [], never, seen=lambda x, y: math.dist((x, y), near) < 3)
  assert math.dist(over, near) >= 3, "a lattice point it had looked over"


def test_a_find_goes_where_the_row_says_then_to_the_address_then_round_it(quad_world):
  """The order, on a body whose walks and looks are stubbed: the feed plate
  not yet seen but the shock plate's sign read, the first walk is to where
  the row puts the feed plate's standoff; then the address; then the
  lattice round it, the address itself first. And its own interrupt ends
  it, as it ends a walk."""
  body = _quad(quad_world, 20.0, 0.0, 0.0)
  m = body.mission
  went = []

  def drive(x, y, timeout=90.0, stop=None):
    went.append((round(x, 3), round(y, 3)))
    return tick.result(False)

  try:
    m.drive_to_routine = drive
    m.face_routine = lambda h: tick.result(True)
    m.look_for_places = lambda: []
    m._look_around_routine = lambda stop: tick.result(False)
    sx, sy = _sign("shock")
    m.places.see(SHOCK, sx, sy, 0.0, view=SOUTH, tag_facing=SOUTH)
    near = (25.2, -2.4)
    rec = body.run(m.find_routine(FEED, near=near, patience=600.0))
    fx, fy = _sign("feed")
    assert went[0] == pytest.approx((fx, fy - lp.STANDOFF_M), abs=1e-3)
    assert went[1] == near and went[2] == near
    assert math.dist(went[3], near) == pytest.approx(lp.SEARCH_STEP_M)
    assert rec["why"] == "not found" and rec["guesses"] == 1 and not rec["found"]
    went.clear()
    rec = body.run(m.find_routine(FEED, near=near, patience=600.0, stop=lambda: bool(went)))
    assert rec["why"] == "interrupted" and len(went) == 1
  finally:
    body.close()


def test_a_find_ends_on_its_patience_and_hands_each_walk_only_what_is_left(quad_world):
  """Each walk is handed what is LEFT of the find's patience, never more
  (a 1 s floor walked a program past its budget), and asks the find's time
  as it goes; the find says it ran out, at its patience and not after."""
  body = _quad(quad_world, 20.0, 0.0, 0.0)
  m = body.mission
  timeouts, halted = [], []

  def drive(x, y, timeout=90.0, stop=None):
    timeouts.append(round(timeout, 3))
    m.data.time += min(timeout, 60.0)          # every walk takes a minute
    halted.append(bool(stop()))
    return tick.result(False)

  try:
    m.drive_to_routine = drive
    m.face_routine = lambda h: tick.result(True)
    m.look_for_places = lambda: []
    m._look_around_routine = lambda stop: tick.result(False)
    t0 = float(m.data.time)
    rec = body.run(m.find_routine(FEED, near=(25.2, -2.4), patience=120.5))
    assert timeouts == [120.5, 60.5, 0.5], "the address, a viewpoint, and what was left"
    assert halted == [False, False, True], "a walk is told when the time is up"
    assert rec["why"] == "out of time" and not rec["found"]
    assert float(m.data.time) - t0 == pytest.approx(120.5)
  finally:
    body.close()


def test_a_press_steps_onto_a_plate_only_with_time_to_step_off_it(quad_world):
  """The walk in, the hold and the walk out run to their end, so a press
  starts them only with `FINAL_S` of its patience left, and the walk to
  the standoff is handed the rest."""
  body = _quad(quad_world, 25.0, 2.0, -math.pi / 2)
  m = body.mission
  timeouts, walked_in = [], []

  def drive(x, y, timeout=90.0, stop=None):
    timeouts.append(round(timeout, 3))
    return tick.result(True)

  try:
    m.places.see(FEED, *_sign("feed"), 0.0, view=SOUTH, tag_facing=SOUTH)
    m.drive_to_routine = drive
    m.face_routine = lambda h: tick.result(True)
    m.look_for_places = lambda: [FEED]
    m._press_walk_in_routine = lambda tag: walked_in.append(tag) or tick.result("stopped")
    m._hold_on_routine = lambda pad: tick.result(True)
    m._press_back_out_routine = lambda: tick.result(None)
    rec = body.run(m.press_routine(FEED, patience=lp.FINAL_S - 1.0))
    assert rec["why"] == "out of time" and not walked_in and not timeouts
    rec = body.run(m.press_routine(FEED, patience=lp.FINAL_S + 30.0))
    assert rec["pressed"] and walked_in == [FEED] and timeouts == [30.0]
    # ...and a walk to the standoff that ran a second over its time ends it there
    m.drive_to_routine = lambda x, y, timeout=90.0, stop=None: (
      setattr(m.data, "time", m.data.time + timeout + 1.0) or tick.result(True))
    rec = body.run(m.press_routine(FEED, patience=lp.FINAL_S + 30.0))
    assert rec["why"] == "out of time" and walked_in == [FEED]
  finally:
    body.close()


def test_a_place_forgotten_under_a_press_ends_it_backed_off_the_plate(quad_world):
  """A true death forgets the places on the physics seam while the dying
  robot's errand still runs: the press ends "not found" -- backed off the
  plate if it was on its way onto it -- and never on a forgotten pose."""
  body = _quad(quad_world, 25.0, 2.0, -math.pi / 2)
  m = body.mission
  backed = []

  def forget():
    m.places.forget()

  try:
    m.face_routine = lambda h: tick.result(True)
    m._press_back_out_routine = lambda: backed.append(True) or tick.result(None)
    # ...under the walk in: the sign read, then forgotten
    m.places.see(FEED, *_sign("feed"), 0.0, view=SOUTH, tag_facing=SOUTH)
    m.drive_to_routine = lambda x, y, timeout=90.0, stop=None: tick.result(True)
    m.look_for_places = lambda: forget() or [FEED]
    rec = body.run(m.press_routine(FEED, patience=300.0))
    assert rec["why"] == "not found" and backed == [True]
    assert rec["attempts"][0]["walkIn"] == "not found"
    # ...on the walk to its standoff
    backed.clear()
    m.places.see(FEED, *_sign("feed"), 1.0, view=SOUTH, tag_facing=SOUTH)
    m.drive_to_routine = lambda x, y, timeout=90.0, stop=None: forget() or tick.result(True)
    rec = body.run(m.press_routine(FEED, patience=300.0))
    assert rec["why"] == "not found" and not backed
    # ...and on a look round for a sign not in view there, before a second try
    m.places.see(FEED, *_sign("feed"), 2.0, view=SOUTH, tag_facing=SOUTH)
    m.drive_to_routine = lambda x, y, timeout=90.0, stop=None: tick.result(True)
    m.look_for_places = lambda: []
    m._look_around_routine = lambda stop: forget() or tick.result(False)
    rec = body.run(m.press_routine(FEED, patience=300.0))
    assert rec["why"] == "not found" and rec["attempts"][0]["why"] == "lost" and not backed
  finally:
    body.close()


# ---- 5. the verbs ----------------------------------------------------------------------


def test_find_is_a_legs_verb_where_the_world_has_places_and_press_where_its_lab_is(
    monkeypatch):
  facts = world_facts(QUAD_HOME)
  assert facts.places == PLATE_TAG_IDS and facts.plates == PLATE_TAG_IDS
  assert "find" in facts.verbs and "press" in facts.verbs
  rover = world_facts("home")
  assert rover.verbs is None and rover.places == ()
  assert st.check_step(st.VERBS["find"], {"tag": FEED, "x": 25.2, "y": -2.4}, facts) == []
  assert "no place" in st.check_step(st.VERBS["find"], {"tag": 99, "x": 0, "y": 0}, facts)[0]
  assert "outside the map" in st.check_step(st.VERBS["find"],
                                            {"tag": FEED, "x": 999.0, "y": 0.0}, facts)[0]
  assert "no place" in st.check_step(st.VERBS["find"], {"tag": FEED, "x": 1, "y": 1}, rover)[0]
  assert st.check_step(st.VERBS["press"], {"tag": FEED}, facts) == []
  assert "no plate" in st.check_step(st.VERBS["press"], {"tag": 99}, facts)[0]
  # ...and `press` only beside the lab and its rule (#403 brought them back):
  # a world with the plates' tags and no lab has `find` alone
  real = world_config

  def without_lab(world):
    cfg = real(world)
    return {k: v for k, v in cfg.items() if k != "lab"} if world == QUAD_HOME else cfg
  from pluggybot import lifecycle
  monkeypatch.setattr(lifecycle, "world_config", without_lab)
  unlabbed = world_facts(QUAD_HOME)
  assert "press" not in unlabbed.verbs and unlabbed.plates == ()
  assert "not on this body" in st.check_step(st.VERBS["press"], {"tag": FEED}, unlabbed)[0]


def _stub_quad_life(**kw):
  return stub_life(QUAD_HOME, **kw)


def test_the_verbs_say_what_was_found_and_why_not():
  life = _stub_quad_life()
  try:
    lost = life.body.run(st.run_verb(life, st.VERBS["find"],
                                     {"tag": FEED, "x": 25.2, "y": -2.4}))
    assert not lost["ok"] and "did not find tag 36 round (25.2, -2.4)" in lost["reason"]
    pressed = life.body.run(st.run_verb(life, st.VERBS["press"], {"tag": FEED}))
    assert not pressed["ok"] and "find" in pressed["reason"]
    life.body.places.see(FEED, 25.0, 4.83, 0.0, SOUTH)
    found = life.body.run(st.run_verb(life, st.VERBS["find"],
                                      {"tag": FEED, "x": 25.2, "y": -2.4}))
    assert found["ok"] and found["at"] == [25.0, 4.83]
    assert life.body.run(st.run_verb(life, st.VERBS["press"], {"tag": FEED}))["ok"]
    assert life.body.found == [FEED, FEED] and life.body.pressed == [FEED, FEED]
  finally:
    life.body.close()


def test_find_and_press_keep_the_programs_budget_and_pass_on_every_why(monkeypatch):
  """Both verbs are handed their own patience, and never past the program's
  budget (`run_verb(until=)`); a why the verb has no words for is passed
  on, never read as running out of time."""
  life = _stub_quad_life()
  asked, whys = [], ["out of time", "the rover keeps no places"]

  def find(tag, near, patience, stop=None):
    asked.append(("find", round(patience, 3)))
    return tick.result({"tag": tag, "found": False, "why": whys.pop(0), "seconds": 1.0})

  def press(tag, patience, stop=None):
    asked.append(("press", round(patience, 3)))
    return tick.result({"tag": tag, "pressed": False, "why": "out of time"})

  monkeypatch.setattr(life.body, "find_tag_routine", find)
  monkeypatch.setattr(life.body, "press_plate_routine", press)
  args = {"tag": FEED, "x": 25.2, "y": -2.4}
  try:
    out = life.body.run(st.run_verb(life, st.VERBS["find"], args))
    assert "ran out of time" in out["reason"]
    out = life.body.run(st.run_verb(life, st.VERBS["find"], args,
                                    until=float(life.data.time) + 50.0))
    assert out["reason"] == "did not find tag 36: the rover keeps no places"
    out = life.body.run(st.run_verb(life, st.VERBS["press"], {"tag": FEED}))
    assert "ran out of time" in out["reason"]
    life.body.run(st.run_verb(life, st.VERBS["press"], {"tag": FEED},
                              until=float(life.data.time) + 50.0))
    assert asked == [("find", st.FIND_PATIENCE_S), ("find", 50.0),
                     ("press", st.PRESS_PATIENCE_S), ("press", 50.0)]
  finally:
    life.body.close()


# ---- 5b. the offer: an address and directions, never finer ----------------------------


def test_the_address_is_the_houses_middle_off_by_its_error_and_within_its_word():
  data = addresses.load()
  for house, (ex, ey) in data["houses"].items():
    a = addresses.address(house)
    mx, my = addresses.house_middle(house)
    off = math.hypot(a["x"] - mx, a["y"] - my)
    assert 2.0 < off <= a["withinM"], (house, off, "a few metres, and inside its word")
    assert (a["x"], a["y"]) == pytest.approx((mx + ex, my + ey), abs=0.06)


def test_the_directions_pass_the_dispatcher_test():
  """TaskPattern.md §2: could someone who cannot see the room have written
  it? A number in the directions is one of the area's tags, and never a
  position; every tag the area has, it names."""
  for name, spec in addresses.load()["areas"].items():
    numbers = {int(n) for n in re.findall(r"\d+(?:\.\d+)?", spec["directions"])}
    tags = {int(t) for t in spec["tags"]}
    assert numbers == tags, (name, numbers - tags)
  assert set(addresses.tag_names()) == set(PLATE_TAG_IDS)


def test_a_job_at_a_task_area_carries_its_address_and_directions_and_no_other_does():
  from pluggybot.economy.cadence import Cadence, TaskProducer
  from pluggybot.economy.tasks import TaskBoard
  from pluggybot.lifecycle import task_producer
  board = TaskBoard(path=None)
  assert task_producer(board, QUAD_HOME).facts["places"] == addresses.areas()
  assert "places" not in task_producer(TaskBoard(path=None), "home").facts
  beat = Cadence(world=QUAD_HOME, first_at_s=0.0, every_s=1.0, ttl_s=100.0,
                 cooldown_s=0.0, max_offered=5, initial=1, kinds={"feed_mouse": {}})
  maker = TaskProducer(board, beat, {"cage": ["lab"], "module": ["module_lcd"]},
                       facts={"places": addresses.areas()})
  job = maker._offer("feed_mouse", "lab", 0.0)
  shown = job.as_context(0.0)
  assert shown["address"] == addresses.address("facility")
  assert shown["directions"] == addresses.load()["areas"]["lab"]["directions"]
  assert not re.search(r"\d", job.description), "a position in the description"
  assert job.as_dict()["params"]["address"] == shown["address"], "on the wire too"
  other = maker._offer("fetch_module", "module_lcd", 0.0)
  assert not {"address", "directions"} & set(other.as_context(0.0))


# ---- 5c. the mind's view -----------------------------------------------------------------


def test_the_mind_sees_the_places_it_found_by_name_where_and_when():
  boss = ov.Overseer(ov.Menu.for_world(QUAD_HOME), client=1)
  life = _stub_quad_life(overseer=boss)
  try:
    assert overseer_context(life)["places"] == []
    life.body.places.see(FEED, 25.004, 4.8312, 0.0, SOUTH)
    life.body.run(life.body.hold_routine(2.0))
    assert places_context(life) == [{"tag": FEED, "name": "the lab's feed plate",
                                     "at": [25.0, 4.83], "seenSAgo": 2}]
    assert overseer_context(life)["places"] == places_context(life)
    life.body.places = None                     # a body that keeps none
    assert "places" not in overseer_context(life)
  finally:
    life.body.close()


def test_a_legs_world_is_never_shown_where_its_furniture_stands(monkeypatch):
  """The lab's block names no position on a body that finds its places:
  #403 brings the lab back to legs, and the rover's `lab.bench` came with
  it. The rover's world keeps its surveyed furniture until it goes."""
  from dataclasses import replace
  from types import SimpleNamespace

  from pluggybot import lifecycle
  from pluggybot.lifecycle import HubLifecycle
  real = world_config

  def with_lab(world):
    cfg = real(world)
    return {**cfg, "lab": real("home")["lab"]} if world == QUAD_HOME else cfg
  monkeypatch.setattr(lifecycle, "world_config", with_lab)
  monkeypatch.setattr(HubLifecycle, "cage", property(lambda self: SimpleNamespace(
    context=lambda data, root: {"inRoom": False, "mouse": None})))
  shown = {}
  for world in (QUAD_HOME, "home"):
    boss = ov.Overseer(replace(ov.Menu.for_world(world), lab="lab"), client=1)
    life = stub_life(world, overseer=boss)
    try:
      shown[world] = overseer_context(life)["lab"]
    finally:
      life.body.close()
  assert "bench" not in shown[QUAD_HOME] and "route" not in shown[QUAD_HOME]
  assert "bench" in shown["home"], "premise: the rover's world still shows it"


def test_a_place_found_is_said_once_in_history_and_one_a_restart_brought_back_is_not(
    tmp_path):
  from pluggybot.body import StubBody
  from pluggybot.mind.thoughts import HISTORY, ThoughtFiles
  cfg = world_config(QUAD_HOME)
  body = StubBody(rack=cfg["rack"], grid_bounds=cfg["grid_bounds"])
  body.places.see(SHOCK, 24.0, 4.83, 0.0, SOUTH)          # as a restart brings it back
  thoughts = ThoughtFiles.open(str(tmp_path / "thoughts"))
  life = stub_life(QUAD_HOME, body=body, thoughts=thoughts)

  def said(what):
    return sum(what in line for line in thoughts.volatile()[HISTORY])

  try:
    life.body.run(life.body.hold_routine(0.01))
    assert said("found") == 0, "a place a restart brought back was found again"
    body.places.see(FEED, 25.0, 4.83, 1.0, SOUTH)
    life.body.run(life.body.hold_routine(0.01))
    body.places.see(FEED, 25.0, 4.83, 2.0, SOUTH)
    life.body.run(life.body.hold_routine(0.01))
    assert said("found the lab's feed plate (tag 36) at (25.0, 4.8)") == 1
    body.forget_world()                                    # a true death's
    body.places.see(FEED, 25.0, 4.83, 3.0, SOUTH)
    life.body.run(life.body.hold_routine(0.01))
    assert said("found the lab's feed plate") == 2, "the new robot found it too"
  finally:
    life.body.close()


# ---- 5d. a true death ----------------------------------------------------------------------


def test_a_true_death_clears_the_map_and_the_places_and_grants_the_start():
  ledger = Ledger()
  life = _stub_quad_life(ledger=ledger, mortal=True, start_points=30)
  seen = []
  life.on_event.append(seen.append)
  try:
    life.body.places.see(FEED, 25.0, 4.83, 0.0, SOUTH)
    life.body.grid.grid[3:6, 3:6] = 2.0
    ledger.robots["pluggybot"]["hearts"] = 1
    life._die("flat", "the pack ran out")
    assert life.body.forgot == 1 and len(life.body.places) == 0
    assert not life.body.grid.grid.any() and not life.floor_explored
    assert ledger.balance() == 30 and ledger.granted() == 30
    acct = ledger.robots["pluggybot"]
    assert acct["balance"] == acct["granted"] == 30 and acct["earned"] == 0
    event = next(e for e in seen if e["type"] == "true_death")
    assert event["granted"] == 30
  finally:
    life.body.close()
  # ...and none where the world gives none, as before
  plain = Ledger()
  other = _stub_quad_life(ledger=plain, mortal=True)
  try:
    plain.robots["pluggybot"]["hearts"] = 1
    other._die("flat", "the pack ran out")
    assert plain.balance() == 0 and plain.granted() == 0
  finally:
    other.body.close()


def test_the_ledger_books_a_start_as_granted_and_keeps_every_term_across_a_load(tmp_path):
  """`granted` is a term in the identity and never `earned`. And a LOAD
  keeps the transfers' terms too: before this, `given` and `received` were
  written and never read back, so a restart after a gift broke the
  identity."""
  path = tmp_path / "ledger.json"
  ledger = Ledger(robots=("pluggybot", "r2_pluggybot"), path=path)
  ledger.robots["pluggybot"]["balance"] = 40
  ledger.transfer(15, "r2_pluggybot")
  out = ledger.archive("r2_pluggybot", start=30)
  assert out["granted"] == 30
  again = Ledger(robots=("pluggybot", "r2_pluggybot"), path=path)
  assert again.given("pluggybot") == 15 and again.granted("r2_pluggybot") == 30
  a = again.robots["r2_pluggybot"]
  assert (a["granted"] + a["earned"] - a["consumed"] - a["spent"] - a.get("given", 0)
          + a.get("received", 0)) == a["balance"] == 30
  assert again.snapshot()["r2_pluggybot"]["granted"] == 30
