"""The claw on the arm (issue #407): the eye that finds a cube, the grip
judged off the world, the search in front of a cube's area, the verbs a
program names, and the arm's motors as a scale.

The flown claims -- a tower stacked from the dock and graded by its own
predicate, the bench weighed -- are ladder A's (`scripts/solve.py --feature
tower|bench`, reported on #407). Every rule they stand on is pinned here
without a mission:

  1. A CUBE SHOWS ITS TAG ON EVERY FACE, and the face seen most nearly
     square-on is the one read -- keyed by id, the last decoded won, and
     lying in front of a cube that was its top seen edge-on, 16 mm over
     the face it faced; each face is cut at its known height, never read
     off PnP's range, and the cube in the jaws is PnP's.
  2. THE GRIP IS JUDGED OFF THE WORLD: both pads within a hair of one body
     that can move -- by distance, since in the flight the stiff pads'
     contacts came and went -- and the floor between the pads is no cube.
  3. THE SEARCH STOPS ALONG THE AREA'S ROW, each stop planned off the tags
     as they are remembered when it is walked to.
  4. The verbs: an empty fork fetches the claw first, another tool refuses,
     a verb that did not do it says why, and a survey's count rides its
     verdict and goes on the LCD's face only while the LCD is on the fork.
  5. The arm's drivers report torque in counts, with the current sense's
     noise: the bench's scale, as a GDS68 field is the legs'.
"""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot.challenge.stack import BLOCK_HALF, block_xml
from pluggybot.legs import arm as am
from pluggybot.legs import claw as lc
from pluggybot.legs import rack as rk
from pluggybot.lifecycle import QUAD_HOME
from pluggybot.perception import encoders
from pluggybot.procedure import steps as st
from pluggybot.tools import claw as tc
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

H = 0.20                          # the torso's centre over the floor
F, CX, CY = 900.0, 640.0, 360.0   # a pinhole: focal length and centre, px


def _mount(pitch_deg: float = 30.0):
  """A camera 0.25 m ahead of the torso's centre, 0.05 m up, pitched down:
  its axes (MuJoCo's: x right, y up, looking down -z) in the torso frame."""
  p = math.radians(pitch_deg)
  fwd = np.array([math.cos(p), 0.0, -math.sin(p)])
  right = np.array([0.0, -1.0, 0.0])
  up = np.cross(right, fwd)
  return np.column_stack([right, up, -fwd]), np.array([0.25, 0.0, 0.05])


def _decode(tag, centre, outward, r_mount, p_mount):
  """What the detector reports of a tag face whose middle is `centre` and
  whose outward normal is `outward`, both in the torso frame: its centre
  pixel, PnP's translation and the tag's z (into the face), the camera's
  frame being x right, y down, z ahead."""
  q = r_mount.T @ (np.asarray(centre) - p_mount)
  t = np.array([q[0], -q[1], -q[2]])
  n = r_mount.T @ -np.asarray(outward, dtype=float)
  return {"id": tag, "t": tuple(t), "center": (CX + F * t[0] / t[2], CY + F * t[1] / t[2]),
          "normal": (n[0], -n[1], -n[2])}


# ---- 1. the eye -----------------------------------------------------------------


def test_the_face_seen_most_square_on_is_read_and_cut_at_its_height():
  r_mount, p_mount = _mount()
  x, y = 0.55, 0.03
  floor = -H
  side = _decode(20, (x - BLOCK_HALF, y, floor + BLOCK_HALF), (-1, 0, 0), r_mount, p_mount)
  top = _decode(20, (x, y, floor + 2 * BLOCK_HALF), (0, 0, 1), r_mount, p_mount)
  # ...a top seen edge-on reads 16 mm long through PnP: its translation lies
  top_off = {**top, "t": tuple(np.array(top["t"]) * 1.03)}
  for dets in ([side, top_off], [top_off, side]):
    cube = tc.cubes_seen(dets, np.eye(3), r_mount, p_mount, (F, F, CX, CY), H)[20]
    assert cube.face == "side", "the face seen edge-on was read"
    assert (cube.x, cube.y, cube.z) == pytest.approx((x, y, floor + BLOCK_HALF), abs=1e-6)
    assert cube.layer == 0
  # the top alone is cut at the top's height, wherever PnP put it
  cube = tc.cubes_seen([top_off], np.eye(3), r_mount, p_mount, (F, F, CX, CY), H)[20]
  assert cube.face == "top" and (cube.x, cube.y) == pytest.approx((x, y), abs=1e-6)


def test_a_cube_on_a_cube_is_the_second_layer():
  r_mount, p_mount = _mount()
  z = -H + 3 * BLOCK_HALF
  side = _decode(21, (0.6 - BLOCK_HALF, 0.0, z), (-1, 0, 0), r_mount, p_mount)
  cube = tc.cubes_seen([side], np.eye(3), r_mount, p_mount, (F, F, CX, CY), H)[21]
  assert cube.layer == 1 and cube.z == pytest.approx(z, abs=1e-6)


def test_the_cube_in_the_jaws_is_pnps_and_a_tag_not_a_cube_or_ignored_is_not_read():
  """A held cube stands on no layer: cut at one, its ray put it 18 mm long."""
  r_mount, p_mount = _mount()
  centre = np.array([0.45 - BLOCK_HALF, 0.0, -0.05])
  held = _decode(22, centre, (-1, 0, 0), r_mount, p_mount)
  cube = tc.cubes_seen([held], np.eye(3), r_mount, p_mount, (F, F, CX, CY), H, held=(22,))[22]
  assert cube.face == "held"
  assert (cube.x, cube.y, cube.z) == pytest.approx((0.45, 0.0, -0.05), abs=1e-6)
  other = _decode(38, (0.6, 0.0, -H + BLOCK_HALF), (-1, 0, 0), r_mount, p_mount)
  assert tc.cubes_seen([other, held], np.eye(3), r_mount, p_mount, (F, F, CX, CY), H,
                       ignore=(22,)) == {}


# ---- 2. the grip -----------------------------------------------------------------


def _claw_world(cube_z: float | None, fixed_box: bool = False):
  """The claw fixed in space with a cube (or an immovable box) between its
  jaws: what `ClawHand.held` asks of."""
  jaw_z = 1.0 - rk.CLAW_JAW_DROP
  body = (f'<body name="ledge" pos="0 0 {jaw_z:.4f}"><geom type="box" '
          f'size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}"/></body>' if fixed_box else
          block_xml("block_1", 0.0, 0.0, None).replace(
            f'pos="0.0000 0.0000 {BLOCK_HALF:.4f}"', f'pos="0 0 {cube_z:.4f}"'))
  xml = (f'<mujoco><compiler angle="radian"/><option timestep="0.002" integrator="implicitfast"/>'
         f'<worldbody><body name="module_claw" pos="0 0 {1.0 - 0.022}">'
         f'<geom name="plate" type="box" size="0.01 0.02 0.03" mass="0.1"/>{rk.claw_face()}'
         f'</body>{body}</worldbody>'
         f'<actuator>{rk.tool_actuators_xml(("module_claw",))}</actuator></mujoco>')
  m = mujoco.MjModel.from_xml_string(xml)
  return m, mujoco.MjData(m)


def _hand(m, d):
  mission = SimpleNamespace(model=m, data=d, arm=None, arm_spec=am.ArmSpec(),
                            posture="lying", falls=0, carrying="module_claw")
  return tc.ClawHand(mission)


def _close(m, d, closed: bool, gravity_after: int = 0):
  jaws = [m.actuator(j).id for j in rk.CLAW_JAWS]
  full = rk.CLAW_JAW_OPEN - rk.CLAW_JAW_CLOSED
  m.opt.gravity[:] = 0.0
  for k in range(500):
    for a in jaws:
      d.ctrl[a] = full * (max(0.0, 1 - k / 400) if closed else 1.0)
    mujoco.mj_step(m, d)
  m.opt.gravity[:] = (0.0, 0.0, -9.81)
  for _ in range(gravity_after):
    mujoco.mj_step(m, d)


def test_a_cube_in_the_jaws_is_held_at_every_step_for_a_second():
  """Both pads within `HELD_GAP_M` of the cube at every step. (Why by
  distance: in the flight, read off the contact list at one instant, a cube
  in the jaws was "nothing held" -- the pads' contacts came and went under
  the moving body. This fixed claw does not reproduce that; it pins that a
  held cube reads held, step after step.)"""
  m, d = _claw_world(1.0 - rk.CLAW_JAW_DROP)
  _close(m, d, closed=True, gravity_after=200)
  hand = _hand(m, d)
  for _ in range(500):
    mujoco.mj_step(m, d)
    assert hand.held() == "block_1"
  assert hand.held_tag() == 21, "the claw names the cube it holds by its tag"


def test_open_jaws_hold_nothing_and_the_floor_between_the_pads_is_no_cube():
  m, d = _claw_world(1.0 - rk.CLAW_JAW_DROP)
  _close(m, d, closed=False)
  assert _hand(m, d).held() is None
  m, d = _claw_world(None, fixed_box=True)
  _close(m, d, closed=True)
  assert _hand(m, d).held() is None, "something that cannot move was held"


# ---- 3. the search ---------------------------------------------------------------


class _Searcher(lc.CubeWork):
  """The search on its own: an area's two tags remembered where `tags`
  says, a walk that arrives, a look that finds the cube only from `at`."""

  def __init__(self, tags, at=None):
    self.tags = dict(tags)
    self.pose = (0.0, 0.0, 0.0)
    self.cubes, self.walks, self.at = {}, [], at
    self.data = SimpleNamespace(time=0.0)
    self.places = SimpleNamespace(expected=lambda t: self.tags.get(t))

  def look_cubes(self, ignore=(), held=()):
    if self.at is not None and math.dist(self.pose[:2], self.at) < 0.05:
      return {23: object()}
    return {}

  def drive_to_routine(self, x, y, timeout=90.0, stop=None):
    self.walks.append((round(x, 3), round(y, 3)))
    self.pose = (x, y, self.pose[2])
    self.data.time += 10.0
    return True
    yield

  def face_routine(self, heading):
    self.pose = (*self.pose[:2], heading)
    return
    yield

  def _turn_by_routine(self, deg):
    return
    yield


def _run(gen):
  try:
    while True:
      next(gen)
  except StopIteration as stop:
    return stop.value


def test_the_search_stops_along_the_row_in_front_of_the_area_facing_it():
  """Planned on the tags' middle alone, a pair remembered 0.2 m off left a
  cube 0.96 m from the one stop -- past the colour imager's 0.9."""
  s = _Searcher({44: (10.0, 1.45, math.pi), 45: (10.0, 0.55, math.pi)})
  points = s._look_points(23)
  assert len(points) == len(lc.SEARCH_SIDE_M) == 3
  for (x, y, heading), side in zip(points, lc.SEARCH_SIDE_M):
    assert x == pytest.approx(10.0 - lc.SEARCH_OUT_M)
    assert y == pytest.approx(1.0 - side)
    assert heading == pytest.approx(0.0, abs=1e-9), "it faces the tags"
  assert not _run(s._sight_routine(23, (), 1e9, None))
  assert len(s.walks) == 3


def test_each_stop_is_planned_off_the_tags_as_remembered_when_it_is_walked_to():
  """The bench's tags, first seen from the street, are remembered better
  as the robot nears them: a stop planned at the search's start stands
  where the far sighting put the row."""
  s = _Searcher({44: (10.0, 1.45, math.pi), 45: (10.0, 0.55, math.pi)})
  first = s.drive_to_routine

  def nearing(x, y, timeout=90.0, stop=None):
    s.tags = {44: (10.2, 1.45, math.pi), 45: (10.2, 0.55, math.pi)}   # seen from near
    return (yield from first(x, y, timeout, stop))
  s.drive_to_routine = nearing
  _run(s._sight_routine(23, (), 1e9, None))
  assert s.walks[0][0] == pytest.approx(10.0 - lc.SEARCH_OUT_M)
  assert all(x == pytest.approx(10.2 - lc.SEARCH_OUT_M) for x, _ in s.walks[1:])


def test_the_search_ends_once_a_look_has_the_cube():
  s = _Searcher({44: (10.0, 1.45, math.pi), 45: (10.0, 0.55, math.pi)},
                at=(10.0 - lc.SEARCH_OUT_M, 1.0 - lc.SEARCH_SIDE_M[1]))
  assert _run(s._sight_routine(23, (), 1e9, None)) is True
  assert len(s.walks) == 2


# ---- 4. the verbs ----------------------------------------------------------------


def _claw_life():
  life = stub_life(QUAD_HOME)
  life.body.cubes = {20, 21, 22, 23, 24}
  return life


def test_an_empty_fork_fetches_the_claw_first_and_another_tool_refuses():
  life = _claw_life()
  out = life.body.run(st._pick(life, {"tag": 20}))
  assert out["ok"] and life.body.holding == "module_claw" and life.body.held == 20
  life = _claw_life()
  life.body.holding = "module_pen"
  out = life.body.run(st._pick(life, {"tag": 20}))
  assert not out["ok"] and "holds module_pen, not the claw" in out["reason"]
  assert life.body.picked == [], "it walked to the cube with the pen on"


def test_a_cube_verb_that_did_not_do_it_says_why():
  life = _claw_life()
  out = life.body.run(st._place(life, {"tag": 21}))
  assert not out["ok"] and out["reason"] == st.CUBE_WHY["no claw"]
  life.body.holding = "module_claw"
  out = life.body.run(st._place(life, {"tag": 21}))
  assert not out["ok"] and out["reason"] == st.CUBE_WHY["nothing held"]
  out = life.body.run(st._pick(life, {"tag": 30}))
  assert not out["ok"] and out["reason"].startswith("did not find cube 30")
  assert life.body.run(st._pick(life, {"tag": 20}))["ok"]
  assert life.body.run(st._place(life, {"tag": 21}))["ok"]
  assert life.body.stacked == [(20, 21)]


def test_a_survey_count_rides_its_verdict_and_the_lcd_shows_it_while_carried():
  life = _claw_life()
  life.body.counts = {46: 4}
  out = life.body.run(st._survey(life, {"tag": 46}))
  assert not out["ok"] and out["reason"] == st.SURVEY_WHY["not found"].format(tag=46)
  life.body.places.see(46, 9.98, 0.7, 0.0, math.pi)
  shown = []
  life.screen = SimpleNamespace(module="module_lcd",
                                show_count=lambda n, *a, **kw: shown.append(n))
  out = life.body.run(st._survey(life, {"tag": 46}))
  assert out["ok"] and out["count"] == 4 and "coverage" in out
  assert not out["shown"] and shown == [], "a face on an LCD hanging on the rack"
  life.body.holding = "module_lcd"
  out = life.body.run(st._survey(life, {"tag": 46}))
  assert out["shown"] and shown and shown[-1] == 4


# ---- 5. the scale ----------------------------------------------------------------


def test_the_drivers_report_torque_in_counts_with_the_sense_noise():
  """The bench's scale is the arm's elbow as its driver reports it: a
  12-bit field over +-22 N*m, read off a noisy current -- deterministic on
  the step and the motor, as every sensor's noise is."""
  lsb = encoders.TORQUE_LSB
  assert lsb == pytest.approx(44.0 / 4095)
  reads = np.array([encoders.torque_reading(1.0, "arm_elbow", k) for k in range(4000)])
  assert np.allclose(reads / lsb, np.round(reads / lsb))
  assert reads.mean() == pytest.approx(1.0, abs=3 * encoders.TORQUE_NOISE_NM / 60)
  assert reads.std() == pytest.approx(math.hypot(encoders.TORQUE_NOISE_NM, lsb / math.sqrt(12)),
                                      rel=0.1)
  assert encoders.torque_reading(1.0, "arm_elbow", 7) == encoders.torque_reading(1.0, "arm_elbow", 7)
  assert encoders.torque_reading(99.0, "arm_elbow", 1) <= encoders.TORQUE_RANGE_NM


def test_a_tag_named_to_the_wrong_verb_is_refused_with_what_it_is():
  """A first weighing on legs (ladder B) asked `find` for cube 23 and was
  told only which tags are places. The refusal says what the tag is."""
  from pluggybot.lifecycle import world_facts
  facts = world_facts(QUAD_HOME)
  [why] = st.check_step(st.VERBS["find"], {"tag": 23, "x": 25.2, "y": -2.4}, facts)
  assert why.startswith("tag 23 is no place") and "a cube's, and `pick` finds" in why
  [why] = st.check_step(st.VERBS["pick"], {"tag": 44}, facts)
  assert why.startswith("tag 44 is no cube") and "marks a place, which `find`" in why
  [why] = st.check_step(st.VERBS["pick"], {"tag": 99}, facts)
  assert why.endswith(")"), "a tag that is nothing gets no gloss"
