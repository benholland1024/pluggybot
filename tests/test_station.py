"""The humanoid's charging station (issue #559, stage 2; `activity/station.py`,
`humanoid/charge.py`): its two catches' rules, each pinned on the house with
one humanoid standing at the station, the plug set where a rule asks; and
one flight, plugging in from the seat. The success rate walking in, the
charge's watts and a fall while plugged in are `scripts/station_spike.py`'s.
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.activity import station as st
from pluggybot.humanoid import charge
from pluggybot.perception.tags import STATION_TAG_IDS


def _house(at_seat=True, pair=False):
  """The house, a humanoid at the station's seat, its activities hooked:
  (model, data, body, station)."""
  from pluggybot.body import body_for
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  from pluggybot.robot import world_spec
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"], cfg["start2"][:2] if pair else None).compile()
  data = mujoco.MjData(model)
  body = body_for(model, data)
  s = body._station
  x, y = s.to_world(st.SEAT_X if at_seat else st.STANDOFF_X, st.SEAT_Y)
  body.start_at(x, y, s.yaw)
  acts = cfg["activities"](model, data)
  station = next(a for a in acts if a.name == "station")
  body.step_hooks.append(acts.step_hook(model, data))
  body.run(body.hold_routine(0.5))
  return model, data, body, station


def _set_plug(model, data, station, pos, axis):
  """The plug free of its cup, its face at `pos`, its nose along `axis`."""
  data.eq_active[station.cup] = 0
  station._armed[station.cup] = False
  x = np.asarray(axis, float) / np.linalg.norm(axis)
  z = np.array([0.0, 0.0, 1.0])
  y = np.cross(z, x)
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, np.column_stack([x, y, np.cross(x, y)]).flatten())
  j = model.joint("station_plug")
  data.qpos[j.qposadr[0]:j.qposadr[0] + 3] = pos
  data.qpos[j.qposadr[0] + 3:j.qposadr[0] + 7] = q
  data.qvel[j.dofadr[0]:j.dofadr[0] + 6] = 0.0
  mujoco.mj_forward(model, data)


def _at_port(model, data, station, gap=0.0, across=0.0):
  port = model.site("charge_port").id
  out = data.site_xmat[port].reshape(3, 3)[:, 2]
  side = np.cross((0.0, 0.0, 1.0), out)
  _set_plug(model, data, station, data.site_xpos[port] + out * gap + side * across, -out)
  return port, out


def _hold(body, seconds):
  body.run(body.hold_routine(seconds))


@pytest.fixture
def house():
  model, data, body, station = _house()
  yield model, data, body, station
  body.close()


def test_the_plug_rests_in_its_cup(house):
  """At rest the cup holds the plug's back on its seat, nose out to the
  room, still: what a robot's hand finds by the station's tags."""
  model, data, body, station = house
  off = []
  body.step_hooks.append(lambda: off.append(float(np.linalg.norm(
    data.site_xpos[station.back] - data.site_xpos[station.seat]))))
  _hold(body, 1.0)
  assert station.flags["parked"] and not station.flags["mated"]
  assert max(off) < 0.002


def test_the_port_takes_a_free_plug_anywhere_in_its_funnel_and_seats_it(house):
  """The funnel round the port: a plug let go 10 mm off the connector and
  18 mm across it is drawn home and seated -- the hand knows where it holds
  one to 1-2 cm -- and charges; one 30 mm across is not."""
  model, data, body, station = house
  port, _ = _at_port(model, data, station, gap=0.010, across=0.018)
  _hold(body, 0.3)
  assert station.flags["mated"] == "pluggybot" and body.charging()
  gap, across, tilt = station._port_offset(data, port)
  assert abs(gap) < 0.002 and across < 0.002 and tilt < 1.0
  model, data, body, station = _house()
  try:
    _at_port(model, data, station, gap=0.010, across=0.030)
    _hold(body, 0.3)
    assert not station.flags["mated"] and not body.charging()
  finally:
    body.close()


def test_the_port_never_takes_a_plug_a_hand_holds(house, monkeypatch):
  """A magnet and a funnel do not draw a plug in against the hand holding
  it: held so, the catch fought the hand past 20 N and let go, again and
  again. Let go, the same plug is taken."""
  model, data, body, station = house
  monkeypatch.setattr(station, "_held", lambda data: True)
  _at_port(model, data, station, gap=0.008, across=0.010)
  _hold(body, 0.05)
  assert not station.flags["mated"]
  monkeypatch.setattr(station, "_held", lambda data: False)
  _at_port(model, data, station, gap=0.008, across=0.010)
  _hold(body, 0.2)
  assert station.flags["mated"] == "pluggybot"


def test_a_seated_plug_lets_go_past_the_connectors_20_n_and_not_before(house):
  """The RoPD's hold: a pull off its face under 20 N leaves it seated, past
  it the plug comes away -- what a fall or a walk-off does, and what the
  robot's own unplug must pull past."""
  model, data, body, station = house
  port, out = _at_port(model, data, station)
  _hold(body, 0.3)
  assert station.flags["mated"]
  for f, mated in ((12.0, True), (24.0, False)):
    for _ in range(int(0.1 / model.opt.timestep)):
      data.xfrc_applied[station.plug, :3] = out * f
      _hold(body, model.opt.timestep)
    assert bool(station.flags["mated"]) is mated, f
  data.xfrc_applied[station.plug] = 0.0
  assert station.breaks == 1


def test_the_cup_frees_the_plug_only_to_a_pull_straight_out(house, monkeypatch):
  """Its walls take a load across it: the jaws closing round the parked plug
  let it go when any 4 N did, and it slid 15 mm in them. A pull out of it
  past `PARK_N` -- a hand's, holding it -- frees it."""
  model, data, body, station = house
  monkeypatch.setattr(station, "_held", lambda data: True)
  out = -data.site_xmat[station.seat].reshape(3, 3)[:, 2] * -1      # out of the cup
  for force, parked in ((np.array([0.0, 0.0, 8.0]), True), (out * 8.0, False)):
    for _ in range(int(0.1 / model.opt.timestep)):
      data.xfrc_applied[station.plug, :3] = force
      _hold(body, model.opt.timestep)
    assert station.flags["parked"] is parked, force
  data.xfrc_applied[station.plug] = 0.0


def test_a_catch_takes_the_plug_again_only_once_it_has_left_the_funnel(house):
  """Re-armed inside its own reach, a pull that broke the catch was caught
  again the next step -- 30 times in one unplug -- and the plug never came
  free. After a breakaway the plug back in the funnel is not taken until it
  has been `CLEAR_M` away."""
  model, data, body, station = house
  e = next(iter(station.ports))
  _at_port(model, data, station)
  _hold(body, 0.3)
  station._let_go(data, e)                      # a breakaway
  _at_port(model, data, station, gap=0.014)     # inside the funnel's 15 mm
  _hold(body, 0.02)
  assert not station.flags["mated"], "taken again inside its reach"
  _at_port(model, data, station, gap=st.CLEAR_M + 0.01)
  _hold(body, 0.05)
  _at_port(model, data, station, gap=0.008)
  _hold(body, 0.2)
  assert station.flags["mated"]


def test_the_cup_takes_back_only_a_plug_no_hand_holds(house, monkeypatch):
  """Caught under the hand carrying it home, the cup felt the hand's pull,
  let go and never took the plug back: 3 unplugs in 10 left it unparked.
  Held at its mouth it is not taken; let go there, it is."""
  model, data, body, station = house
  seat = data.site_xpos[station.seat].copy()
  out = data.site_xmat[station.seat].reshape(3, 3)[:, 2]          # out to the room
  monkeypatch.setattr(station, "_held", lambda data: True)
  _set_plug(model, data, station, seat + out * (st.PLUG_LEN_M + 0.01), out)
  _hold(body, 0.05)
  assert not station.flags["parked"]
  monkeypatch.setattr(station, "_held", lambda data: False)
  _hold(body, 0.5)
  assert station.flags["parked"]


def test_a_free_plug_swings_home_its_spin_bounded():
  """Let out 0.4 m with a spin, the plug hangs on its rope and the balancer
  brings it home to the cup -- ⚠ its spin damped apart from its swing: at
  the swing's 0.5 the spin grew four times a step until MuJoCo reset the
  world. (The robot at its standoff, clear of the plug.)"""
  model, data, body, station = _house(at_seat=False)
  seat = data.site_xpos[station.seat].copy()
  into = data.site_xmat[station.seat].reshape(3, 3)[:, 2]          # out to the room
  _set_plug(model, data, station, seat + into * 0.4 + np.array([0.0, 0.0, -0.2]), into)
  j = model.joint("station_plug")
  data.qvel[j.dofadr[0] + 3:j.dofadr[0] + 6] = (2.0, 1.0, 3.0)
  t0, spins = float(data.time), []
  body.step_hooks.append(lambda: spins.append(float(np.abs(data.qvel[j.dofadr[0] + 3:
                                                                    j.dofadr[0] + 6]).max())))
  _hold(body, 8.0)
  assert float(data.time) > t0 + 7.9, "MuJoCo reset the world"
  assert max(spins) < 50.0
  assert station.flags["parked"]


def test_the_station_is_fitted_off_its_two_tags():
  """Its facing off the pair's baseline (#88), its origin between them;
  none off one tag."""
  pose = (3.5, -1.88, -math.pi / 2)
  spots = {i: (*st.to_world(pose, x, y), z) for i, (x, y, z) in st.tag_layout().items()}
  f = charge.fit(spots)
  assert (f.x, f.y) == pytest.approx(pose[:2]) and f.yaw == pytest.approx(pose[2])
  assert charge.fit({STATION_TAG_IDS[0]: spots[STATION_TAG_IDS[0]]}) is None


def test_charging_is_its_own_ports_catch():
  """A pair's two ports are each their robot's: the other's plugged in is
  not this one charging."""
  model, data, body, station = _house(pair=True)
  try:
    mine, theirs = (model.equality(f"station_port_{r}").id for r in ("pluggybot", "r2_pluggybot"))
    data.eq_active[theirs] = 1
    assert not body.charging()
    data.eq_active[mine] = 1
    assert body.charging()
  finally:
    body.close()


def test_a_humanoid_plugs_itself_in_from_its_seat():
  """The plug-in flown (`humanoid.charge`): off its own senses -- the
  station off its tags, its port off its model of itself -- it docks and
  charges, and the hand is home and empty after."""
  model, data, body, station = _house()
  try:
    assert body.run(body.dock_routine()) == "docked"
    assert body.charging() and station.flags["mated"] == "pluggybot"
    assert all(p == "rest" for p in body.reach.phase.values()) and not body.gripped("right")
    assert body.falls == 0
  finally:
    body.close()


# ---- the plug-in's grips and the unplug's squeeze, on a body that only says -------


class _Says:
  """A body whose hand does what each test says, and records what it was
  asked: the plug-in's and the unplug's rules, with no physics."""

  def __init__(self, reaches=lambda rot: True):
    self.reaches, self.asked, self.grips, self.moves = reaches, [], [], []
    self.plugged = False
    self.gap = 2.0 * st.PLUG_R_M          # ...shut on the plug's jack

  def _done(self, value):
    return value
    yield

  def reach_routine(self, side, pos, rot=None, **kw):
    self.asked.append(np.asarray(rot))
    return self._done(self.reaches(np.asarray(rot)))

  def move_hand_routine(self, side, pos, turns=None, **kw):
    self.moves.append((np.asarray(pos), turns))
    return self._done({"why": "arrived"})

  def grip_routine(self, side, closed, force=None):
    self.grips.append((closed, force))
    return self._done(True)

  def pull_routine(self, side, way, force, distance, **kw):
    self.plugged = force == charge.PUSH_N
    return self._done({"why": "held fast", "along": 0.02})

  def hold_routine(self, seconds):
    return self._done(None)

  def release_routine(self, side):
    return self._done("home")

  def hand_why(self, side):
    return "stopped short"

  def grip_gap(self, side):
    return self.gap

  def charging(self):
    return self.plugged


def _drive(routine):
  try:
    while True:
      next(routine)
  except StopIteration as done:
    return done.value


def test_the_hand_takes_the_plug_by_its_grips_in_turn(monkeypatch):
  """`charge.GRIPS`: the fingers swung toward the plug's back first, where
  the arm reaches from furthest behind the seat; straight across where the
  swung hand's way in is barred -- and the port read only once the hand
  holds the plug, carried to in the grip that took it, and every squeeze
  on a plug a catch holds begun at the least the gripper does."""
  from pluggybot.body import RackPose
  monkeypatch.setattr(charge, "port", lambda body: (np.array([0.0, 0.0, 0.9]),
                                                   np.array([-1.0, 0.0, 0.0])))
  station = RackPose(1.0, 0.0, 0.0)
  into = np.array([1.0, 0.0, 0.0])
  (swung, far), (straight, near) = charge.GRIPS
  assert swung > 0.0 and far > near == st.GRIP_BACK_M and straight == 0.0
  body = _Says(reaches=lambda rot: np.allclose(rot, charge._grip(into, straight)))
  trace = []
  assert _drive(charge.plug_in_routine(body, station, trace)) == "plugged in"
  assert [np.allclose(r, charge._grip(into, s)) for r, s in zip(body.asked, (swung, straight))] \
      == [True, True]
  assert np.allclose(body.moves[0][0], charge.parked_grip(station, near))
  assert np.allclose(body.moves[1][1][0], charge._grip(np.array([-1.0, 0.0, 0.0]), straight))
  assert trace[-1]["swing"] == 0
  shut = [f for closed, f in body.grips if closed]
  assert shut == [charge.GRIP_MIN_N, charge.GRIP_N], shut
  # ...and the fingers swing about the jaws' line, which stays upright
  for s in (0.0, swung):
    g = charge._grip(into, s)
    assert np.allclose(g[:, 0], (0.0, 0.0, 1.0))
    assert -g[2, 2] == pytest.approx(0.0, abs=1e-12)
    assert math.atan2(-g[0, 2], -g[1, 2]) == pytest.approx(s)


def test_jaws_shut_on_the_boss_draw_back_along_the_plug_and_shut_again(monkeypatch):
  """Held is not the plug: the jaws' gap must be the jack's width
  (`PLUG_GAP_TOL_M`). Wider, they shut on the station's boss -- the fit is
  1-2 cm off along the plug, and a hand that pushed then pushed nothing --
  so the hand opens, draws `BOSS_BACK_M` back along the plug and shuts
  again; on the boss twice, the next grip fetches it. No push is made
  without the plug."""
  from pluggybot.body import RackPose
  monkeypatch.setattr(charge, "port", lambda body: (np.array([0.0, 0.0, 0.9]),
                                                   np.array([-1.0, 0.0, 0.0])))
  station = RackPose(1.0, 0.0, 0.0)
  into = np.array([1.0, 0.0, 0.0])
  (swung, back), (straight, _) = charge.GRIPS
  for gaps, swing, reaches in (((0.040, 2.0 * st.PLUG_R_M), swung, 1),
                               ((0.040, 0.040, 2.0 * st.PLUG_R_M), straight, 2)):
    body = _Says()
    seq = iter(gaps)
    real_grip = body.grip_routine

    def grip(side, closed, force=None, body=body, seq=seq, real_grip=real_grip):
      if closed and force == charge.GRIP_N:
        body.gap = next(seq)
      return real_grip(side, closed, force)
    body.grip_routine = grip
    trace = []
    assert _drive(charge.plug_in_routine(body, station, trace)) == "plugged in"
    assert [t["swing"] for t in trace] == [round(math.degrees(swing))]
    assert len(body.asked) == reaches
    drawn = charge.parked_grip(station, back) - into * charge.BOSS_BACK_M
    assert any(np.allclose(pos, drawn) for pos, _ in body.moves), "drawn back off the boss"


def test_the_unplug_squeezes_the_seated_plug_lightly_first(monkeypatch):
  """Shut at once on it, the first pad to meet the plug pushed it across the
  port past the connector's 20 N, and it broke away (`charge._squeeze_routine`)."""
  from pluggybot.body import RackPose
  monkeypatch.setattr(charge, "port", lambda body: (np.array([0.0, 0.0, 0.9]),
                                                   np.array([-1.0, 0.0, 0.0])))
  body = _Says()
  body.plugged = True
  body.pull_routine = lambda *a, **kw: (setattr(body, "plugged", False),
                                        body._done({"why": "distance", "along": 0.05}))[1]
  assert _drive(charge.unplug_routine(body, RackPose(1.0, 0.0, 0.0), [])) == "unplugged"
  shut = [f for closed, f in body.grips if closed]
  assert shut[:2] == [charge.GRIP_MIN_N, charge.UNPLUG_GRIP_N], shut


# ---- lost, and found by the station -------------------------------------------------


def test_a_look_at_the_station_fixes_the_belief_and_past_the_matchers_reach_drops_the_map():
  """The station is commissioned as the dock is (#476): its look puts the
  belief where it says the robot stands, and a fix past the matcher's own
  search (`humanoid.body.ASKEW_M`) drops the map laid on the old belief."""
  from pluggybot.body import RackPose
  model, data, body, station = _house(at_seat=False)
  try:
    s, m = body._station, body.mission
    x, y, yaw = body.pose
    near = RackPose(s.x + 0.3, s.y - 0.2, s.yaw)       # the belief 0.36 m off
    fix = body._fix_at(near)
    assert body.pose[:2] == pytest.approx((x - 0.3, y + 0.2), abs=1e-6)
    assert fix["fixture"] == "station" and fix["moved"] and not fix["dropped"]
    assert body._fix_at(s) is None, "a look that agrees moves nothing"
    m.grid.grid[10, 10] = 5.0
    far = RackPose(s.x + 0.7, s.y, s.yaw)
    assert body._fix_at(far)["dropped"] and not m.grid.grid.any()
  finally:
    body.close()


def test_a_lost_humanoid_steps_in_by_its_lidar_and_along_the_wall_until_it_sees_it(monkeypatch):
  """`lost_routine`: facing where it believes the station is, it steps in
  until its LIDAR reads the wall `LOST_WALL_M` ahead -- never by its belief,
  which is what is wrong -- and looks ahead and either side; then from a
  step along the wall. The look that finds it fixes the belief."""
  from pluggybot.body import RackPose
  model, data, body, station = _house(at_seat=False)
  try:
    s = body._station
    steps, looks = [], []

    def step(ahead, left=0.0, turn=0.0, patience=0.0):
      steps.append((round(ahead, 3), round(left, 3)))
      return {"why": "stepped"}
      yield

    def face(heading):
      return True
      yield
    monkeypatch.setattr(body, "_wall_ahead", lambda: 1.5)
    monkeypatch.setattr(body, "step_routine", step)
    monkeypatch.setattr(body, "face_routine", face)
    def look(b):
      looks.append(1)
      return RackPose(s.x + 0.7, s.y, s.yaw) if len(looks) == len(charge.LOST_LOOKS) + 1 else None
      yield
    monkeypatch.setattr(charge, "look_routine", look)
    assert body.run(body.lost_routine("dock"))
    assert steps == [(round(1.5 - charge.LOST_WALL_M, 3), 0.0), (0.0, charge.LOST_ALONG_M[0])]
    events = [e["why"] for e in body.belief_events]
    assert events == ["lost", "fixed", "searched"], events
    assert body.belief_events[-1]["found"] and body.belief_events[-1]["viewpoints"] == 1
  finally:
    body.close()


def test_a_lost_humanoid_says_it_looks_for_its_station_and_whether_it_found_it():
  """History, in its own world's words: its station, never the dock or the
  rack it does not have (`HubLifecycle._tell_belief`)."""
  from pluggybot.body import StubBody
  from pluggybot.mind.thoughts import HISTORY
  from pluggybot.navigator import wire_pose
  from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
  body = StubBody()
  life = stub_life(body=body)
  life._drift_due = math.inf
  at = wire_pose((1.0, 2.0, 0.0))
  for seq, why, kw in ((1, "lost", {}), (2, "searched", {"found": False, "seconds": 41.0})):
    body.belief_events.append({"seq": seq, "t": 5.0, "why": why, "before": at, "after": at,
                               "truth": at, "fixture": "station", **kw})
  life._belief_step()
  said = [ln for ln in life.thoughts.texts[HISTORY].splitlines() if ln.strip()]
  assert any("my charging station was not in sight" in ln and "from nearer" in ln
             for ln in said), said
  assert any(ln.endswith("I looked for my charging station for 41 s and did not see it")
             for ln in said), said
  assert not any("dock" in ln or "rack" in ln for ln in said), said
