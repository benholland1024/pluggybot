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


def test_a_plug_in_its_port_rides_a_restart():
  """A restart's fresh model has the port's weld as compiled, its anchor
  1 m off where a catch puts it (`_take`): read so, the weld pulled 3,100 N,
  let the plug go on the first step and threw it across the room (#654).
  Everything a restart brings back -- the plug, the catch's seat and its
  anchor -- and the plug stays in."""
  from pluggybot import continuation
  model, data, body, station = _house()
  try:
    _at_port(model, data, station, gap=0.005)
    _hold(body, 0.3)
    assert station.flags["mated"] == "pluggybot" and body.charging()
    kept, physics = station.kept_state(), continuation.physics(model, data)
  finally:
    body.close()
  model, data, body, station = _house()
  try:
    eq = next(e for e in station.ports)
    assert model.eq_data[eq, 1] == 1.0, "the premise: a fresh model's anchor is 1 m off"
    continuation.put_physics(model, data, *physics, forward=False)
    station.restore_kept(kept)
    mujoco.mj_forward(model, data)
    assert list(model.eq_data[eq, 0:3]) == [0.0, 0.0, 0.0]
    _hold(body, 0.05)
    assert body.charging() and station.breaks == 0, "the first step after a restart threw it"
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
  asked: the plug-in's and the unplug's rules, with no physics. Its belief
  (`stance`, where it believes its feet are) stays put unless a test's
  `drift` moves it on a routine's call, and it sees the station where a
  test's `sees` puts it in the frame of its feet (None: nothing in sight);
  every hand's target is kept with the belief it was sent under."""

  #: The station it charges at: the house's pair (`HumanoidBody.station_tags`).
  station_tags = STATION_TAG_IDS

  def __init__(self, reaches=lambda rot: True, sees=None, drift=None):
    self.reaches, self.asked, self.grips, self.moves = reaches, [], [], []
    self.plugged = False
    self.gap = 2.0 * st.PLUG_R_M          # ...shut on the plug's jack
    self.stance = (0.0, 0.0, 0.0)
    self.sees, self.drift = sees, drift
    self.reached, self.pushed, self.said, self.stepped = [], [], [], []

  #: ...where it believes it stands: its stance, for a body that only says
  pose = property(lambda self: self.stance)

  def _done(self, value):
    return value
    yield

  def _call(self, what):
    self.said.append(what)
    if self.drift is not None:
      self.stance = self.drift(self.stance)

  def to_world(self, p):
    x, y, yaw = self.stance
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2]])

  def spot(self, tag):
    self.said.append("spot")
    at = self.sees() if callable(self.sees) else self.sees
    if at is None:
      return None
    lx, ly, lz = st.tag_layout()[tag]
    x, y, z = self.to_world((*st.to_world((at.x, at.y, at.yaw), lx, ly), lz))
    return {"x": x, "y": y, "z": z}

  def gaze_routine(self, down):
    self._call("gaze")
    return self._done(None)

  def reach_routine(self, side, pos, rot=None, **kw):
    self.asked.append(np.asarray(rot))
    self.reached.append((np.asarray(pos), self.stance))
    self._call("reach")
    return self._done(self.reaches(np.asarray(rot)))

  def move_hand_routine(self, side, pos, turns=None, **kw):
    self.moves.append((np.asarray(pos), turns, self.stance))
    self._call("move")
    return self._done({"why": "arrived"})

  def grip_routine(self, side, closed, force=None):
    self.grips.append((closed, force))
    self._call("grip")
    return self._done(True)

  def pull_routine(self, side, way, force, distance, **kw):
    self.plugged = force == charge.PUSH_N
    self.pushed.append((np.asarray(way), self.stance))
    self._call("push")
    return self._done({"why": "held fast", "along": 0.02})

  def hold_routine(self, seconds):
    self._call(("hold", seconds))
    return self._done(None)

  def step_routine(self, ahead, left=0.0, turn=0.0):
    """A step as asked, its stance moved there."""
    self.stepped.append((ahead, left, turn))
    x, y, yaw = self.stance
    c, s = math.cos(yaw), math.sin(yaw)
    self.stance = (x + c * ahead - s * left, y + s * ahead + c * left, yaw + turn)
    self._call("step")
    return self._done({"why": "stepped"})

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
  carry = next(turns for _, turns, _ in body.moves if turns is not None)
  assert np.allclose(carry[0], charge._grip(np.array([-1.0, 0.0, 0.0]), straight))
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
    assert any(np.allclose(pos, drawn) for pos, _, _ in body.moves), "drawn back off the boss"


def _unplugs(body):
  """A body plugged in, whose pull takes the plug off its port."""
  body.plugged = True
  body.pull_routine = lambda *a, **kw: (setattr(body, "plugged", False),
                                        body._done({"why": "distance", "along": 0.05}))[1]
  return body


def test_the_unplug_squeezes_the_seated_plug_lightly_first(monkeypatch):
  """Shut at once on it, the first pad to meet the plug pushed it across the
  port past the connector's 20 N, and it broke away (`charge._squeeze_routine`)."""
  from pluggybot.body import RackPose
  monkeypatch.setattr(charge, "port", lambda body: (np.array([0.0, 0.0, 0.9]),
                                                   np.array([-1.0, 0.0, 0.0])))
  body = _unplugs(_Says())
  assert _drive(charge.unplug_routine(body, RackPose(1.0, 0.0, 0.0), [])) == "unplugged"
  shut = [f for closed, f in body.grips if closed]
  assert shut[:2] == [charge.GRIP_MIN_N, charge.UNPLUG_GRIP_N], shut


def test_the_unplug_steps_back_to_its_seat_first_where_a_charge_crept_it_in(monkeypatch):
  """#703: plugged in, the balancer's 2 N on its port creeps a standing robot
  in toward the station, 7.5 mm a minute, until its plug meets the station 18
  cm past its seat (SimNotes, "The unplug after a long charge"), and from
  there the arm's way out behind the plug drove its jaw into the wall beside
  the station and the station's box: after
  the live 1,944 and 1,504 s charges, "my hand could not get the plug out",
  twice of twice. Further in than `UNPLUG_IN_M` it backs off to its seat
  before its hand goes out, by the station it believes in -- from there no
  look sees its tags -- never turned on the spot first (pinned there, a turn
  swings its plug across the box's face, which the port lets go to); no
  nearer in it steps not at all."""
  from pluggybot.body import RackPose
  monkeypatch.setattr(charge, "port", _port)
  station = RackPose(0.35, -0.075, 0.0)          # its seat where its stance is
  for crept, turned, steps in ((0.0, 0.0, 0), (charge.UNPLUG_IN_M - 0.005, 0.0, 0),
                               (0.18, 0.0, 1), (0.18, math.radians(11.0), 1)):
    body = _unplugs(_Says())
    body.stance = (crept, 0.0, turned)
    trace = []
    assert _drive(charge.unplug_routine(body, station, trace)) == "unplugged"
    assert len(body.stepped) == steps, (crept, turned, body.stepped)
    if steps:
      assert body.said.index("step") < body.said.index("reach"), body.said
      assert trace[0]["seat"][0] == pytest.approx(-0.18)
      # ...back along the station's way to its seat, its feet as they stood
      assert body.stepped[0][2] == 0.0
      assert body.stance[:2] == pytest.approx((0.0, 0.0), abs=1e-9)
      assert body.stance[2] == pytest.approx(turned)
      # ...and its look after the step taken standing still, never bowed
      after = body.said[body.said.index("step") + 1:]
      assert after[:2] == [("hold", charge.LOOK_SETTLE_S), "spot"], after
      assert "gaze" not in body.said, body.said


def test_an_unplug_whose_step_back_freed_the_plug_reaches_for_none(monkeypatch):
  """A step back that let the plug go -- the port lets go to a pull across
  its face -- leaves nothing to take out: no hand goes out for it."""
  from pluggybot.body import RackPose
  monkeypatch.setattr(charge, "port", _port)
  body = _unplugs(_Says())
  stepped = body.step_routine

  def step(*a, **kw):
    body.plugged = False
    return stepped(*a, **kw)
  body.step_routine = step
  body.stance = (0.18, 0.0, 0.0)
  why = _drive(charge.unplug_routine(body, RackPose(0.35, -0.075, 0.0), []))
  assert "reach" not in body.said and not body.plugged, body.said
  assert why == "it came off its port as it stepped back to its seat (stepped)", why


def test_a_back_off_the_other_robot_keeps_it_from_asks_it_no_more(monkeypatch):
  """A step kept off the other robot standing behind (`blocked`, as a walk
  keeps off it) goes the same way asked again: it is the back-off's last, and
  the try goes on from where it stands."""
  from pluggybot.body import RackPose
  monkeypatch.setattr(charge, "port", _port)
  body = _unplugs(_Says())
  body.step_routine = lambda *a, **kw: (body.stepped.append(a), body._done({"why": "blocked"}))[1]
  body.stance = (0.18, 0.0, 0.0)
  trace = []
  assert _drive(charge.unplug_routine(body, RackPose(0.35, -0.075, 0.0), trace)) == "unplugged"
  assert len(body.stepped) == 1 and {"step": "blocked"} in trace, trace


def test_an_unplug_that_failed_tries_once_more_off_a_look_of_its_own(monkeypatch):
  """#703: a hand that could not take the plug out left it to the walk, and
  after the undock's turn its rope pulled the plug straight into the port's
  face, which the catch holds: the walk stalled. One more try, after a look of
  its own taken standing still (`LOOK_SETTLE_S`: just after a move, a look
  read the trunk's tilt 0.5 deg off, #596); two that fail say why, and try
  no third."""
  from pluggybot.body import RackPose
  monkeypatch.setattr(charge, "port", _port)
  station = RackPose(0.35, -0.075, 0.0)
  went = iter((False, True))
  body = _unplugs(_Says(reaches=lambda rot: next(went)))
  assert _drive(charge.unplug_routine(body, station, [])) == "unplugged"
  first, again = (i for i, w in enumerate(body.said) if w == "reach")
  between = body.said[first:again]
  assert ("hold", charge.LOOK_SETTLE_S) in between, body.said
  assert between.index(("hold", charge.LOOK_SETTLE_S)) < between.index("spot"), between
  body = _unplugs(_Says(reaches=lambda rot: False))
  trace = []
  why = _drive(charge.unplug_routine(body, station, trace))
  assert why.startswith("its hand did not get behind the plug"), why
  assert body.said.count("reach") == charge.UNPLUG_TRIES == 2 and body.plugged
  assert [t["unplugTry"] for t in trace if "unplugTry" in t] == [0, 1]


def _station_at(seat, yaw):
  """The station (`RackPose`) whose seat is at `seat` (x, y) and that
  faces `yaw`."""
  from pluggybot.body import RackPose
  c, s = math.cos(yaw), math.sin(yaw)
  return RackPose(seat[0] - (c * st.SEAT_X - s * st.SEAT_Y),
                  seat[1] - (s * st.SEAT_X + c * st.SEAT_Y), yaw)


def _crept_body(crept, turned=math.radians(11.0), sees=True):
  """A body plugged in that truly stands `crept` m past its seat, turned
  `turned` to the station's left, as a long charge creeps it (SimNotes, "The
  unplug after a long charge"), while its belief puts it at its seat -- as
  live, 0.13-0.21 m off, the creep hidden. Its look sees the station only
  from 2 cm in front of its seat or further back (X1's probe at the lobby's
  station), where the station truly is. (body, the station it believes in,
  the station as it truly stands, in the frame it stood in.)"""
  believed = _station_at((0.0, 0.0), 0.0)
  true = _station_at((-crept * math.cos(turned), crept * math.sin(turned)), -turned)

  def look():
    from pluggybot.body import RackPose
    x, y, yaw = body.stance
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = true.x - x, true.y - y
    at = RackPose(c * dx + s * dy, -s * dx + c * dy, true.yaw - yaw)
    sx, sy = at.to_world(st.SEAT_X, st.SEAT_Y)
    ahead = sx * math.cos(at.yaw) + sy * math.sin(at.yaw)
    return at if ahead >= -0.02 else None
  body = _unplugs(_Says(sees=look if sees else None))
  return body, believed, true


def test_after_a_charge_long_enough_to_creep_the_unplug_steps_back_and_looks_first(monkeypatch):
  """#703's option 2 (Ben, 2026-11-23): on the weekend of 11-21/22 the hand
  could not take the plug out 3 times of 8, each after a charge carried over
  a run's end. Plugged in, the balancer creeps the body 16-18 cm past its
  seat, and its belief, 0.13-0.21 m off, put it at its seat, so it never
  backed off (`to_seat`): flown so, 4 of 6 failed. After a charge its own
  clock says was long enough to creep (`charge.crept`), its first try steps
  straight back first, whatever its belief says -- never turned, its feet as
  they stood -- then looks standing still, never bowed, and its back-off and
  its hand go by the station that look places."""
  monkeypatch.setattr(charge, "port", _port)
  body, believed, true = _crept_body(0.18)
  back = charge.crept(3000.0)
  trace = []
  assert _drive(charge.unplug_routine(body, believed, trace, back=back)) == "unplugged"
  assert body.stepped[0] == (-back, 0.0, 0.0), body.stepped
  assert body.said.index("step") < body.said.index("reach"), body.said
  after = body.said[body.said.index("step") + 1:]
  assert after[:2] == [("hold", charge.LOOK_SETTLE_S), "spot"], after
  assert "gaze" not in body.said, body.said
  assert trace[:3] == [{"crept": round(back, 3)}, {"step": "stepped"}, {"looked": True}], trace
  # ...behind its seat now by the look, no step more, and the plug let go at
  # the cup where it truly is
  assert len(body.stepped) == 1, body.stepped
  into = np.array([math.cos(true.yaw), math.sin(true.yaw), 0.0])
  cup = charge.parked_grip(true) - into * charge.CUP_SHORT_M
  assert any(np.allclose(pos, cup) for pos, _, _ in body.moves), "let go at the cup it saw"
  # ...the premise: by its belief alone, as before, it never stepped back
  body, believed, _ = _crept_body(0.18)
  assert _drive(charge.unplug_routine(body, believed, [])) == "unplugged"
  assert body.stepped == [], "the premise: its belief at its seat, it stood where it crept"


def test_a_step_back_whose_look_sees_nothing_goes_on_by_the_station_it_had(monkeypatch):
  """Its look after the step back sees no pair (from 4 cm in front of its
  seat or nearer the station, none does): it goes on by the station it had,
  as before, from where its step left it."""
  monkeypatch.setattr(charge, "port", _port)
  body, believed, _ = _crept_body(0.18, sees=False)
  trace = []
  assert _drive(charge.unplug_routine(body, believed, trace,
                                      back=charge.crept(3000.0))) == "unplugged"
  assert {"looked": False} in trace and len(body.stepped) == 1, trace
  into = np.array([math.cos(believed.yaw), math.sin(believed.yaw), 0.0])
  cup = charge.parked_grip(believed) - into * charge.CUP_SHORT_M
  assert any(np.allclose(pos, cup) for pos, _, _ in body.moves)
  # ...its back-off by that station as before: one it still believes itself
  # further in than `UNPLUG_IN_M` from steps on to its seat
  body, _, _ = _crept_body(0.18, sees=False)
  deep = _station_at((-0.30, 0.0), 0.0)        # believed 30 cm past its seat
  trace = []
  assert _drive(charge.unplug_routine(body, deep, trace,
                                      back=charge.crept(3000.0))) == "unplugged"
  assert len(body.stepped) == 2 and body.stepped[1][2] == 0.0, body.stepped
  assert body.stance[0] == pytest.approx(-0.30, abs=1e-9), "on to its seat as it believes it"


def test_a_later_try_steps_back_by_no_clock_again(monkeypatch):
  """The step back by its clock is the first try's alone: a try that failed
  is tried again from where the first left it, after a look of its own and
  its back-off by that look, as before."""
  monkeypatch.setattr(charge, "port", _port)
  went = iter((False, True))
  body, believed, _ = _crept_body(0.18)
  body.reaches = lambda rot: next(went)
  trace = []
  assert _drive(charge.unplug_routine(body, believed, trace,
                                      back=charge.crept(3000.0))) == "unplugged"
  assert len(body.stepped) == 1 and sum("crept" in t for t in trace) == 1, trace
  first, again = (i for i, w in enumerate(body.said) if w == "reach")
  between = body.said[first:again]
  assert between.index(("hold", charge.LOOK_SETTLE_S)) < between.index("spot"), between


def test_a_step_back_the_other_robot_keeps_it_from_looks_and_steps_no_more(monkeypatch):
  """Kept off the other robot standing behind (`blocked`), the same ask goes
  the same way: no look, no back-off's step after it, and the try goes on
  from where it stands -- as the back-off's own step does."""
  monkeypatch.setattr(charge, "port", _port)
  body, believed, _ = _crept_body(0.18)
  body.step_routine = lambda *a, **kw: (body.stepped.append(a), body._done({"why": "blocked"}))[1]
  trace = []
  assert _drive(charge.unplug_routine(body, believed, trace,
                                      back=charge.crept(3000.0))) == "unplugged"
  assert len(body.stepped) == 1 and {"step": "blocked"} in trace, trace
  assert not any("looked" in t or "seat" in t for t in trace), trace


def test_a_step_back_that_freed_the_plug_reaches_for_none(monkeypatch):
  """A step back that let the plug go leaves nothing to take out."""
  monkeypatch.setattr(charge, "port", _port)
  body, believed, _ = _crept_body(0.18)
  stepped = body.step_routine

  def step(*a, **kw):
    body.plugged = False
    return stepped(*a, **kw)
  body.step_routine = step
  why = _drive(charge.unplug_routine(body, believed, [], back=charge.crept(3000.0)))
  assert "reach" not in body.said and not body.plugged, body.said
  assert why == "it came off its port as it stepped back to its seat (stepped)", why


def test_the_step_back_is_the_creep_its_clock_gives_and_lands_it_behind_its_seat():
  """`charge.crept`: a charge too short to creep it past where its hand gets
  behind the plug steps nothing first (its unplug as it was); a longer one
  the creep its clock gives -- 7.5 mm a minute, stopping 18.3 cm past the
  seat -- and `CREPT_PAST_M` more, past the `SEAT_IN_M` it may have docked
  in (its look sees the pair turned only from its seat or further back),
  never under a step its walker takes (`STEP_SHORT_M`); one whose length it
  does not know (a save from before the clock), as far as a creep goes."""
  from pluggybot.humanoid import body as hb
  assert charge.crept(0.0) is None and charge.crept(charge.CREPT_AFTER_S) is None
  just = charge.crept(charge.CREPT_AFTER_S + 1e-6)
  assert just == pytest.approx(charge.CREEP_M_S * charge.CREPT_AFTER_S + charge.CREPT_PAST_M)
  assert just >= hb.STEP_SHORT_M
  assert charge.crept(1200.0) == pytest.approx(charge.CREEP_M_S * 1200.0 + charge.CREPT_PAST_M)
  most = charge.CREEP_MOST_M + charge.CREPT_PAST_M
  assert charge.crept(1e5) == pytest.approx(most) and charge.crept(None) == pytest.approx(most)
  assert charge.CREPT_PAST_M > charge.SEAT_IN_M


def test_the_undock_steps_back_by_its_own_clock_kept_from_its_dock(monkeypatch):
  """The clock is its own (#703's option 2), never its belief: set as its
  dock puts the plug in, read by its undock (`HumanoidBody.plugged_s`),
  forgotten once the plug is out; where it does not know (a save from
  before the clock), as far as a creep goes."""
  import types
  from pluggybot.body import RackPose
  from pluggybot.humanoid.body import HumanoidBody
  asked, out = [], [False]

  def unplug(body, station, trace, back=None):
    asked.append(back)
    body.plugged = not out[0]
    return "unplugged" if out[0] else "its hand did not get behind the plug (stopped short)"
    yield
  monkeypatch.setattr(charge, "unplug_routine", unplug)
  monkeypatch.setattr(charge, "look", lambda body: None)
  monkeypatch.setattr(charge, "look_routine", lambda body: _done(RackPose(0.35, -0.075, 0.0)))
  monkeypatch.setattr(charge, "plug_in_routine", lambda body, seen, trace: _done("plugged in"))

  class Body:
    plugged_s = HumanoidBody.plugged_s
    dock_routine = HumanoidBody.dock_routine
    undock_routine = HumanoidBody.undock_routine
    _station = RackPose(0.35, -0.075, 0.0)
    pose = (0.0, 0.0, 0.0)

    def __init__(self):
      self._d = types.SimpleNamespace(time=100.0)
      self.mission = types.SimpleNamespace(plugged_t=None)
      self._station_seen, self._charge_trace, self.plugged = None, [], False

    def charging(self):
      return self.plugged

    def face_routine(self, yaw):
      return _done(None)

    def step_routine(self, *a, **kw):
      return _done({"why": "stepped"})
  body = Body()
  assert _drive(body.dock_routine()) == "docked" and body.mission.plugged_t == 100.0
  body.plugged = True
  for plugged_s in (300.0, charge.CREPT_AFTER_S + 60.0, 3000.0):
    body._d.time = 100.0 + plugged_s
    _drive(body.undock_routine())
    assert asked.pop() == charge.crept(plugged_s), plugged_s
    assert body.mission.plugged_t == 100.0, "kept while its plug is in"
  assert charge.crept(300.0) is None and charge.crept(3000.0) is not None
  out[0] = True
  _drive(body.undock_routine())
  assert body.mission.plugged_t is None, "forgotten once the plug is out"
  # ...and a charge whose length it does not know -- a save from before the
  # clock, or a clock ahead of the world's -- as far as a creep goes
  body.plugged, out[0] = True, False
  _drive(body.undock_routine())
  assert asked.pop() == charge.crept(None) == pytest.approx(
    charge.CREEP_MOST_M + charge.CREPT_PAST_M)
  body.mission.plugged_t = body._d.time + 50.0
  _drive(body.undock_routine())
  assert asked.pop() == charge.crept(None)


def _done(value):
  return value
  yield


def _in_feet(v, stance, point=True):
  """A believed world point (or a direction, or a turn's columns) in the
  frame of the feet a belief `stance` put them in."""
  x, y, yaw = stance
  c, s = math.cos(yaw), math.sin(yaw)
  r = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
  v = np.asarray(v, float)
  return r @ (v - ([x, y, 0.0] if point else 0.0)) if v.ndim == 1 else r @ v


def _turn(v, rad):
  c, s = math.cos(rad), math.sin(rad)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.asarray(v, float)


#: The port as the plug-in's tests' bodies' models of themselves have it, in
#: the frame of their feet: 8 cm ahead, on the right, facing the station.
_PORT = (np.array([0.08, -0.075, 0.9]), np.array([1.0, 0.0, 0.0]))


def _port(body):
  """`charge.port` on a body that only says: its model's port, in the
  believed world at this instant."""
  return body.to_world(_PORT[0]), _turn(_PORT[1], body.stance[2])


def test_the_plug_in_carries_the_station_as_its_feet_would(monkeypatch):
  """#596: from a look to the push the feet turned 1.6-7.7 deg where they
  stood while the belief moved its stance 5-25 mm -- the matcher's
  corrections, its legs reading the turn as a walk. A hand sent through
  the believed world went with that move (a plug-in missed four times the
  same way); one held in the frame of its feet, with the turn. The station
  is carried as the feet carry it: never moved by the belief's move, turned
  as the belief turns -- the line, the grip and the hand's turn there."""
  from pluggybot.body import RackPose
  monkeypatch.setattr(charge, "port", _port)
  at = RackPose(0.35, -0.075, 0.0)          # where the look sees it, from its feet

  def drift(s):
    return s[0] + 0.006, s[1] - 0.004, s[2] + math.radians(0.4)
  body = _Says(sees=at, drift=drift)
  station = RackPose(*body.to_world((at.x, at.y, 0.0))[:2], at.yaw)
  assert _drive(charge.plug_in_routine(body, station, [])) == "plugged in"
  looked = body.said.index("spot")
  turned0 = sum(1 for w in body.said[:looked] if w != "spot") * math.radians(0.4)
  swing, back = charge.GRIPS[0]

  def truly(q, stance):
    """A point of the station's frame where it truly is from the feet now:
    the look's, turned back by what the feet have turned since."""
    return _turn((*st.to_world((at.x, at.y, at.yaw), *q[:2]), q[2]), turned0 - stance[2])
  grip = (-st.BOSS_M - 0.003 - st.PLUG_LEN_M + back, 0.0, st.CUP_Z)
  (line, s0), = body.reached
  assert np.allclose(_in_feet(line, s0), truly((grip[0] - charge.LINE_BACK_M, 0.0, grip[2]), s0))
  assert np.allclose(_in_feet(body.asked[0], s0),
                     charge._grip(_turn((1.0, 0.0, 0.0), turned0 - s0[2]), swing))
  # ...slid there, and there again as it is once there: the feet turned
  (to_grip, _, s1), (again, _, s2) = body.moves[:2]
  assert np.allclose(_in_feet(to_grip, s1), truly(grip, s1))
  assert np.allclose(_in_feet(again, s2), truly(grip, s2)) and not np.allclose(again, to_grip)


def test_each_try_looks_at_the_station_again(monkeypatch):
  """#596: a retry on the first try's look is a retried constant -- the
  same look, the same hand, the same miss, four times -- and pushing the
  plug home turns the body on its feet (#581). Each try is placed off a
  look of its own; one that sees nothing goes on the look it came to its
  seat by."""
  from pluggybot.body import RackPose
  monkeypatch.setattr(charge, "port", _port)
  looks = [RackPose(0.35, -0.075, 0.0), RackPose(0.33, -0.05, math.radians(-5.0)), None]
  body = _Says()
  body.sees = lambda: looks[len(body.pushed)]
  plugged = body.pull_routine

  def push(*a, **kw):
    r = yield from plugged(*a, **kw)
    body.plugged = len(body.pushed) == 3
    return r
  body.pull_routine = push
  station = RackPose(0.36, -0.07, math.radians(2.0))
  trace = []
  assert _drive(charge.plug_in_routine(body, station, trace)) == "plugged in"
  back = charge.GRIPS[0][1]
  # ...each try's move onto the plug, the one after its reach to the line
  moves = [w for w in body.said if w in ("reach", "move")]
  grips = [body.moves[moves[:i].count("move")][0] for i, w in enumerate(moves) if w == "reach"]
  assert [np.allclose(g, charge.parked_grip(s, back)) for g, s in
          zip(grips, (looks[0], looks[1], station))] == [True, True, True]
  assert [t["looked"] for t in trace] == [True, True, False]


def test_the_first_look_waits_for_the_trunks_tilt_to_settle(monkeypatch):
  """#596: for a second or two after a step, its IMU's filter read the
  trunk's tilt 0.4-0.6 deg off, and the look that placed the hand put the
  station 5-12 mm off. The first try's look is taken after `LOOK_SETTLE_S`
  standing still, a few of the filter's time constants; a later one's after
  the balancer's `PARK_WAIT_S`, no shorter."""
  from pluggybot.body import RackPose
  from pluggybot.perception.imu import TAU_S
  monkeypatch.setattr(charge, "port", _port)
  body = _Says(sees=RackPose(0.35, -0.075, 0.0))
  assert _drive(charge.plug_in_routine(body, RackPose(0.35, -0.075, 0.0), [])) == "plugged in"
  first = body.said.index("spot")
  assert ("hold", charge.LOOK_SETTLE_S) in body.said[:first]
  assert charge.LOOK_SETTLE_S >= 2.0 * TAU_S and charge.PARK_WAIT_S >= charge.LOOK_SETTLE_S


def test_the_plug_goes_to_the_port_as_read_at_each_move_and_is_let_go_where_it_stops(monkeypatch):
  """#596: read once, a posture taken to carry the plug moved the port 6 mm
  before the push. Each move goes to the port as read at its instant, and
  the push along its way out as read then. ⚠ The jaws open where the push
  stops, pressed on the body: put back on its line after (the arm's spring
  slid the hand 6-11 mm on every push), the plug was let go 9-12 mm off the
  body, near the port's 15, and the port took fewer."""
  from pluggybot.body import RackPose
  reads = []

  def moving_port(body):
    reads.append(len(reads))
    p, out = _PORT
    return body.to_world(p + (0.0, 0.003 * len(reads), 0.0)), _turn(out, body.stance[2])
  monkeypatch.setattr(charge, "port", moving_port)
  body = _Says(sees=RackPose(0.35, -0.075, 0.0))
  assert _drive(charge.plug_in_routine(body, RackPose(0.35, -0.075, 0.0), [])) == "plugged in"
  swing, back = charge.GRIPS[0]
  p, out = _PORT
  (carry, turns, _), (push_from, _, _) = body.moves[-2:]
  port_at = [p + (0.0, 0.003 * (k + 1), 0.0) for k in range(len(reads))]
  assert len(reads) == 3
  assert np.allclose(carry, port_at[0] + out * (back + charge.OUT_M))
  assert np.allclose(push_from, port_at[1] + out * (back + charge.PUSH_FROM_M))
  (way, _), = body.pushed
  assert np.allclose(way, -out)
  after = body.said[body.said.index("push") + 1:]
  assert after[:2] == ["grip", ("hold", 0.3)] and body.grips[-1] == (False, None)


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


def _sighting(s, eye_x, tags=STATION_TAG_IDS, facing=None):
  """A look of the head's camera at the station `s` from `eye_x` m out of
  its face (`HumanoidPlaces.see_station`'s record): each of `tags` where it
  is, its face out of the station, or as `facing` says."""
  out = s.yaw + math.pi if facing is None else facing
  layout = st.tag_layout()
  return {"t": 0.0, "eye": s.to_world(-eye_x, 0.0),
          "tags": {t: ((*s.to_world(0.0, layout[t][1]), layout[t][2]), out) for t in tags}}


def test_a_far_look_says_where_to_walk_nearer_to_and_only_a_near_one_fixes():
  """#572: the head's camera places a tag to a few cm from 4 m, but the
  pair's facing only to 6-66 deg there; within 1.5 m to 2.5. So a far look
  says where to walk nearer to (`rough`) -- the pair fitted, or one tag
  where it is, facing as it reads, never facing away from where it was
  seen from -- and only a near one is fitted to fix the belief by."""
  from pluggybot.body import RackPose
  s = RackPose(3.5, -1.88, -math.pi / 2)
  for tags in (STATION_TAG_IDS, STATION_TAG_IDS[:1], STATION_TAG_IDS[1:]):
    r = charge.rough(_sighting(s, 3.0, tags))
    assert (r.x, r.y, r.yaw) == pytest.approx((s.x, s.y, s.yaw)), tags
  # ...a facing read 80 deg off the way it was seen from: that way instead,
  # 3 m out on its axis and 9 cm across it from the tag
  r = charge.rough(_sighting(s, 3.0, STATION_TAG_IDS[:1], facing=s.yaw + math.pi + 1.4))
  assert abs(math.remainder(r.yaw - s.yaw, 2 * math.pi)) < math.radians(2.0)
  assert math.hypot(r.x - s.x, r.y - s.y) < 0.01
  assert charge.near_fit(_sighting(s, 3.0)) is None, "a fix off a far look"
  assert charge.near_fit(_sighting(s, 1.2, STATION_TAG_IDS[:1])) is None, "a fix off one tag"
  near = charge.near_fit(_sighting(s, 1.2))
  assert (near.x, near.y, near.yaw) == pytest.approx((s.x, s.y, s.yaw))


def test_the_head_camera_places_the_stations_tags_from_across_the_room():
  """One decode of the head's camera, 2 m out of the station's face: both
  tags placed through its model of itself (`see_station`) within a few cm,
  where the colour imager the dock looks by sees neither."""
  model, data, body, station = _house(at_seat=False)
  try:
    s, m = body._station, body.mission
    body.start_at(*s.to_world(-2.0, 0.0), s.yaw)
    body.run(body.hold_routine(0.6))
    assert charge.look(body) is None, "the colour imager read it from 2 m"
    m.look_for_places()
    seen = m.station_sighting
    assert seen is not None and set(seen["tags"]) == set(STATION_TAG_IDS)
    for t, (p, _) in seen["tags"].items():
      x, y = s.to_world(0.0, st.tag_layout()[t][1])
      assert math.hypot(p[0] - x, p[1] - y) < 0.04, (t, p, (x, y))
  finally:
    body.close()


def test_a_lost_humanoid_looks_round_by_its_head_and_walks_nearer_by_what_it_saw(monkeypatch):
  """`lost_routine` (#572): a look round where it stands first, by its
  head's camera; seen, it walks to `NEAR_M` out of the station's face as
  the look placed it -- by what it saw, never by its belief, which is what
  is wrong -- faces it, and the look that fits the station fixes the
  belief."""
  from pluggybot.body import RackPose
  model, data, body, station = _house(at_seat=False)
  try:
    s, m = body._station, body.mission
    acts = []

    def around(halt):
      acts.append("around")
      m.station_sighting = {**_sighting(s, 3.0), "t": float(data.time)}
      return True
      yield

    def drive(x, y, timeout=90.0, stop=None):
      acts.append(("walk", round(x, 3), round(y, 3)))
      return True
      yield

    def face(heading):
      return True
      yield

    def look(b):
      acts.append("look")
      return RackPose(s.x + 0.7, s.y, s.yaw)
      yield
    monkeypatch.setattr(m, "_look_around_routine", around)
    monkeypatch.setattr(m, "drive_to_routine", drive)
    monkeypatch.setattr(body, "face_routine", face)
    monkeypatch.setattr(charge, "look_routine", look)
    assert body.run(body.lost_routine("station"))
    ax, ay = s.to_world(-charge.NEAR_M, 0.0)
    assert acts == ["around", ("walk", round(ax, 3), round(ay, 3)), "look"], acts
    events = [e["why"] for e in body.belief_events]
    assert events == ["lost", "fixed", "searched"], events
    assert body.belief_events[-1]["found"] and body.belief_events[-1]["dropped"]
  finally:
    body.close()


def test_unseen_a_lost_humanoid_searches_outward_from_where_it_believes_the_station_is(
    monkeypatch):
  """No station in its look round: it walks to viewpoints round where it
  believes the station's standoff is, over its own map -- never round
  itself, wherever its lost walk ended -- a look round at each, until the
  station is seen or `stop` says."""
  from pluggybot.humanoid.places import LOOKED_M, SEARCH_STEP_M
  model, data, body, station = _house(at_seat=False)
  try:
    s, m = body._station, body.mission
    body.start_at(*s.to_world(-4.5, 2.5), s.yaw)              # lost, across the room
    body.run(body.hold_routine(1.0))
    walks, arounds = [], []

    def around(halt):
      arounds.append(body.pose_xy())
      return False
      yield

    def drive(x, y, timeout=90.0, stop=None):
      walks.append((x, y))
      return True
      yield
    monkeypatch.setattr(m, "_look_around_routine", around)
    monkeypatch.setattr(m, "drive_to_routine", drive)
    assert not body.run(body.lost_routine("station", stop=lambda: len(walks) >= 2))
    sx, sy = body.charge_standoff()[:2]
    here = body.pose_xy()
    assert len(arounds) == 1 and len(walks) == 2, (arounds, walks)
    assert math.hypot(walks[0][0] - sx, walks[0][1] - sy) <= SEARCH_STEP_M, walks
    assert math.hypot(walks[0][0] - here[0], walks[0][1] - here[1]) > LOOKED_M
    last = body.belief_events[-1]
    assert last["why"] == "searched" and last["ended"] == "interrupted" and not last["found"]
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
  assert any("I did not find my charging station where I believed it was" in ln
             and "looking round for it" in ln for ln in said), said
  assert any(ln.endswith("I looked for my charging station for 41 s and did not see it")
             for ln in said), said
  assert not any("dock" in ln or "rack" in ln for ln in said), said
