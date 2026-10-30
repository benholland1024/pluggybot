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
from pluggybot.rack.tags import STATION_TAG_IDS


def _house(at_seat=True, pair=False):
  """The house, a humanoid at the station's seat, its activities hooked:
  (model, data, body, station)."""
  from pluggybot.body import body_for
  from pluggybot.lifecycle import HUMANOID_HOME, home_activities, world_config
  from pluggybot.robot import world_spec
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"], cfg["start2"][:2] if pair else None, body="humanoid").compile()
  data = mujoco.MjData(model)
  body = body_for(model, data)
  s = body._station
  x, y = s.to_world(st.SEAT_X if at_seat else st.STANDOFF_X, st.SEAT_Y)
  body.start_at(x, y, s.yaw)
  acts = home_activities(model, data)
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
  _at_port(model, data, station, gap=0.008)
  _hold(body, 0.1)
  assert not station.flags["mated"], "taken again inside its reach"
  _at_port(model, data, station, gap=st.CLEAR_M + 0.01)
  _hold(body, 0.05)
  _at_port(model, data, station, gap=0.008)
  _hold(body, 0.2)
  assert station.flags["mated"]


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
