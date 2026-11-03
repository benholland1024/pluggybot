"""The grown world's three charging stations (issue #560): one in each
building, each commissioned, each its own pair of tags. Which one a charge
goes to is the robot's body's to choose (`Body.choose_station`): the one
its plug is in, else the nearest by a walk over its OWN map. A station seen
is named by its pair, and a restart keeps the choice. Pinned on one
compiled world, nothing stepped; the loop's half (a search that found
another station) is `test_fixtures.py`'s. Each station's own routines are
`test_station.py`'s."""

import math
import sys
from pathlib import Path

import mujoco
import pytest

from pluggybot.body import body_for
from pluggybot.home import world as home
from pluggybot.humanoid.world import home_spec, stations
from pluggybot.robot import FIRST

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def world():
  model = home_spec().compile()
  return model, mujoco.MjData(model)


def _body(world):
  model, data = world
  return body_for(model, data, FIRST, grid_bounds=home.GRID_BOUNDS)


def _true_map(body):
  """Its map the world's static floor and walls (`make_way_spike.true_floor`):
  a robot that has walked everywhere."""
  sys.path.insert(0, str(ROOT / "scripts"))
  from make_way_spike import true_floor
  g = body.mission.grid
  g.grid[:] = true_floor(body.model, g)


def test_the_world_has_a_station_in_each_building_each_with_its_own_pair(world):
  model, _ = world
  frames = _body(world).mission.stations
  assert list(frames) == list(stations()) == ["station", "station_facility", "station_mall"]
  pairs = [tags for _, tags in frames.values()]
  assert len({t for p in pairs for t in p}) == 6, f"two stations share a tag: {pairs}"
  for name, (frame, _) in frames.items():
    x, y, yaw = stations()[name][0]
    assert (frame.x, frame.y) == pytest.approx((x, y)) and \
        abs(math.remainder(frame.yaw - yaw, 2 * math.pi)) < 1e-6


@pytest.mark.parametrize("at, nearest", [
  ((1.5, 0.5), "station"),                 # the living room
  ((20.5, 2.0), "station_facility"),       # the research center's lobby
  ((17.5, -15.0), "station_mall"),         # the mall's concourse
  # the park's west sidewalk: the house's station is nearer as the crow
  # flies (22.8 m) and further by every way round the fenced park (49 m
  # against the mall's 47)
  ((-12.75, -17.0), "station_mall"),
])
def test_a_charge_goes_to_the_nearest_station_by_a_walk_over_its_own_map(world, at, nearest):
  body = _body(world)
  try:
    _true_map(body)
    body.mission._set_pose(at[0], at[1], 0.0)
    body.choose_station()
    assert body.mission.station == nearest
    assert body.charge_standoff() == pytest.approx(
      (*stations()[nearest][0][:2], stations()[nearest][0][2]), abs=1.0)
  finally:
    body.close()


def test_with_nothing_mapped_the_walk_through_the_unknown_decides(world):
  # ...its OWN map, never the world's: from the park's west sidewalk with
  # nothing seen, the house's station is the nearer walk through the
  # unknown, where the true map's walk round the park makes it the mall's
  body = _body(world)
  try:
    body.mission._set_pose(-12.75, -17.0, 0.0)
    body.choose_station()
    assert body.mission.station == "station"
  finally:
    body.close()


def test_plugged_in_it_charges_where_its_plug_is(world):
  model, data = world
  body = _body(world)
  e = model.equality("station_mall_port_pluggybot").id
  try:
    body.mission._set_pose(1.5, 0.5, 0.0)       # the living room, the house's nearest
    data.eq_active[e] = 1
    body.choose_station()
    assert body.mission.station == "station_mall" and body.station_tags == (66, 67)
  finally:
    data.eq_active[e] = 0
    body.close()


def test_a_sighting_names_the_station_its_pair_belongs_to(world):
  # a lost robot looking for one station sees another as often: the
  # sighting says which, and its fit is that station's pair
  from pluggybot.humanoid import charge
  body = _body(world)
  try:
    m = body.mission
    face = {"t": (0.0, 0.0, 1.0), "normal": (0.0, 0.0, 1.0), "yaw": 0.0}
    m.see_station({64: dict(face, t=(0.09, 0.0, 1.0)), 65: dict(face, t=(-0.09, 0.0, 1.0))})
    seen = m.station_sighting
    assert seen["station"] == "station_facility" and seen["ids"] == (64, 65)
    assert set(seen["tags"]) == {64, 65}
    assert charge.rough(seen) is not None
    # ...of two stations in one look, the one more of whose tags it saw
    m.see_station({58: face, 66: dict(face, t=(0.09, 0.0, 2.0)),
                   67: dict(face, t=(-0.09, 0.0, 2.0))})
    assert m.station_sighting["station"] == "station_mall"
  finally:
    body.close()


def test_a_fix_is_laid_off_the_station_whose_pair_it_fitted(world, monkeypatch):
  # the colour imager's look failed and the head's fresh decode fitted
  # ANOTHER station's pair: the belief is put right off that station's
  # commissioned pose, never off the one the try began with
  from pluggybot.body import RackPose
  from pluggybot.humanoid import charge
  _, data = world
  body = _body(world)
  try:
    m = body.mission
    m.station, m.station_sighting = "station", {"station": "station", "t": 0.0}

    def done(*a, **kw):
      return True
      yield

    def unseen(*a, **kw):
      return None
      yield

    def decoded():
      m.station_sighting = {"station": "station_mall", "t": float(data.time) + 1.0}
    monkeypatch.setattr(m, "drive_to_routine", done)
    monkeypatch.setattr(body, "face_routine", done)
    monkeypatch.setattr(m, "look_for_places", decoded)
    monkeypatch.setattr(charge, "rough", lambda seen: RackPose(0.0, 0.0, 0.0))
    monkeypatch.setattr(charge, "look_routine", unseen)
    monkeypatch.setattr(charge, "near_fit", lambda seen: RackPose(1.0, 2.0, 0.0))
    monkeypatch.setattr(body, "_fix_at", lambda seen: {"at": m.station})
    assert body.run(body._near_station_routine(lambda: False)) == (True, {"at": "station_mall"})
  finally:
    body.close()


def test_a_station_whose_plug_is_parked_reads_no_port_once_armed(world, monkeypatch):
  # three stations sensed every step, most with nobody near: armed beside a
  # parked plug a port has nothing to take and nothing to re-arm, and its
  # offset is not read (32 us a station a step, 12 without it)
  from pluggybot.activity.station import Station
  model, data = world
  mujoco.mj_forward(model, data)
  s = Station(model, data, "station_mall")
  read = []
  monkeypatch.setattr(s, "_port_offset", lambda d, port: read.append(port) or (1.0, 1.0, 0.0))
  monkeypatch.setattr(s, "_take", lambda d, eq: None)
  assert data.eq_active[s.cup]
  s.sense(model, data)
  assert read == []
  # ...a port still to re-arm is read, to be
  eq, (_, port) = next(iter(s.ports.items()))
  s._armed[eq] = False
  s.sense(model, data)
  assert read == [port] and s._armed[eq]
  # ...and out of its cup, every port is read again
  read.clear()
  data.eq_active[s.cup] = 0
  try:
    s.sense(model, data)
    assert len(read) == len(s.ports)
  finally:
    data.eq_active[s.cup] = 1


def test_the_station_a_charge_went_to_rides_a_restart(world):
  # restarted plugged in at the mall, an unplug would reach for the house's
  # plug: the choice is kept state (`HumanoidMission.kept_state`)
  body = _body(world)
  try:
    body.mission.station = "station_facility"
    state, arrays = body.kept_state()
  finally:
    body.close()
  back = _body(world)
  try:
    assert back.mission.station == "station"
    back.restore_kept(state, arrays)
    assert back.mission.station == "station_facility"
    assert back.station_tags == (64, 65)
  finally:
    back.close()
