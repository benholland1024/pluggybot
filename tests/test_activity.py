"""Guards for the activity layer (issue #8): task state machines over physics.

Split by cost. The Threshold/toggle/telemetry tests are pure or near-pure and
run in milliseconds; only the last one drives the robot.
"""

import math

import mujoco
import pytest

from pluggybot.activity.base import (
  Activity, ActivitySet, GeomToggle, MocapToggle, Threshold,
)
from pluggybot.activity.plate import (
  PLATE_ON, PLATE_TRAVEL, PlateGate, plate_center,
)
from pluggybot.control import slew, wheel_targets
from pluggybot.telemetry.protocol import PROTOCOL_VERSION
from pluggybot.telemetry.recorder import FrameBuilder


@pytest.fixture(scope="module")
def home_model():
  return mujoco.MjModel.from_xml_path("models/home_world.xml")


# ---- Threshold: hysteresis and latching -------------------------------------

def test_hysteresis_needs_both_edges_to_flip():
  t = Threshold(on=0.006, off=0.003)
  assert t.update(0.0) is False
  assert t.update(0.005) is False        # below the on-level
  assert t.update(0.007) is True         # crosses on
  assert t.update(0.004) is True         # between the levels: HOLDS
  assert t.update(0.002) is False        # crosses off


def test_a_bare_threshold_chatters_where_hysteresis_does_not():
  """The defect hysteresis exists to prevent, pinned as a unit test.

  A signal dithering around one level flips a bare comparison on every
  wobble; the same signal moves a hysteretic one once. Measured on the real
  plate under a wheel: 4 flips bare, 2 with -- and 2 is the floor, being one
  press and one release.
  """
  dither = [0.0, 0.0065, 0.0055, 0.0065, 0.0055, 0.0065, 0.0]
  bare, hyst = Threshold(on=0.006), Threshold(on=0.006, off=0.003)

  def flips(th):
    n, last = 0, False
    for x in dither:
      v = th.update(x)
      if v != last:
        n, last = n + 1, v
    return n

  assert flips(bare) > flips(hyst), "hysteresis bought nothing"
  assert flips(hyst) == 2, "a single crossing should be one press, one release"


def test_a_latch_is_one_way():
  t = Threshold(on=0.5, latch=True)
  assert t.update(0.0) is False
  assert t.update(1.0) is True
  assert t.update(0.0) is True, "a latched fact must not un-happen"


# ---- toggles ----------------------------------------------------------------

def test_geom_rgba_toggles_take_effect(home_model):
  lamp = GeomToggle(home_model, "garden_lamp_bulb",
                    {"a": {"rgba": [1, 0, 0, 1]}, "b": {"rgba": [0, 1, 0, 1]}})
  lamp.select("a")
  assert list(home_model.geom_rgba[lamp.gid]) == [1, 0, 0, 1]
  lamp.select("b")
  assert list(home_model.geom_rgba[lamp.gid]) == [0, 1, 0, 1]


def test_an_unknown_state_is_an_error_not_a_no_op(home_model):
  lamp = GeomToggle(home_model, "garden_lamp_bulb", {"a": {"rgba": [1, 0, 0, 1]}})
  with pytest.raises(KeyError):
    lamp.select("nope")


def test_geom_pos_does_nothing_on_a_body_welded_to_the_world(home_model):
  """The trap that cost this build an afternoon, pinned so nobody re-learns it.

  A body welded to the world (all static scenery) has its geoms' WORLD poses
  computed once when MjData is created; `mj_kinematics` never revisits them.
  So writing `model.geom_pos` lands in the model, changes nothing anybody
  reads, and reports no error -- and the renderer draws `geom_xpos`.

  Asserting the DEFECT, deliberately: it is why `MocapToggle` exists. If
  MuJoCo ever starts honouring this, that is good news wearing a test
  failure -- re-measure, and simplify the gate back to a geom toggle.
  """
  data = mujoco.MjData(home_model)
  gid = home_model.geom("garden_plate_pad").id      # a pad on a JOINTED body
  bid = home_model.geom_bodyid[gid]
  assert home_model.body_weldid[bid] != 0, "the plate should not be welded"

  fence = home_model.geom("fence_east_0_geom")      # welded scenery
  fid, fbid = fence.id, home_model.geom_bodyid[fence.id]
  assert home_model.body_weldid[fbid] == 0, "the fence should be welded"
  mujoco.mj_forward(home_model, data)
  before = float(data.geom_xpos[fid][2])
  home_model.geom_pos[fid] = [0.0, 0.0, -1.2]
  mujoco.mj_forward(home_model, data)
  mujoco.mj_step(home_model, data)
  assert float(data.geom_xpos[fid][2]) == pytest.approx(before), (
    "geom_pos now moves welded scenery -- MocapToggle may be unnecessary")
  home_model.geom_pos[fid] = [0.0, 0.0, 0.0]        # leave the fixture clean


def test_mocap_toggle_actually_moves_the_gate(home_model):
  """The fix: a mocap pose IS re-read by kinematics every step."""
  data = mujoco.MjData(home_model)
  gate = PlateGate(home_model, data)
  gid = home_model.geom("garden_gate_panel").id
  mujoco.mj_forward(home_model, data)
  closed = float(data.geom_xpos[gid][2])
  gate.gate.select("open")
  mujoco.mj_forward(home_model, data)
  assert float(data.geom_xpos[gid][2]) < closed - 1.0, \
    "the gate did not move -- the toggle is inert again"


def test_mocap_toggle_refuses_a_non_mocap_body(home_model):
  data = mujoco.MjData(home_model)
  with pytest.raises(ValueError, match="mocap"):
    MocapToggle(home_model, data, "garden_plate", {"a": {"pos": [0, 0, 0]}})


# ---- the reference activity -------------------------------------------------

def test_plate_rests_below_its_own_trigger(home_model):
  """The pad's own weight must not press it. If a plate triggers itself the
  activity is on from t=0 and nothing about it is testable."""
  data = mujoco.MjData(home_model)
  gate = PlateGate(home_model, data)
  for _ in range(2000):
    mujoco.mj_step(home_model, data)
    gate.sense(home_model, data)
  assert gate.depth(data) < PLATE_ON, "the plate presses itself"
  assert gate.flags == {"state": "closed", "pressed": False, "depressMm": 1}


def test_pressing_latches_the_gate_open_and_releasing_does_not_close_it(home_model):
  data = mujoco.MjData(home_model)
  gate = PlateGate(home_model, data)
  adr = home_model.joint("garden_plate_joint").qposadr[0]
  mujoco.mj_forward(home_model, data)
  gate.sense(home_model, data)
  assert gate.flags["state"] == "closed"

  data.qpos[adr] = -PLATE_TRAVEL * 0.9
  mujoco.mj_forward(home_model, data)
  gate.sense(home_model, data)
  assert gate.flags["pressed"] and gate.flags["state"] == "open"

  data.qpos[adr] = 0.0
  mujoco.mj_forward(home_model, data)
  gate.sense(home_model, data)
  assert not gate.flags["pressed"], "pressed is a LIVE flag"
  assert gate.flags["state"] == "open", "state is LATCHED"


def test_the_robot_can_actually_drive_onto_the_plate(home_model):
  """The criterion is only worth anything if the robot can trip it.

  A plate too tall for a 90 mm wheel to climb, or too stiff for its load,
  would pass every unit test above and be useless.
  """
  data = mujoco.MjData(home_model)
  gate = PlateGate(home_model, data)
  px, py = plate_center(home_model)
  yaw = math.pi
  data.qpos[0], data.qpos[1], data.qpos[2] = px + 1.2, py, 0.045
  data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(home_model, data)
  left = home_model.actuator("left_motor").id
  right = home_model.actuator("right_motor").id
  tl, tr = wheel_targets(0.25, 0.0)
  deepest = 0.0
  for _ in range(9000):
    data.ctrl[left] = slew(data.ctrl[left], tl, home_model.opt.timestep)
    data.ctrl[right] = slew(data.ctrl[right], tr, home_model.opt.timestep)
    mujoco.mj_step(home_model, data)
    gate.sense(home_model, data)
    deepest = max(deepest, gate.depth(data))
  assert deepest > PLATE_ON, (
    f"a wheel only pressed the plate {deepest * 1000:.1f} mm against a "
    f"{PLATE_ON * 1000:.1f} mm trigger")
  assert gate.flags["state"] == "open"


# ---- telemetry --------------------------------------------------------------

class _Fake(Activity):
  """A hand-driven activity, so the frame tests need no physics."""

  name = "fake"

  def sense(self, model, data) -> None:
    pass


@pytest.fixture
def builder_pair(home_model):
  """Two FrameBuilders over ONE ActivitySet -- serve.py --record's shape."""
  data = mujoco.MjData(home_model)
  act = _Fake()
  act.set(state="closed")
  acts = ActivitySet([act])
  fb = FrameBuilder(home_model, data, hz=1000.0, model_name="home_world",
                    keyframe_s=0.0, activities=acts)
  return data, act, acts, fb


def test_header_advertises_activities(builder_pair):
  _, _, _, fb = builder_pair
  h = fb.header()
  assert h["activities"] == ["fake"]
  assert h["protocolVersion"] == PROTOCOL_VERSION == "0.4.0"


def test_activity_flags_ride_in_frames_and_are_sparse(builder_pair):
  data, act, _, fb = builder_pair
  first = fb.build()
  assert first["activities"] == {"fake": {"state": "closed"}}, \
    "the opening keyframe must carry activity state"
  data.time += 1.0
  assert "activities" not in fb.build(), "unchanged state must not re-ship"
  act.set(state="open")
  data.time += 1.0
  assert fb.build()["activities"] == {"fake": {"state": "open"}}


def test_a_keyframe_reships_activity_state(builder_pair):
  """A mid-stream joiner has never seen the flags, and for a gate moved by
  toggle the flag is the ONLY record -- the panel sits on a static body that
  ships once in the scene and never again."""
  data, act, _, fb = builder_pair
  fb.build()
  data.time += 1.0
  assert "activities" not in fb.build()
  fb.reset()                                  # what a reconnect does
  data.time += 1.0
  frame = fb.build()
  assert frame.get("key") is True
  assert frame["activities"] == {"fake": {"state": "closed"}}


def test_two_sinks_over_one_activity_set_stay_independent(home_model):
  """`serve.py --record` runs a publisher AND a recorder over one physics,
  each with its own FrameBuilder. This failed while it was being written:
  the "already emitted" memory lived on the Activity, so the two sinks ate
  each other's deltas and each shipped a random half of the changes.
  """
  data = mujoco.MjData(home_model)
  act = _Fake()
  act.set(state="closed")
  acts = ActivitySet([act])
  a = FrameBuilder(home_model, data, hz=1000.0, keyframe_s=0.0, activities=acts)
  b = FrameBuilder(home_model, data, hz=1000.0, keyframe_s=0.0, activities=acts)
  assert a.build()["activities"] == {"fake": {"state": "closed"}}
  assert b.build()["activities"] == {"fake": {"state": "closed"}}, \
    "the second sink lost the opening state to the first"
  act.set(state="open")
  data.time += 1.0
  assert a.build()["activities"] == {"fake": {"state": "open"}}
  assert b.build()["activities"] == {"fake": {"state": "open"}}, \
    "the second sink lost a state change to the first"


def test_a_real_gate_opening_reaches_a_telemetry_frame(home_model):
  """End to end: robot presses plate -> flag flips -> the flip is on the wire.

  Worth having as a test rather than trusting the committed recording. The
  home lifecycle never happens to drive over the garden plate, so the
  fixture shows `closed` for its whole length -- which proves activity
  state is *carried* but not that a CHANGE propagates. This closes that gap
  without inventing a fixture nobody replays.
  """
  data = mujoco.MjData(home_model)
  acts = ActivitySet([PlateGate(home_model, data)])
  fb = FrameBuilder(home_model, data, hz=20.0, model_name="home_world",
                    activities=acts)
  px, py = plate_center(home_model)
  yaw = math.pi
  data.qpos[0], data.qpos[1], data.qpos[2] = px + 1.2, py, 0.045
  data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(home_model, data)
  left = home_model.actuator("left_motor").id
  right = home_model.actuator("right_motor").id
  tl, tr = wheel_targets(0.25, 0.0)
  seen = []
  hook = acts.step_hook(home_model, data)
  for _ in range(9000):
    data.ctrl[left] = slew(data.ctrl[left], tl, home_model.opt.timestep)
    data.ctrl[right] = slew(data.ctrl[right], tr, home_model.opt.timestep)
    mujoco.mj_step(home_model, data)
    hook()
    frame = fb.build()
    if frame is not None and "activities" in frame:
      seen.append((frame["t"], frame["activities"]["garden_gate"]))

  states = [f["state"] for _, f in seen if "state" in f]
  assert states[0] == "closed"
  assert "open" in states, "the gate opened in the sim but never on the wire"
  # ...and it stays open in every later frame that mentions state at all
  after = states[states.index("open"):]
  assert set(after) == {"open"}, f"the wire un-opened the gate: {after}"
  assert len(seen) < fb.frames * 0.35, (
    f"activity flags shipped in {len(seen)}/{fb.frames} frames -- that is "
    "not sparse; check the analogue-flag quantisation")


def test_activity_set_snapshot_and_hook(home_model):
  data = mujoco.MjData(home_model)
  acts = ActivitySet([PlateGate(home_model, data)])
  hook = acts.step_hook(home_model, data)
  mujoco.mj_forward(home_model, data)
  hook()
  assert acts.names == ["garden_gate"]
  assert acts.snapshot()["garden_gate"]["state"] == "closed"
  assert len(acts) == 1
