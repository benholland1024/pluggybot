"""Guards for the activity layer (issue #8): task state machines over physics.

The Threshold/toggle/telemetry tests are pure or near-pure and run in
milliseconds; the plate is pressed by a load set on it -- a sole-sized box,
or the robot set down on it -- never a robot walking.
"""

import mujoco
import pytest

from pluggybot.activity.base import (
  Activity, ActivitySet, GeomToggle, MocapToggle, Threshold,
)
from pluggybot.activity.plate import LOAD_PAINT_HALF, PlateLight, plate_light_xml
from pluggybot.humanoid import world as hw
from pluggybot.telemetry.protocol import PROTOCOL_VERSION
from pluggybot.telemetry.recorder import FrameBuilder


@pytest.fixture(scope="module")
def home_model():
  """The house alone: its activities, and no robot."""
  return mujoco.MjModel.from_xml_path("models/home_world.xml")


@pytest.fixture(scope="module")
def served_model():
  """The served house, `home_humanoid`: a stream needs a robot on it."""
  return hw.home_spec().compile()


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
  lamp = GeomToggle(home_model, "garden_light_bulb",
                    {"a": {"rgba": [1, 0, 0, 1]}, "b": {"rgba": [0, 1, 0, 1]}})
  lamp.select("a")
  assert list(home_model.geom_rgba[lamp.gid]) == [1, 0, 0, 1]
  lamp.select("b")
  assert list(home_model.geom_rgba[lamp.gid]) == [0, 1, 0, 1]


def test_an_unknown_state_is_an_error_not_a_no_op(home_model):
  lamp = GeomToggle(home_model, "garden_light_bulb", {"a": {"rgba": [1, 0, 0, 1]}})
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
  failure -- re-measure, and MocapToggle can retire (its last live consumer,
  the garden gate, already did -- issue #93).
  """
  data = mujoco.MjData(home_model)
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


#: MocapToggle's own little world. It used to be tested against the garden
#: GATE, which issue #93 removed -- the tool now has NO live consumer, and
#: that is precisely why these tests moved to a synthetic model instead of
#: being deleted with the gate: the welded-body trap (above) is still real,
#: `MocapToggle` is still the answer for the next activity that must MOVE
#: something, and a tool guarded only by its consumer dies with it.
_MOCAP_XML = """<mujoco><worldbody>
  <body name="panel" mocap="true" pos="0 0 0.5">
    <geom name="panel_geom" type="box" size="0.5 0.02 0.5"/>
  </body>
  <body name="jointed"><joint type="slide" axis="0 0 1"/>
    <geom type="box" size="0.1 0.1 0.1"/></body>
</worldbody></mujoco>"""


def test_mocap_toggle_actually_moves_the_body():
  """The fix the welded-body trap demands: a mocap pose IS re-read by
  kinematics every step."""
  model = mujoco.MjModel.from_xml_string(_MOCAP_XML)
  data = mujoco.MjData(model)
  toggle = MocapToggle(model, data, "panel", {
    "up": {"pos": [0.0, 0.0, 0.5]},
    "down": {"pos": [0.0, 0.0, -0.7]},
  })
  toggle.select("up")
  mujoco.mj_forward(model, data)
  up = float(data.geom_xpos[model.geom("panel_geom").id][2])
  toggle.select("down")
  mujoco.mj_forward(model, data)
  assert float(data.geom_xpos[model.geom("panel_geom").id][2]) < up - 1.0, \
    "the panel did not move -- the toggle is inert again"


def test_mocap_toggle_refuses_a_non_mocap_body():
  model = mujoco.MjModel.from_xml_string(_MOCAP_XML)
  data = mujoco.MjData(model)
  with pytest.raises(ValueError, match="mocap"):
    MocapToggle(model, data, "jointed", {"a": {"pos": [0, 0, 0]}})


# ---- the reference activity -------------------------------------------------

def test_every_plate_is_flush_with_the_floor(home_model):
  """#684 (Ben's calls, 2026-11-12): the garden's pad, 21 mm proud on a
  spring, tripped the served walker onto the lamp's pole, and the lab's three,
  the same pad, stalled it at their edge from a standstill; now each plate's
  top is the floor's -- nothing of it moves or touches anything, its pad
  paint over the ground's, its load read off the floor's own square."""
  data = mujoco.MjData(home_model)
  mujoco.mj_forward(home_model, data)
  for prefix in ("garden", "lab_shock", "lab_feed", "lab_toy"):
    pad = home_model.geom(f"{prefix}_plate_pad").id
    assert home_model.body(f"{prefix}_plate").jntnum == 0, prefix
    assert home_model.geom_contype[pad] == home_model.geom_conaffinity[pad] == 0, prefix
    top = data.geom_xpos[pad][2] + home_model.geom_size[pad][2]
    assert top <= 2 * LOAD_PAINT_HALF + 1e-9, f"{prefix}: paint, never a step"
    zone = home_model.site(f"{prefix}_plate_zone").id
    assert home_model.site_bodyid[zone] == 0 and data.site_xpos[zone][2] == 0.0, \
      f"{prefix}: the floor's own square"
    sensor = home_model.sensor(f"{prefix}_plate_load")
    assert sensor.type[0] == mujoco.mjtSensor.mjSENS_TOUCH and sensor.objid[0] == zone


def test_a_flush_plate_with_nothing_on_it_reads_nothing(home_model):
  """...and with nothing on it, it reads nothing: a plate that pressed
  itself would be on from t=0, and nothing about it testable."""
  data = mujoco.MjData(home_model)
  light = PlateLight(home_model, data)
  for _ in range(500):
    mujoco.mj_step(home_model, data)
    light.sense(home_model, data)
  assert light.load(data) == 0.0
  assert light.flags == {"state": "off", "pressed": False}


def _plate_and_a_foot(kg: float):
  """A floor, the garden's plate and lamp at the origin, and a box the size
  of the humanoid's sole under `kg`, standing 1 m east of the plate."""
  robot = mujoco.MjModel.from_xml_path("models/humanoid.xml")
  sole = robot.geom_size[robot.geom("right_sole").id]
  plate, sensor = plate_light_xml((0.0, 0.0))
  model = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>{plate}
    <body name="foot" pos="1 0 {sole[2]:.4f}"><freejoint/>
      <geom type="box" size="{sole[0]:.4f} {sole[1]:.4f} {sole[2]:.4f}" mass="{kg}"/>
    </body></worldbody><sensor>{sensor}</sensor></mujoco>""")
  return model, mujoco.MjData(model), robot


def test_a_foot_on_the_flush_plate_presses_it_and_lifted_off_it_latches_the_light_on():
  """The flush plate's whole pattern through the solver: a sole under a
  third of the humanoid's weight -- less than one foot carries standing --
  set on it presses it, the light comes on, and lifted off it is released
  and the light stays on. The floor's square reads all of what stands on
  it."""
  _, _, robot = _plate_and_a_foot(1.0)
  weight_n = float(robot.body_subtreemass[robot.body("pluggybot").id]) * 9.81
  model, data, _ = _plate_and_a_foot(weight_n / 3 / 9.81)
  light = PlateLight(model, data)
  adr = model.joint(model.body("foot").jntadr[0]).qposadr[0]

  def settle(x: float, z: float = 0.0) -> None:
    data.qpos[adr:adr + 3] = [x, 0.0, data.qpos[adr + 2] + z]
    data.qvel[:] = 0.0
    for _ in range(150):
      mujoco.mj_step(model, data)
      light.sense(model, data)

  settle(1.0)
  assert light.flags["pressed"] is False and light.flags["state"] == "off"
  settle(0.0)
  assert light.load(data) == pytest.approx(weight_n / 3, rel=0.02)
  assert light.flags["pressed"] and light.flags["state"] == "on"
  settle(1.0, z=0.5)
  assert light.load(data) == 0.0 and not light.flags["pressed"], "pressed is a LIVE flag"
  assert light.flags["state"] == "on", "state is LATCHED -- a real motion "\
      "light times out; this one leaving a mark is the activity's point"


def test_a_save_from_the_sprung_plates_days_comes_back_lit_and_without_its_depth(home_model):
  """A world saved before #684 carries the garden light's flags with the
  sprung pad's `depressMm`: put back, the light is as it was and the depth
  is gone from the wire, never shipped for good beside the flags it has."""
  light = PlateLight(home_model, mujoco.MjData(home_model))
  light.restore_kept({"flags": {"state": "on", "pressed": False, "depressMm": 1},
                      "pressed": False, "latched": True})
  assert light.flags == {"state": "on", "pressed": False}
  assert light.latch.value and light.lamp.current == "on"


# ---- telemetry --------------------------------------------------------------

class _Fake(Activity):
  """A hand-driven activity, so the frame tests need no physics."""

  name = "fake"

  def sense(self, model, data) -> None:
    pass


@pytest.fixture
def builder_pair(served_model):
  """Two FrameBuilders over ONE ActivitySet -- serve.py --record's shape."""
  data = mujoco.MjData(served_model)
  act = _Fake()
  act.set(state="closed")
  acts = ActivitySet([act])
  fb = FrameBuilder(served_model, data, hz=1000.0, model_name="home_humanoid",
                    keyframe_s=0.0, activities=acts)
  return data, act, acts, fb


def test_header_advertises_activities(builder_pair):
  _, _, _, fb = builder_pair
  h = fb.header()
  assert h["activities"] == ["fake"]
  # The literal is deliberate: a version bump is a two-repo event, so it
  # SHOULD have to touch a test rather than slide through. Bumped 0.5.0 ->
  # 0.6.0 with the ledger block and `earned` messages (issue #14), and
  # 0.6.0 -> 0.7.0 when the socket became two-way (issue #16): visitor
  # visitor messages and ratings DOWN, `visitor_reply` and `journal` UP.
  # 0.7.0 -> 0.8.0 for the `goals` message (rooftop-media-2026 #30): what the
  # robot is FOR, sent when a stream opens, so the site can show it. 0.8.0 ->
  # 0.9.0 for the `tasks` block and its three messages (issue #21): the robot
  # is GIVEN work, and the block is the one that ships WHOLE rather than as a
  # per-key delta, because a task can cease to exist. 0.9.0 -> 0.10.0 for
  # `robotNames` in the header (issue #39): the robot has a display NAME,
  # keyed by the species (ROBOT_ROOT), which stays the id of everything else.
  # 0.10.0 -> 0.11.0 for the thought files (issue #38), and 0.11.0 -> 0.12.0
  # for money and the operator's switch (issue #37): a `spend` block, `mode`
  # on the robot's record, and a `mode` MESSAGE -- which exists because a
  # paused robot steps no physics and so sends no frames at all, leaving the
  # field that rides them unable to say so. 0.12.0 -> 0.13.0 for the
  # `metabolism` block (issue #36): points are food, and how hungry the robot
  # is rides every frame. 0.13.0 -> 0.14.0 for the visitor channel collapsing
  # to ONE inbound kind (issue #61): `suggestion` and `question` were a
  # classification asked of the sender, who was the wrong party for it, and
  # what the robot DID about a message -- `replied` where the old vocabulary
  # said `answered` -- is the distinction that survived. 0.14.0 -> 0.15.0
  # for DEATH (issue #107): the robot can now die -- `flat` (the pack
  # reached zero, a decision failure) or `stuck` (knocked over, or unable
  # to reach the rack, a physics one), never summed -- and stands where it
  # fell until an admin's `reset_robot` picks it up, with the survival
  # clock riding every frame beside the battery. 0.16.0 -> 0.17.0 for
  # HEARTS (issues #135, #136) and its third death cause, `unpaid`; 0.17.0
  # -> 0.18.0 for the fourth, `unminded` (issue #127) -- an agent that
  # configured its own event map so that nothing consults its mind any more.
  # A consumer that renders causes has to know the producer can emit it.
  # 0.18.0 -> 0.19.0: the robot owns its goals (issue #154). 0.19.0 ->
  # 0.20.0: every robot on the stream reads the same shape (issue #167) --
  # `spend` and `metabolism` under `robots[<root>]`, one `goals` and one
  # `grid` message per robot.
  assert h["protocolVersion"] == PROTOCOL_VERSION == "0.21.0"
  # ...and this builder has no task board, so it must say so rather than
  # advertise a vocabulary it will never use -- the `accepts` rule.
  assert h["taskKinds"] == []


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


def test_two_sinks_over_one_activity_set_stay_independent(served_model):
  """`serve.py --record` runs a publisher AND a recorder over one physics,
  each with its own FrameBuilder. This failed while it was being written:
  the "already emitted" memory lived on the Activity, so the two sinks ate
  each other's deltas and each shipped a random half of the changes.
  """
  data = mujoco.MjData(served_model)
  act = _Fake()
  act.set(state="closed")
  acts = ActivitySet([act])
  a = FrameBuilder(served_model, data, hz=1000.0, keyframe_s=0.0, activities=acts)
  b = FrameBuilder(served_model, data, hz=1000.0, keyframe_s=0.0, activities=acts)
  assert a.build()["activities"] == {"fake": {"state": "closed"}}
  assert b.build()["activities"] == {"fake": {"state": "closed"}}, \
    "the second sink lost the opening state to the first"
  act.set(state="open")
  data.time += 1.0
  assert a.build()["activities"] == {"fake": {"state": "open"}}
  assert b.build()["activities"] == {"fake": {"state": "open"}}, \
    "the second sink lost a state change to the first"


def test_a_real_light_turning_on_reaches_a_telemetry_frame(served_model):
  """End to end: a load on the plate -> flag flips -> the flip is on the wire.

  Activity state being CARRIED is not a CHANGE propagating: here the plate
  is pressed through the solver -- the robot set on it from 2 s, its feet
  on the plate, to 4 s, then set down 3 m off it -- the hook senses it, and
  the frames carry the light coming on, and keep it on, sparsely.
  """
  from pluggybot.humanoid import model as hm
  data = mujoco.MjData(served_model)
  acts = ActivitySet([PlateLight(served_model, data)])
  fb = FrameBuilder(served_model, data, hz=20.0, model_name="home_humanoid",
                    activities=acts)
  plate = served_model.body("garden_plate").pos[:2]
  root = served_model.joint(f"{hm.ROBOT_ROOT}_root")
  stand = hm.stand_qpos()

  def set_down(x: float, y: float) -> None:
    adr, dof = root.qposadr[0], root.dofadr[0]
    data.qpos[adr:adr + 7] = [x, y, hm.stand_height(), 1.0, 0.0, 0.0, 0.0]
    for name in hm.qpos_order():
      data.qpos[served_model.joint(name).qposadr[0]] = stand.get(name, 0.0)
    data.qvel[dof:dof + 6] = 0.0
  seen = []
  hook = acts.step_hook(served_model, data)
  steps = round(6.0 / served_model.opt.timestep)
  for k in range(steps):
    if k == steps // 3:
      set_down(*plate)
    elif k == 2 * steps // 3:
      set_down(plate[0] + 3.0, plate[1])
    mujoco.mj_step(served_model, data)
    hook()
    frame = fb.build()
    if frame is not None and "activities" in frame:
      seen.append((frame["t"], frame["activities"]["garden_light"]))

  states = [f["state"] for _, f in seen if "state" in f]
  assert states[0] == "off"
  assert "on" in states, "the light came on in the sim but never on the wire"
  # ...and it stays on in every later frame that mentions state at all
  after = states[states.index("on"):]
  assert set(after) == {"on"}, f"the wire turned the light back off: {after}"
  assert any(f.get("pressed") is False for _, f in seen[1:]), \
    "the load came off and `pressed` never said so"
  assert len(seen) < fb.frames * 0.35, (
    f"activity flags shipped in {len(seen)}/{fb.frames} frames -- that is "
    "not sparse; check the analogue-flag quantisation")


def test_activity_set_snapshot_and_hook(home_model):
  data = mujoco.MjData(home_model)
  acts = ActivitySet([PlateLight(home_model, data)])
  hook = acts.step_hook(home_model, data)
  mujoco.mj_forward(home_model, data)
  hook()
  assert acts.names == ["garden_light"]
  assert acts.snapshot()["garden_light"]["state"] == "off"
  assert len(acts) == 1
