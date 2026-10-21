"""The humanoid (issue #507, stages 1 and 2): our own body from real parts,
its model, drivers and sensors; and standing, reaching and gripping off its
own senses. Each rule pinned as cheaply as it fails for the
right reason: the generator against its committed files, a joint's driver
against its part's datasheet, a motor's mass against its sheet, the soles
and the pads against MuJoCo's friction MAX, the gripper's squeeze against
its sheet's force, and what the drivers report against their fields.
SimNotes and Parts.md, "The humanoid body", carry the design."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.humanoid.senses import GripperSense, JointSense
from pluggybot.legs.actuator import JointLimits, driver_gains

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def standing():
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return model, data


def test_the_committed_model_and_constants_are_the_generators():
  text = (ROOT / hm.MODEL_XML).read_text()
  assert text.split("\n", 1)[1] == hm.body_xml(), "stale: uv run python -m pluggybot.humanoid.model"
  constants = json.loads((ROOT / hm.CONSTANTS_JSON).read_text())
  assert constants == json.loads(json.dumps(hm.training_constants())), \
    "stale: uv run python -m pluggybot.humanoid.model"


def test_the_humanoid_attaches_with_its_joint_ranges_in_radians():
  # The include form carries no <compiler>: parsed alone, MuJoCo read the
  # quadruped's ranges as degrees and its stand threw it 0.4 m up (#386).
  alone = mujoco.MjModel.from_xml_string(hm.body_xml())
  spec = mujoco.MjSpec()
  spec.attach(hm.attachable(), prefix="", frame=spec.worldbody.add_frame())
  attached = spec.compile()
  for name in ("left_knee", "right_hip_roll", "left_shoulder_pitch"):
    assert np.allclose(attached.jnt_range[attached.joint(name).id],
                       alone.jnt_range[alone.joint(name).id])
    assert np.allclose(alone.jnt_range[alone.joint(name).id], hm.joint_range(name))


def test_the_stand_puts_both_soles_flat_on_the_floor(standing):
  model, data = standing
  for s in hm.SIDES:
    g = model.geom(f"{s}_sole").id
    bottom = data.geom_xpos[g][2] - model.geom_size[g][2]
    assert abs(bottom) < 1e-5, (s, bottom)
    up = data.geom_xmat[g].reshape(3, 3)[:, 2]
    assert up[2] > 1 - 1e-9, (s, up)


def test_every_joint_is_driven_by_its_parts_datasheet():
  """A joint's driver is its part's: the clip at the sheet's peak, the
  torque-speed line through its no-load speed on the pack's bus, its rotor
  reflected through its gearbox, and the friction its part is given."""
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  lim = JointLimits.of_motors([hm.CHOSEN.motor(j) for j in hm.JOINT_NAMES])
  for k, name in enumerate(hm.JOINT_NAMES):
    m = hm.CHOSEN.motor(name)
    act = model.actuator(name).id
    assert model.actuator_forcerange[act][1] == pytest.approx(m.peak_torque), name
    k_motor = model.actuator_gainprm[act][1]
    assert k_motor == pytest.approx(lim.bus_v / lim.noload[k]), name
    j = model.joint(name).id
    assert model.dof_armature[model.jnt_dofadr[j]] == pytest.approx(
      m.rotor_inertia * m.ratio ** 2, rel=1e-5), name
    assert model.dof_frictionloss[model.jnt_dofadr[j]] == pytest.approx(
      hp.friction_nm(m), rel=1e-5), name


def test_the_drivers_are_the_pd_inside_each_joints_envelope():
  # The quadruped's algebra (#385), joint by joint: every joint its own
  # part's line and its own gains. At any state inside its no-load speed a
  # driver's force is its clipped PD.
  from pluggybot.legs.drivers import Drivers
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  drivers = Drivers(model, data, joints=hm.JOINT_NAMES)
  gains = np.array([driver_gains(hm.CHOSEN.motor(j)) for j in hm.JOINT_NAMES])
  kp, kd = gains[:, 0] * 30, gains[:, 1] * 3          # stiff enough to clip
  lim = JointLimits.of_motors([hm.CHOSEN.motor(j) for j in hm.JOINT_NAMES])
  rng = np.random.default_rng(0)
  qadr, vadr = drivers.qadr, np.array(
    [model.jnt_dofadr[model.joint(j).id] for j in hm.JOINT_NAMES])
  clipped = 0
  for _ in range(200):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[qadr] += rng.normal(0, 0.3, len(qadr))
    data.qvel[vadr] = rng.normal(0, 0.3, len(vadr)) * lim.noload
    target = data.qpos[qadr] + rng.normal(0, 0.5, len(qadr))
    drivers.pd(target, kp, kd)
    mujoco.mj_forward(model, data)
    q, qd = data.qpos[qadr], data.qvel[vadr]
    want = lim.clip(kp * (target - q) - kd * qd, qd)
    inside = np.abs(qd) < lim.noload
    assert np.allclose(data.actuator_force[drivers.act][inside], want[inside],
                       rtol=0, atol=1e-8)
    clipped += int((np.abs(want) >= lim.peak - 1e-9).sum())
  assert clipped > 100                     # the peaks were exercised


def test_a_motor_weighs_its_datasheet_with_its_bracket():
  spec = mujoco.MjSpec.from_string(hm.body_xml())
  geoms = {g.name: g for g in spec.geoms}
  for name in hm.JOINT_NAMES:
    m = hm.CHOSEN.motor(name)
    g = geoms[f"{name}_motor"]
    assert g.mass == pytest.approx(m.mass + hm._bracket(m).kg, rel=1e-5), name
    assert g.size[0] == pytest.approx(m.diameter / 2), name


def test_the_soles_and_the_pads_grip_with_their_own_friction():
  # Friction combines as the pair's MAX unless the geom claims priority
  # (SimNotes, "THE caster lesson"): a sole's rubber, not the floor's 1.0.
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  for _ in range(20):
    mujoco.mj_step(model, data)
  soles = {model.geom(f"{s}_sole").id for s in hm.SIDES}
  touching = [c for c in data.contact[:data.ncon] if {c.geom1, c.geom2} & soles]
  assert touching
  sole_mu = model.geom_friction[next(iter(soles))][0]
  assert sole_mu < 1.0
  for c in touching:
    assert c.friction[0] == pytest.approx(sole_mu)
  for s in hm.SIDES:
    for jaw in hm.JAWS:
      assert model.geom_priority[model.geom(f"{s}_{jaw}_pad").id] == 1


def _gripper_rig(block: float):
  """The left hand fixed where its arm hangs at zero -- the arm's joints
  taken out, the pelvis welded to a hook -- and a block between its jaws
  fixed to the world, `block` m across."""
  arm = {f"left_{j}" for j in hm.ARM}

  def rig(with_block=None):
    spec = mujoco.MjSpec.from_string(hm.body_xml())
    for act in list(spec.actuators):
      if act.name in arm:
        spec.delete(act)
    for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):
      for joint in list(body.joints):
        if joint.name in arm:
          spec.delete(joint)
    for key in list(spec.keys):
      spec.delete(key)
    spec.worldbody.add_body(name="hook", pos=[0.0, 0.0, hm.stand_height()])
    spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=hm.ROBOT_ROOT, name2="hook",
                      objtype=mujoco.mjtObj.mjOBJ_BODY,
                      data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    if with_block is not None:
      pos, quat = with_block
      held = spec.worldbody.add_body(name="held", pos=list(pos), quat=list(quat))
      held.add_geom(name="held", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[block / 2, 0.01, 0.01])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data

  model, data = rig()
  site = model.site("left_grip").id
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, data.site_xmat[site])
  return rig((data.site_xpos[site].copy(), quat))


def test_the_gripper_squeezes_with_its_datasheets_force_and_says_so():
  """Shut on a block it cannot close past, the gripper presses each pad
  with the force its sheet gives, and reports the gap and the squeeze off
  its own encoder and torque estimate."""
  g = hp.GRIPPER
  block = 0.04
  model, data = _gripper_rig(block)
  jaws = [model.jnt_qposadr[model.joint(f"left_{j}").id] for j in hm.JAWS]
  data.qpos[jaws] = g.stroke / 2
  data.ctrl[model.actuator("left_gripper").id] = 0.0   # shut
  for _ in range(int(1.5 / model.opt.timestep)):
    mujoco.mj_step(model, data)
  pads = {model.geom(f"left_{j}_pad").id for j in hm.JAWS}
  held = model.geom("held").id
  normal = []
  for i, c in enumerate(data.contact[:data.ncon]):
    pair = {int(c.geom1), int(c.geom2)}
    if held in pair and pair & pads:
      f = np.zeros(6)
      mujoco.mj_contactForce(model, data, i, f)
      normal.append(f[0])
  assert normal, "the jaws never touched the block"
  assert sum(normal) / 2 == pytest.approx(g.grip_force_n, rel=0.05)
  gap, squeeze = GripperSense(model, "left").read(data)
  assert gap == pytest.approx(block, abs=0.002)
  assert squeeze == pytest.approx(g.grip_force_n, rel=0.05)


def test_a_joint_reads_in_its_drivers_counts(standing):
  model, data = standing
  sense = JointSense(model)
  q, qd, tau = sense.read(data)
  assert np.allclose(q / sense.pos_lsb, np.round(q / sense.pos_lsb))
  assert np.allclose(tau / sense.tau_lsb, np.round(tau / sense.tau_lsb))
  truth = data.qpos[sense.qadr]
  assert np.all(np.abs(q - truth) <= sense.backlash / 2 + sense.pos_lsb)
  # ...and two robots never draw one stream.
  a, b = JointSense(model, key="a"), JointSense(model, key="b")
  assert not np.allclose(a.read(data)[2], b.read(data)[2])


# ---- the bill (Parts.md, "The humanoid body") ------------------------------------------

def test_the_bill_in_parts_md_is_the_data_and_every_entry_is_honest():
  """The catalog's rules on the humanoid's bill: rendered from its data, every
  part and line honest, and every `expect` -- a sheet's number that IS the
  sim's -- agreeing with the model as generated."""
  from pluggybot.humanoid import bill
  fixture = bill.build()
  assert bill.problems(fixture) == {}
  doc = (ROOT / "docs" / "Parts.md").read_text()
  assert bill.with_bill(doc, fixture) == doc, "stale bill: uv run python -m pluggybot.humanoid.bill"


def test_a_moved_datasheet_number_is_caught():
  """The pin bites both ways: every RS04 driver's clip moved off the sheet's
  120 N*m fails the bill's `expect`, and one moved alone is refused as
  drivers come apart."""
  from pluggybot.humanoid import bill
  from pluggybot.rack import catalog
  rs04 = next(p for p in bill.PARTS if p.id == "rs04")
  joints = bill.driven_by(hp.RS04)

  def moved(names):
    spec = mujoco.MjSpec.from_string(hm.body_xml())
    for act in spec.actuators:
      if act.name in names:
        act.forcerange = [-110.0, 110.0]
    return spec

  assert catalog.mismatches({"parts": [rs04.as_dict(mujoco.MjSpec.from_string(hm.body_xml()))]}) == []
  assert catalog.mismatches({"parts": [rs04.as_dict(moved(joints))]})
  with pytest.raises(ValueError, match="asymmetric"):
    rs04.as_dict(moved(joints[:1]))


def test_one_part_number_is_one_set_of_facts_across_both_bills():
  from pluggybot.humanoid import bill
  from pluggybot.rack import catalog
  quad = {p.partNumber: p for p in catalog.PARTS if p.partNumber}
  shared = [p for p in bill.PARTS if p.partNumber in quad]
  assert len(shared) > 10
  for p in shared:
    q = quad[p.partNumber]
    assert (p.priceEur, p.massG) == (q.priceEur, q.massG), p.partNumber


def test_the_bill_buys_every_actuator_the_model_drives():
  from collections import Counter
  from pluggybot.humanoid import bill
  wanted = Counter(hm.CHOSEN.motor(j).name.split()[-1] for j in hm.JOINT_NAMES)
  bought = Counter()
  for ln in bill.LINES:
    if ln.group == "actuation" and ln.part.startswith("rs"):
      bought[ln.part.upper()] += ln.quantity
  assert bought == wanted
  grippers = sum(ln.quantity for ln in bill.LINES if ln.part == "dex1_1")
  assert grippers == len(hm.SIDES)


# ---- the scene: `body` -----------------------------------------------------------------

def test_the_gantry_holds_the_body_where_it_hangs_while_its_legs_swing():
  """`--view body`'s hold: the pelvis welded to a hook at its own pose,
  the weld's data written. Left at MjSpec's default the weld's anchor sat
  a metre off and the body sagged 42 mm as its thighs swung forward."""
  import sys
  sys.path.insert(0, str(ROOT / "scripts"))
  import humanoid_spike as sp
  model, data, drivers = sp.held_world()
  scene = sp.BodyScene(model, data, drivers)
  root = model.body(hm.ROBOT_ROOT).id
  start = data.xpos[root].copy()
  worst = 0.0
  while scene.label != "hip roll: legs out":          # through the thighs' swings
    scene.step()
    worst = max(worst, float(np.linalg.norm(data.xpos[root] - start)))
  assert worst < 0.005, worst


# ---- stage 2: standing, reaching, grasping ---------------------------------------------

def _standing(scenery=None):
  """The robot on a bare floor at the stand, its standing policy and its
  reach, and the world's model and data."""
  from pluggybot.humanoid.policy import StandDriver
  from pluggybot.humanoid.reach import Reach
  spec = mujoco.MjSpec.from_string(hm.body_xml())
  if scenery is not None:
    scenery(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  stand = StandDriver(model, data)
  return model, data, stand, Reach(model, data, stand)


def _fly(model, data, stand, reach, seconds):
  for _ in range(int(round(seconds / model.opt.timestep))):
    stand.step()
    reach.step()
    mujoco.mj_step(model, data)


def test_the_standing_policy_drives_the_legs_off_what_the_robot_senses():
  """The exported file says what the policy reads and drives; the driver
  builds exactly those terms, from the robot's senses, and drives the legs
  alone (the arms are the reach's)."""
  from pluggybot.humanoid.policy import StandPolicy
  p = StandPolicy()
  assert p.observation_names == ("base_ang_vel", "projected_gravity", "joint_pos",
                                 "joint_vel", "actions", "posture")
  assert p.action_joints == tuple(j for j in hm.JOINT_NAMES if kind_is_leg(j))
  assert p.mean.shape == (3 + 3 + 24 + 24 + 12 + 2,)
  model, data, stand, _ = _standing()
  assert stand.observation().shape == p.mean.shape


def kind_is_leg(joint: str) -> bool:
  return hm.kind(joint) in hm.LEG


def test_the_standing_policy_stands_in_our_physics():
  """Flown in the served sim's physics (2 ms, its drivers in C), from its
  own senses: the pelvis held at the stand and the feet where they were."""
  model, data, stand, reach = _standing()
  root = model.body(hm.ROBOT_ROOT).id
  soles = [model.geom(f"{s}_sole").id for s in hm.SIDES]
  feet = data.geom_xpos[soles].copy()
  _fly(model, data, stand, reach, 3.0)
  assert abs(data.xpos[root][2] - hm.stand_height()) < 0.03, data.xpos[root][2]
  assert np.abs(data.geom_xpos[soles] - feet).max() < 0.01


class _Spy:
  """MjData as the robot's code meets it, every attribute it reads noted."""

  def __init__(self, data):
    object.__setattr__(self, "_data", data)
    object.__setattr__(self, "read", set())

  def __getattr__(self, name):
    self.read.add(name)
    return getattr(self._data, name)

  def __setattr__(self, name, value):
    setattr(self._data, name, value)


#: What the simulator knows and a robot does not: where its bodies truly are
#: and what touches what (Ben, 2026-10-15: never steer off them).
TRUTH = {"contact", "ncon", "xpos", "xquat", "geom_xpos", "geom_xmat", "site_xpos",
         "site_xmat", "xipos", "subtree_com", "cfrc_ext", "cfrc_int", "xfrc_applied",
         "efc_force", "qacc", "cvel", "xmat"}


def test_the_robot_steers_off_its_senses_never_the_simulators_truth():
  """Standing and reaching to a counter's height, the policy, the reach and
  the grippers read the drivers' replies, the IMU's sensors and their own
  commands -- nothing of where the world says the bodies are or what they
  touch. The boot's orientation is the one exception, read once at
  construction (the robot is started knowing it, as the quadruped is)."""
  from pluggybot.humanoid.arms import FORWARD, HandGoal
  from pluggybot.humanoid.policy import StandDriver
  from pluggybot.humanoid.reach import Reach
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  spy = _Spy(data)
  stand = StandDriver(model, spy)
  reach = Reach(model, spy, stand)
  spy.read.clear()
  assert reach.reach("left", HandGoal(np.array([0.45, 0.2, 0.94]), FORWARD))
  reach.grippers["left"].open()
  for _ in range(int(1.5 / model.opt.timestep)):
    stand.step()
    reach.step()
    mujoco.mj_step(model, data)
  reach.grippers["left"].shut()
  reach.grippers["left"].held()
  assert spy.read, "nothing was read"
  assert not spy.read & TRUTH, spy.read & TRUTH


def test_a_reach_solved_in_one_jump_from_a_hanging_arm_misses_and_the_plan_finds_it():
  """⚠ From the arm hanging, the counter's height 0.45 m ahead solved in
  one jump lands in the wrong basin (the shoulder thrown back): the premise.
  The plan's solve starts from several arms and reaches it standing."""
  from pluggybot.humanoid.arms import FORWARD, HandGoal, SelfModel
  from pluggybot.humanoid.reach import plan
  sm = SelfModel()
  sm.pose(np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES]),
          np.array([0.0, 0.0, -1.0]))
  goal = HandGoal(np.array([0.45, 0.2, 0.94]), FORWARD)
  _, miss, _ = sm.solve("left", goal.pos, goal.rot)
  assert miss > 0.05, miss
  found = plan("left", goal)
  assert found is not None and found[0] == (0.0, 0.0)


def test_the_floor_is_reached_only_close_in_from_the_deepest_bow():
  """The envelope's low edge, on the robot's model of itself: 0.09 m up,
  fingers down, 0.18 m ahead is reached from the deepest posture the
  standing policy holds, and 0.45 m ahead from none."""
  from pluggybot.humanoid.arms import DOWN, HandGoal
  from pluggybot.humanoid.reach import POSTURES, plan
  near = plan("left", HandGoal(np.array([0.18, 0.2, 0.09]), DOWN))
  assert near is not None and near[0] == POSTURES[-1]
  assert plan("left", HandGoal(np.array([0.45, 0.2, 0.09]), DOWN)) is None


def test_a_grip_is_judged_by_the_grippers_own_reading():
  """`Gripper.held`: shut on a block it stalls short of closed, pressing --
  held; shut on nothing it closes -- not held. Off its encoder and its
  current, never the contacts, sampled every physics step as the reach
  samples it (`Gripper.sample`)."""
  from pluggybot.humanoid.arms import Gripper
  model, data = _gripper_rig(0.04)
  g = Gripper(model, data, "left")
  jaws = [model.jnt_qposadr[model.joint(f"left_{j}").id] for j in hm.JAWS]
  data.qpos[jaws] = hp.GRIPPER.stroke / 2
  g.shut()
  for _ in range(int(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
    g.sample()
  assert g.held()
  model, data = _gripper_rig(0.04)
  model.geom_contype[model.geom("held").id] = 0
  model.geom_conaffinity[model.geom("held").id] = 0
  g = Gripper(model, data, "left")
  data.qpos[jaws] = hp.GRIPPER.stroke / 2
  g.shut()
  for _ in range(int(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
    g.sample()
  assert not g.held()


def test_the_force_at_the_hand_is_read_off_the_arms_motor_currents():
  """The hand held out at a counter's height, its arm on the reach's
  commands (gains and its weight fed forward), a known push put on the
  grip: what the robot reads off its six drivers' torques
  (`SelfModel.hand_force`) is that push, to within the drivers' counts,
  their noise and the joints' friction."""
  from pluggybot.humanoid.arms import FORWARD, HandGoal
  model, data, stand, reach = _standing()
  assert reach.reach("left", HandGoal(np.array([0.42, 0.2, 0.9]), FORWARD))
  _fly(model, data, stand, reach, 3.0)
  assert reach.done("left")
  hand = model.body("left_hand").id
  site = model.site("left_grip").id
  push = np.array([-20.0, 0.0, 0.0])            # the hand pulled back, 20 N
  felt = []
  for k in range(int(1.5 / model.opt.timestep)):
    data.xfrc_applied[hand] = 0.0
    mujoco.mj_applyFT(model, data, -push, np.zeros(3), data.site_xpos[site], hand,
                      data.qfrc_applied)
    stand.step()
    reach.step()
    mujoco.mj_step(model, data)
    data.qfrc_applied[:] = 0.0
    if k > int(1.0 / model.opt.timestep):
      felt.append(reach.force["left"].copy())
  felt = np.mean(felt, axis=0)
  # The world pushes the hand with -push; the hand pushes back with +push.
  assert np.abs(felt - push).max() < 1.5, felt


def test_a_hand_that_meets_something_on_its_way_stops_short():
  """⚠ A box between the hand and its goal: the reach stops where its arm's
  currents read the push (`reach.BLOCKED_N`, held `BLOCKED_S`) -- unstopped,
  a gripper driven into a counter's edge pushed the robot over backwards."""
  from pluggybot.humanoid.arms import FORWARD, HandGoal

  def box(spec):
    spec.worldbody.add_geom(name="in_the_way", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[0.05, 0.3, 0.3], pos=[0.42, 0.2, 0.9])
  model, data, stand, reach = _standing(box)
  root = model.body(hm.ROBOT_ROOT).id
  assert reach.reach("left", HandGoal(np.array([0.55, 0.2, 0.94]), FORWARD))
  _fly(model, data, stand, reach, 4.0)
  assert reach.phase["left"] == "blocked"
  # Standing in the posture its plan asked (a crouch), upright: not fallen.
  assert data.xpos[root][2] > 0.45 and data.xmat[root][8] > 0.9


def test_a_hand_goes_in_only_once_the_trunk_is_still():
  """⚠ The force read off the arm's currents assumes the arm still: begun
  as the deepest bow stopped turning, the floor's approach read 492 N at the
  hand and stopped 6 cm short of it (#520). The posture there, the approach
  waits for the trunk's gyro to read it still (`reach.TRUNK_STILL_RATE`)."""
  model, data, stand, reach = _standing()
  reach._pose()
  reach.posture = reach.posture_now()                       # the plan's posture: there
  stand.trunk.rate = np.array([0.0, 0.5, 0.0])              # ...and still turning into it
  assert not reach._posture_there(0.01)
  stand.trunk.rate = np.zeros(3)
  assert reach._posture_there(0.01)
