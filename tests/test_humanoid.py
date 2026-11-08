"""The humanoid (issue #507, stages 1 and 2): our own body from real parts,
its model, drivers and sensors; and standing, reaching and gripping off its
own senses. Each rule pinned as cheaply as it fails for the
right reason: the generator against its committed files, a joint's driver
against its part's datasheet, a motor's mass against its sheet, the soles
and the pads against MuJoCo's friction MAX, the gripper's squeeze against
its sheet's force, and what the drivers report against their fields.
SimNotes and Parts.md, "The humanoid body", carry the design."""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.humanoid.senses import GripperSense, JointSense
from pluggybot.humanoid.actuator import JointLimits, driver_gains

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
  from pluggybot.humanoid.drivers import Drivers
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


def _hanging_hand():
  """The left hand fixed where its arm hangs at zero -- the arm's joints
  taken out, the pelvis welded to a hook: its spec, and its grip site's
  place and frame (x the jaws' travel, the fingers down its -z)."""
  arm = {f"left_{j}" for j in hm.ARM}
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
  model, data = _compiled(spec)
  site = model.site("left_grip").id
  return spec, data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()


def _compiled(spec):
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def _quat(mat) -> list:
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, np.asarray(mat).flatten())
  return list(quat)


def _gripper_rig(block: float):
  """The hanging hand (`_hanging_hand`) and a block between its jaws fixed
  to the world, `block` m across."""
  spec, pos, mat = _hanging_hand()
  held = spec.worldbody.add_body(name="held", pos=list(pos), quat=_quat(mat))
  held.add_geom(name="held", type=mujoco.mjtGeom.mjGEOM_BOX, size=[block / 2, 0.01, 0.01])
  return _compiled(spec)


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


def test_a_part_added_redeals_no_other_parts_noise():
  """Each joint reads the same noise whether or not the body has the waist
  (#560's stage (c3), `senses.SHARED`): dealt one stream over every joint in
  order, the waist re-drew every other joint's noise -- the one realization
  all of a check's set-outs replay -- and today's body placed the cube 9
  times of 32 dealt so, 22 as served. The waist's own reading is noisy."""
  read = {}
  for spec in (hm.HumanoidSpec(waist_yaw=False), hm.CHOSEN):
    model = mujoco.MjModel.from_xml_string(hm.body_xml(spec))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)
    sense = JointSense(model, spec)
    reads = [sense.read(data) for _ in range(4)]
    read[spec.waist_yaw] = {j: [(r[0][i], r[2][i]) for r in reads]
                            for i, j in enumerate(sense.names)}
  assert set(read[True]) - set(read[False]) == set(hm.WAIST)
  for joint, got in read[False].items():
    assert read[True][joint] == got, joint
  assert len(set(read[True][hm.WAIST[0]])) > 1


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
  from pluggybot.humanoid import catalog
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
  from pluggybot.humanoid import catalog
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
  assert p.mean.shape == (3 + 3 + 2 * len(hm.JOINT_NAMES) + 12 + 2,)
  model, data, stand, _ = _standing()
  assert stand.observation().shape == p.mean.shape


def kind_is_leg(joint: str) -> bool:
  return hm.kind(joint) in hm.LEG


def test_running_the_policies_imports_no_training_stack():
  # The serving image installs deploy/requirements-serve.txt and nothing
  # else; a policy that needed torch or onnx to run would crash there. Its
  # three policies -- standing, walking, getting up -- run in numpy.
  import subprocess
  import sys
  code = ("import sys, pluggybot.humanoid.model, pluggybot.humanoid.policy, "
          "pluggybot.humanoid.walk, pluggybot.humanoid.getup; "
          "bad = [m for m in ('torch', 'jax', 'warp', 'onnx', 'mjlab') "
          "if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
  out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
  assert out.returncode == 0, out.stdout + out.stderr


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
  # ...the hand's force read off the whole body as it goes in (`felt`)
  reach.standing = True
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
  assert found is not None and found[0] == (0.0, 0.0, 0.0)


def test_the_floor_is_reached_close_in_from_the_deepest_bow_and_further_turned():
  """The envelope's low edge, on the robot's model of itself: 0.09 m up,
  fingers down. The waist held home, 0.18 m ahead is reached from the
  deepest posture the standing policy holds, and 0.45 m ahead from none --
  turned at the waist (#560's stage (c3); #620 measured 0.15-0.45 m), 0.45
  is reached too, and 0.65 from none."""
  from pluggybot.humanoid.arms import DOWN, HandGoal
  from pluggybot.humanoid.reach import POSTURES, plan, with_waist
  home = with_waist(POSTURES, yaws=(0.0,))
  near = plan("left", HandGoal(np.array([0.18, 0.2, 0.09]), DOWN), postures=home)
  assert near is not None and near[0] == (*POSTURES[-1], 0.0)
  assert plan("left", HandGoal(np.array([0.45, 0.2, 0.09]), DOWN), postures=home) is None
  far = plan("left", HandGoal(np.array([0.45, 0.2, 0.09]), DOWN))
  assert far is not None and far[0][2] != 0.0
  assert plan("left", HandGoal(np.array([0.65, 0.2, 0.09]), DOWN)) is None


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


def test_a_jaw_takes_at_least_a_step_to_reach_its_drives_line():
  """The rule under the gripper's drive (#628): a jaw's mechanical time
  constant on its drive's line -- its inertia (its rotor's share and its
  finger) over the line's back-EMF, K^2/R -- is at least a physics step.
  MuJoCo integrates a drive explicitly while it is clipped, a shut's whole
  travel: at 0.13 ms (0.2 kg a jaw) its force flipped between its clips
  every 2 ms step."""
  from pluggybot.humanoid import drivers as dv
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  for s in hm.SIDES:
    g = model.actuator_gainprm[model.actuator(f"{s}_gripper").id]
    r, k = g[dv.R], g[dv.K]
    for jaw in hm.JAWS:
      j = model.joint(f"{s}_{jaw}")
      inertia = model.dof_armature[j.dofadr[0]] + model.body_mass[j.bodyid[0]]
      assert inertia * r / k ** 2 >= model.opt.timestep, (s, jaw)


def _shut_from_open(model, data, force: float, seconds: float,
                    width: float = hp.GRIPPER.stroke):
  """The left gripper set out `width` open and held so, then shut at `force`
  for `seconds`: its gripper, and each step's drive force, jaw speed, gap
  and word (`held`)."""
  from pluggybot.humanoid.arms import Gripper
  mujoco.mj_resetData(model, data)
  jaws = [model.jnt_qposadr[model.joint(f"left_{j}").id] for j in hm.JAWS]
  dof = model.jnt_dofadr[model.joint(f"left_{hm.JAWS[0]}").id]
  data.qpos[jaws] = width / 2
  data.ctrl[model.actuator("left_gripper").id] = width / 2
  mujoco.mj_forward(model, data)
  g = Gripper(model, data, "left")
  g.open(width)
  for _ in range(int(0.2 / model.opt.timestep)):
    mujoco.mj_step(model, data)
    g.sample()
  g.shut(force)
  rows = []
  for _ in range(int(round(seconds / model.opt.timestep))):
    mujoco.mj_step(model, data)
    g.sample()
    rows.append((data.actuator_force[g.act], data.qvel[dof], data.qpos[jaws].sum(), g.held()))
  return g, np.array(rows)


def test_the_grippers_drive_keeps_to_its_line_in_free_air():
  """Shut on nothing at the least squeeze, a middling one and the most, its
  drive's force never flips from past one clip's quarter to past the
  other's between two steps, its jaws never outrun the line's top speed,
  and they shut within the grip's wait (`body.GRIP_S`). #628: with jaws too
  light for the line (above) the force flipped between its clips every
  step, the jaws ran at up to 8x the line's speed, and at 120 N shut in
  0.13 s where the motor needs 0.33."""
  from pluggybot.humanoid import drivers as dv
  from pluggybot.humanoid.body import GRIP_S
  spec, _, _ = _hanging_hand()
  model, data = _compiled(spec)
  g = model.actuator_gainprm[model.actuator("left_gripper").id]
  top = g[dv.VMAX] / g[dv.K]                      # the line's no-load speed: bus / K
  for force in (hp.GRIPPER.grip_force_n, 40.0, 5.0):
    _, rows = _shut_from_open(model, data, force, GRIP_S)
    f, v, gap = rows.T[:3]
    big = np.abs(f) > 0.5 * force                 # past a quarter of the clip, 2 * force
    flips = (np.sign(f[1:]) != np.sign(f[:-1])) & big[1:] & big[:-1]
    assert not flips.any(), (force, int(flips.sum()))
    assert np.abs(v).max() <= 1.05 * top, (force, np.abs(v).max() / top)
    assert gap[-1] < 0.001, (force, gap[-1])


def test_a_shut_closes_on_a_light_handle_off_the_jaws_middle_at_every_squeeze():
  """A light handle -- the table fork's, 24 mm across, 60 g -- lying free on
  a ledge under the fingertips, 8 mm off the jaws' middle: shut at 40 N and
  up, the jaws close on it and squeeze it as asked, as they do under 40
  (#628: in the table's set-out the drive's chatter batted the fork's handle
  from 40 N and the jaws stood 4-8 cm apart, nothing held)."""
  half = (0.012, 0.03, 0.011)        # across the jaws, along the pads, up
  spec, pos, mat = _hanging_hand()
  tip = hp.GRIPPER.finger_len / 2    # the pads reach this far under the grip
  ledge = spec.worldbody.add_body(name="ledge", quat=_quat(mat),
                                  pos=list(pos - mat[:, 2] * (tip + 0.007)))
  ledge.add_geom(name="ledge", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.07, 0.04, 0.005],
                 friction=[0.7, 0.005, 0.0001])
  handle = spec.worldbody.add_body(
    name="handle", quat=_quat(mat),
    pos=list(pos + 0.008 * mat[:, 0] - mat[:, 2] * (tip + 0.002 - half[2])))
  handle.add_freejoint()
  handle.add_geom(name="handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), mass=0.06,
                  friction=[0.7, 0.005, 0.0001])
  from pluggybot.humanoid.body import GRIP_S
  model, data = _compiled(spec)
  for force in (40.0, 60.0, 80.0, hp.GRIPPER.grip_force_n):
    g, _ = _shut_from_open(model, data, force, GRIP_S)
    gap, squeeze = g.read()
    assert g.held(), (force, gap, squeeze)
    assert gap == pytest.approx(2 * half[0], abs=0.001), force
    assert squeeze == pytest.approx(force, rel=0.1), force


def test_a_hard_squeeze_on_something_thin_reads_held():
  """Shut at 40-120 N on a block 6-15 mm across -- the plate's 8 mm wall,
  the door's 10 mm lever arm -- the jaws stall on it squeezing what the
  gripper's own command says its drive pushes there (`arms.pushes`: under 2
  cm at 120 N its spring's, short of the force asked), and the grip reads
  held by the time its verb first reads it (`body.GRIP_S`). #646: judged
  against half the force asked, a 1 cm block read nothing held at 120 N, a
  6 mm one from 80."""
  from pluggybot.humanoid.arms import pushes
  from pluggybot.humanoid.body import GRIP_S
  for block in (0.006, 0.008, 0.010, 0.015):
    model, data = _gripper_rig(block)
    first_read = int(round(GRIP_S / model.opt.timestep)) - 1
    for force in (40.0, 80.0, hp.GRIPPER.grip_force_n):
      g, rows = _shut_from_open(model, data, force, 1.0)
      gap, squeeze = g.read()
      assert gap == pytest.approx(block, abs=0.0005), (block, force)
      assert squeeze == pytest.approx(pushes(gap, force), abs=2.0), (block, force)
      assert rows[first_read, 3], (block, force, "not held when its verb first read it")
      assert g.held(), (block, force, gap, squeeze)


def test_a_shut_on_nothing_never_reads_held_as_its_jaws_pass_a_thin_gap():
  """Shut on nothing, sampled every physics step through the longest a
  grip's word is waited for (`body.GRIP_S` and `GRIP_SETTLE_S`), the gripper
  reads nothing held at any step its jaws are under 2 cm apart, closed too:
  where a hard squeeze's spring falls short of the force asked, `held` asks
  less of the squeeze (#646). From open, at every squeeze; and from 1-3 cm
  apart, the drive still speeding the jaws up as they cross those gaps, at
  the squeezes where it asks less."""
  from pluggybot.humanoid.body import GRIP_S, GRIP_SETTLE_S
  spec, _, _ = _hanging_hand()
  model, data = _compiled(spec)
  hardest = hp.GRIPPER.grip_force_n
  cases = [(hp.GRIPPER.stroke, f) for f in (5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, hardest)]
  cases += [(w, f) for w in (0.012, 0.020, 0.030) for f in (40.0, 80.0, hardest)]
  for width, force in cases:
    _, rows = _shut_from_open(model, data, force, GRIP_S + GRIP_SETTLE_S, width)
    gap, held = rows.T[2:]
    thin = gap < 0.02
    assert thin.sum() > 0.5 * len(gap) and not held[thin].any(), (width, force)


def test_a_grip_lost_under_a_hard_squeeze_reads_lost_when_its_squeeze_falls():
  """The block taken from between the jaws under 80 and 120 N, 26-35 mm
  across: the jaws run shut, and the word goes false within two physics
  steps of the squeeze's mean falling under half the force, as it did
  judged against the force. #646: judged at the gap as it read that step,
  the bar fell with the jaws and the word waited for them to read closed,
  up to 40 ms longer."""
  from pluggybot.humanoid.arms import HELD_GAP_M, HELD_SHARE
  for block in (0.026, 0.030, 0.035):
    for force in (80.0, hp.GRIPPER.grip_force_n):
      model, data = _gripper_rig(block)
      g, _ = _shut_from_open(model, data, force, 1.0)
      assert g.held(), (block, force)
      taken = model.geom("held").id
      model.geom_contype[taken] = model.geom_conaffinity[taken] = 0
      fell, gone = None, None
      for i in range(int(0.3 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        g.sample()
        gap, squeeze = g.read()
        if fell is None and (squeeze <= HELD_SHARE * force or gap <= HELD_GAP_M):
          fell = i
        if gone is None and not g.held():
          gone = i
      assert fell is not None and gone is not None and gone - fell <= 2, (block, force, fell, gone)


def test_a_restart_keeps_the_gap_a_grip_is_judged_at():
  """The gap's running mean decides a grip's word (#646), so it rides a
  restart with the rest of the gripper's reading (#345); a save from before
  it carries on from the gap it read."""
  from pluggybot.humanoid.policy import StandDriver
  from pluggybot.humanoid.reach import Reach
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  r = Reach(model, data, StandDriver(model, data))
  r.grippers["left"].gap_mean = 0.0123
  state = r.kept_state()
  again = Reach(model, data, StandDriver(model, data))
  again.restore_kept(state)
  assert again.grippers["left"].gap_mean == 0.0123
  del state["grippers"]["left"]["gapMean"]
  state["grippers"]["left"]["gap"] = 0.0456
  again.restore_kept(state)
  assert again.grippers["left"].gap_mean == 0.0456


def test_the_force_at_the_hand_is_read_off_the_arms_motor_currents():
  """The hand held out at a counter's height, its arm on the reach's
  commands (gains and its weight fed forward), a known push put on the
  grip: what the robot reads off its six drivers' torques
  (`SelfModel.hand_force`) is that push, to within the drivers' counts,
  their noise and the joints' friction."""
  from pluggybot.humanoid.arms import FORWARD, HandGoal
  model, data, stand, reach = _standing()
  assert reach.reach("left", HandGoal(np.array([0.42, 0.2, 0.9]), FORWARD))
  _fly(model, data, stand, reach, 6.0)
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
  # ⚠ What the joints' friction holds at a hold is a draw of the flight
  # there: held within 2 cm of here, the worst axis read 0.6-1.7 N off, on
  # stage 1's head and #581's stage 3's alike (36 holds), 1.5 inside it.
  assert np.abs(felt - push).max() < 2.0, felt


def test_a_hand_that_meets_something_on_its_way_stops_short():
  """⚠ A box between the hand and its goal: the reach stops where its arm's
  currents read the push (`reach.BLOCKED_N`, held `BLOCKED_S`) -- unstopped,
  a gripper driven into a counter's edge pushed the robot over backwards."""
  from pluggybot.humanoid.arms import FORWARD, HandGoal

  def box(spec):
    spec.worldbody.add_geom(name="in_the_way", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[0.05, 0.3, 0.3], pos=[0.42, 0.2, 0.9])
  from pluggybot.humanoid.reach import POSTURES, with_waist
  model, data, stand, reach = _standing(box)
  root = model.body(hm.ROBOT_ROOT).id
  # ...the waist home, as the scene was drawn: turned, the arm's way out to
  # its pre-grasp sweeps into the box, which no reach knows of
  assert reach.reach("left", HandGoal(np.array([0.55, 0.2, 0.94]), FORWARD),
                     postures=with_waist(POSTURES, yaws=(0.0,)))
  _fly(model, data, stand, reach, 8.0)
  assert reach.phase["left"] == "blocked"
  # Standing in the posture its plan asked, its trunk at that posture's
  # pitch: not fallen.
  tilt = math.acos(min(1.0, float(data.xmat[root][8])))
  assert data.xpos[root][2] > 0.45 and abs(tilt - reach.posture[1]) < 0.15, (tilt, reach.posture)


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


def test_the_heads_cameras_are_placed_through_the_waist_on_the_model_of_itself():
  """#560's stage (c3): the robot's model of itself sets the waist off its
  reading (`SelfModel.pose`), so every camera placed through it -- the
  colour imager the tags are read by, the navigation camera the places
  are, the depth camera -- and the charge port are where the world has
  them, in the pelvis's frame, the waist turned; placed with the waist at
  zero, they are 0.4 rad round."""
  from pluggybot.humanoid.arms import SelfModel
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  data.qpos[model.jnt_qposadr[model.joint("waist_yaw").id]] = 0.4
  mujoco.mj_forward(model, data)
  q = np.array([data.qpos[model.jnt_qposadr[model.joint(n).id]] for n in hm.JOINT_NAMES])
  root = model.body(hm.ROBOT_ROOT).id
  gravity = data.xmat[root].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

  def placed(m, d):
    r, at = d.xmat[m.body(hm.ROBOT_ROOT).id].reshape(3, 3), d.xpos[m.body(hm.ROBOT_ROOT).id]
    out = {c: (r.T @ (d.cam_xpos[m.camera(c).id] - at)) for c in ("color_eye", "nav_eye", "depth_eye")}
    out["charge_port"] = r.T @ (d.site_xpos[m.site("charge_port").id] - at)
    return out
  truly = placed(model, data)
  sm = SelfModel()
  for waist, near in ((0.4, True), (0.0, False)):
    q[hm.JOINT_NAMES.index("waist_yaw")] = waist
    sm.pose(q, gravity)
    mujoco.mj_camlight(sm.m, sm.d)
    for name, at in placed(sm.m, sm.d).items():
      assert bool(np.abs(at - truly[name]).max() < 1e-9) is near, (waist, name)


def test_a_posture_is_square_first_then_turned_the_least_bent_first():
  """#560's stage (c3): a posture is chosen with its waist angle -- the legs'
  postures by the waist's grid, as #620's were (Ben's call) -- every one with
  the waist square first, then the turned ones the least bent first by
  #620's own cost (`waist_spike.bent`): by that cost alone the served door
  and chest passed 23 and 14 of 32 draws, square first 28 and 25."""
  import sys
  from pathlib import Path
  from pluggybot.humanoid import reach as rc
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
  import waist_spike as ws
  assert rc.WAIST_YAWS == ws.WAIST_YAWS
  for legs, table in ((rc.POSTURES, rc.REACH_POSTURES), (rc.HOLD_POSTURES, rc.CARRY_POSTURES)):
    pairs = [(i, k) for i in range(len(legs)) for k in range(len(ws.WAIST_YAWS))]
    pairs.sort(key=lambda ik: (ws.WAIST_YAWS[ik[1]] != 0,
                               ws.bent(ik[0], {"waist_yaw": ws.WAIST_YAWS[ik[1]]})))
    assert table == tuple((*legs[i], ws.WAIST_YAWS[k]) for i, k in pairs)


def test_a_turned_trunk_folded_into_a_thigh_stands_nowhere():
  """At the hips' deepest bow a turn of 15-30 deg puts the torso's corner
  1-2.5 cm into a thigh (#620): the robot's drawing meets its own parts only
  in the pairs it names, so the planner rules those postures out
  (`reach.stands`), and no other of the reach's 72 or the carry's 99."""
  import math
  from pluggybot.humanoid import reach as rc
  from pluggybot.humanoid.arms import SelfModel
  sm = SelfModel()
  q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  folds = {(-0.25, 1.1, math.radians(a)) for a in (15, -15, 30, -30)}
  for table in (rc.REACH_POSTURES, rc.CARRY_POSTURES):
    assert {p for p in table if not rc.stands(sm, p, q0)} == folds


def test_no_arm_takes_its_grip_further_from_its_shoulder_than_its_reach():
  """A plan skips a posture whose goal is past the arm's reach of its
  shoulder (`reach.out_of_reach`): nothing lost only if no arm, however it
  is bent, takes the grip further than the reach it measures."""
  from pluggybot.humanoid import reach as rc
  from pluggybot.humanoid.arms import SelfModel
  sm = SelfModel()
  rng = np.random.default_rng(560)
  lo, hi = sm.ranges[:, 0], sm.ranges[:, 1]
  for side in hm.SIDES:
    most = rc.arm_reach(sm, side)
    shoulder = sm.m.body(f"{side}_shoulder_roll_link").id
    for _ in range(500):
      sm.pose(rng.uniform(lo, hi), np.array([0.0, 0.0, -1.0]))
      far = np.linalg.norm(sm.d.site_xpos[sm.grip[side]] - sm.d.xpos[shoulder])
      assert far <= most + 1e-9, (side, far, most)



def test_measuring_an_arms_reach_leaves_the_model_as_it_was_posed():
  """The plan's bound on an arm's reach (`reach.arm_reach`) is measured
  the first time a process asks for it, in the middle of judging a
  posture (`out_of_reach`, after `stands` posed its statue): measured on
  the model as posed, it left the statue with straight legs, and the first
  posture of every process's first plan a hand made was solved on that."""
  from pluggybot.humanoid import reach as rc
  from pluggybot.humanoid.arms import SelfModel
  sm = SelfModel()
  sm.statues.pop(("reach", "left"), None)
  q0 =np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  assert rc.stands(sm, rc.REACH_POSTURES[0], q0)
  posed = sm.d.qpos.copy(), sm.q.copy()
  rc.out_of_reach(sm, "left", (0.3, 0.2, 0.8))
  assert np.array_equal(sm.d.qpos, posed[0]) and np.array_equal(sm.q, posed[1])


def test_the_waist_is_held_damped_over_what_it_turns_inside_its_parts_range():
  """#560's stage (c3): the waist's commands (`arms.Waist`) are sized to the
  trunk above it -- at the arms' scale it rang at 2.7 Hz at a damping ratio
  of 0.27 with both arms out -- and inside the RS03's MIT range (its
  manual's 4.1.2: Kp 0-5000, Kd 0-100)."""
  import math
  from pluggybot.humanoid.arms import Waist
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  w = Waist(model, data)
  kp, kd = float(w.kp[0]), float(w.kd[0])
  assert 0.0 < kp <= 5000.0 and 0.0 < kd <= 100.0
  dof = model.jnt_dofadr[model.joint("waist_yaw").id]
  for out in (0.0, -1.4):                  # the arms hanging, both held out level
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for s in hm.SIDES:
      data.qpos[model.jnt_qposadr[model.joint(f"{s}_shoulder_pitch").id]] = out
    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full)
    inertia = full[dof, dof]
    assert kd / (2.0 * math.sqrt(kp * inertia)) >= 0.9, (out, inertia)


def test_the_waist_is_set_going_and_stopped_gently_and_arrives_exactly():
  """#560's stage (c3): the waist's command (`reach.turned`) turns no faster
  than `WAIST_RATE` and is set going and stopped at `WAIST_ACCEL`, as the
  stand's training turns it, arriving exactly on its goal at rest -- and a
  goal moved behind a turning command slows it, never stops it dead."""
  from pluggybot.humanoid import reach as rc
  dt = rc.EVERY_S
  for goal in (1.0472, -0.26, 0.001):
    at, speed, steps = np.zeros(1), np.zeros(1), 0
    while not (at[0] == goal and speed[0] == 0.0):
      before = speed.copy()
      at, speed = rc.turned(at, speed, np.array([goal]), dt)
      assert abs(speed[0]) <= rc.WAIST_RATE + 1e-12
      assert abs(speed[0] - before[0]) <= 2.0 * rc.WAIST_ACCEL * dt, (goal, steps)
      steps += 1
      assert steps < 1000
  at, speed = rc.turned(np.array([0.5]), np.array([0.6]), np.array([0.499]), dt)
  assert speed[0] == pytest.approx(0.6 - rc.WAIST_ACCEL * dt) and at[0] > 0.5


def test_training_turns_the_waist_as_the_served_reach_does():
  """#560's stage (c3): the stand is served the waist's turns it trained on
  -- its scripted waist's rate, its acceleration and how far either way it
  goes are the served reach's (`training/` cannot import this package, so
  its constants are read off its source)."""
  import ast
  from pathlib import Path
  from pluggybot.humanoid import reach as rc
  src = Path(__file__).resolve().parents[1] / "training" / "humanoid_train" / "stand.py"
  consts = {}
  for node in ast.parse(src.read_text()).body:
    if not isinstance(node, ast.Assign):
      continue
    try:
      value = ast.literal_eval(node.value)
    except ValueError:
      continue
    for t in node.targets:
      if isinstance(t, ast.Name):
        consts[t.id] = value
      elif isinstance(t, ast.Tuple):
        consts.update({n.id: v for n, v in zip(t.elts, value)})
  assert (consts["WAIST_RATE"], consts["WAIST_ACCEL"]) == (rc.WAIST_RATE, rc.WAIST_ACCEL)
  assert consts["WAIST_GOAL"] == pytest.approx(float(np.max(rc.WAIST_YAWS)), abs=1e-4)



def test_the_served_stand_was_trained_at_the_gains_the_body_commands():
  """#560's stage (c3): the standing policy is served beside the arms and
  the waist it was trained beside -- their commands' gains the ones its
  export recorded (`quad_train.export`), and every joint it saw a joint of
  the body."""
  from pluggybot.humanoid.policy import StandPolicy
  policy = StandPolicy()
  assert tuple(policy.training["arm_gain_scale"]) == hm.ARM_GAIN_SCALE
  assert tuple(policy.training["waist_gain_scale"]) == hm.WAIST_GAIN_SCALE
  assert set(hm.WAIST) <= set(policy.joint_names)
