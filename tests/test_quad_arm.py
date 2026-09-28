"""The arm on the served quadruped (issue #405, step 4a): each rule the body's
arm rests on, pinned as cheaply as it can be while still failing for the
right reason -- the arm in the chosen body, its driver held on every step
and reading `qpos` alone, the fold on a fall and at rest, its joints as a
program's axes, and the pack's bill for its two motors. SimNotes, "The arm
on the served body"."""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot.legs import arm as am
from pluggybot.legs import body as qb
from pluggybot.legs import model as qm
from pluggybot.legs import world as lw
from pluggybot.legs.actuator import JointLimits
from pluggybot.lifecycle import QUAD_HOME, world_config, world_facts
from pluggybot.mind import overseer as ov
from pluggybot.procedure import axes, lang
from pluggybot.procedure import steps as st


@pytest.fixture(scope="module")
def quad_world():
  """The home world with one quadruped and its dock, compiled once."""
  return lw.home_spec().compile()


def _quad(model):
  body = qb.QuadBody(model, mujoco.MjData(model), realtime=False,
                     grid_bounds=world_config(QUAD_HOME)["grid_bounds"])
  body.start_at(1.5, 0.5, 0.0)
  return body


STOW = qm.CHOSEN.arm.stow


def _standalone():
  model = mujoco.MjModel.from_xml_string(qm.body_xml(qm.CHOSEN))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return model, data


# ---- the arm is the body's --------------------------------------------------------


def test_the_chosen_body_carries_the_arm_and_the_placeholder_is_the_sizing_premise():
  assert qm.CHOSEN.arm == am.ArmSpec() and qm.SIZING.arm is None
  assert qm.CHOSEN.masses.arm == qm.CHOSEN.masses.tool == 0.0
  model, _ = _standalone()
  names = {model.actuator(i).name for i in range(model.nu)}
  assert {"arm_shoulder", "arm_elbow"} <= names
  # the legs stay qpos[7:19], and every keyframe carries the arm stowed
  assert model.jnt_qposadr[model.joint("arm_shoulder").id] == 19
  for key in range(model.nkey):
    assert model.key_qpos[key][19:21] == pytest.approx(STOW)
  with pytest.raises(ValueError, match="arm_reach"):
    qm.body_xml(qm.CHOSEN.with_(arm_reach=True))


# ---- the driver --------------------------------------------------------------------


def _jacobian_gravity(model, data, arm):
  """What the motors hold, off MuJoCo's own Jacobians (#378's driver): each
  link's weight and the payload, split as the linkage splits it."""
  jac = np.zeros((3, model.nv))
  gen = np.zeros(model.nv)
  for b in arm.bodies:
    mujoco.mj_jacBodyCom(model, data, jac, None, b)
    gen += jac[2] * model.body_mass[b] * am.G
  kg, (ox, oz) = arm.payload
  if kg:
    sid = model.site("arm_seat").id
    point = data.site_xpos[sid] + data.site_xmat[sid].reshape(3, 3) @ np.array([ox, 0.0, oz])
    mujoco.mj_jac(model, data, jac, None, point, model.site_bodyid[sid])
    gen += jac[2] * kg * am.G
  gs, ge, gw = (gen[arm.vadr[n]] for n in ("shoulder", "elbow", "wrist"))
  return np.array([gs - ge, ge - gw])


def test_the_drivers_gravity_is_mujocos_at_any_pose_attitude_and_payload():
  model, data = _standalone()
  spec = qm.CHOSEN.arm
  arm = am.ArmDriver(model, data, spec)
  rng = np.random.default_rng(405)
  for k in range(60):
    qs, qe = rng.uniform(*spec.shoulder_range), rng.uniform(*spec.elbow_range)
    data.qpos[19:22] = qs, qe, -(qs + qe) + rng.normal(0.0, 0.01)
    quat = rng.normal(size=4)
    data.qpos[3:7] = quat / np.linalg.norm(quat)
    arm.payload = (0.3, (0.03, -0.022)) if k % 2 else (0.0, (0.0, 0.0))
    mujoco.mj_forward(model, data)
    assert np.allclose(arm.gravity(), _jacobian_gravity(model, data, arm),
                       rtol=0, atol=1e-12)


def test_the_driver_reads_nothing_a_forward_pass_writes():
  # A restart forwards the world at the saved instant, where a running
  # world's step reads the kinematics one step old: a driver reading them
  # parts the two worlds at the first step back (#345's parity).
  model, data = _standalone()
  arm = am.ArmDriver(model, data, qm.CHOSEN.arm)
  arm.aim(2.0, -2.0)
  arm.payload = (0.2, (0.0, -0.022))
  data.qvel[18:21] = 0.3, -0.2, -0.1
  target = arm.target.copy()
  tau = arm.step()
  for name in ("xpos", "xipos", "xmat", "ximat", "xquat", "xanchor", "xaxis",
               "cdof", "cinert", "subtree_com", "site_xpos", "site_xmat",
               "qfrc_bias", "sensordata"):
    getattr(data, name)[...] = 7.0
  arm.target[:] = target
  assert np.array_equal(arm.step(), tau)


# ---- the served body ---------------------------------------------------------------


def test_the_served_body_holds_its_arm_stowed_through_a_walk(quad_world):
  body = _quad(quad_world)
  try:
    mis = body.mission
    worst = 0.0
    for _ in range(300):                       # 0.6 s: a start into a walk
      body.run(mis._twist_routine(0.5, 0.0, 0.0))
      qs, qf = mis.arm.q()
      worst = max(worst, abs(qs - STOW[0]), abs(qf - qs - STOW[1]))
    assert worst < math.radians(3.0), math.degrees(worst)
    assert np.all(body.data.ctrl[list(mis.arm_acts)] != 0.0)   # driven, not limp
  finally:
    body.close()


def test_a_fall_and_the_rest_reflex_fold_the_arm_first(quad_world):
  body = _quad(quad_world)
  try:
    mis, d = body.mission, body.data
    q = body.handle.qpos_adr(body.model)
    stowed = [STOW[0], STOW[0] + STOW[1]]
    # a fall: the torso on its side
    mis.arm.aim(1.0, -0.5)
    mis.arm.payload = (0.21, (0.0, -0.022))
    d.qpos[q + 3:q + 7] = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    mujoco.mj_forward(body.model, d)
    mis._before_step(qb.STILL)
    assert mis.posture == qb.GETTING_UP
    assert np.allclose(mis.arm.goal, stowed) and mis.arm.payload[0] == 0.0
    # ...and lying down to rest
    body.start_at(1.5, 0.5, 0.0)
    mis.arm.aim(1.0, -0.5)
    mis.want = "lie"
    mis._before_step(qb.STILL)
    assert mis.posture == qb.LYING_DOWN and np.allclose(mis.arm.goal, stowed)
  finally:
    body.close()


def test_a_program_moves_one_arm_joint_and_a_walk_folds_it_back(quad_world, monkeypatch):
  body = _quad(quad_world)
  try:
    mis = body.mission
    stood = []
    real = mis.stand_routine

    def stand():                               # an arm move stands the body first
      stood.append(True)
      return (yield from real())
    monkeypatch.setattr(mis, "stand_routine", stand)
    sh, el = body.actuator("arm_shoulder"), body.actuator("arm_elbow")
    with pytest.raises(KeyError):
      body.actuator("lift")
    assert body.run(body.ramp_routine(sh, 2.2, axes.AXES["shoulder"].speed))
    assert stood
    assert mis.arm.q()[0] == pytest.approx(2.2, abs=qb.ARM_TOL)
    # a setpoint is what the joint is held to, never the motor's torque
    assert body.setpoint(sh) == pytest.approx(2.2)
    assert body.setpoint(el) == pytest.approx(STOW[1])
    life = SimpleNamespace(body=body, model=body.model, data=body.data)
    assert axes.setpoints(life, None) == {"shoulder": 2.2, "elbow": round(STOW[1], 4)}
    # a verb that walks folds it first: the shoulder, then the elbow
    pose = st.travel_pose(life, None)
    assert [(a, t) for a, t, _ in pose] == [(sh, STOW[0]), (el, STOW[1])]
    body.run(st.travel_pose_routine(life))
    assert mis.arm.q()[0] == pytest.approx(STOW[0], abs=qb.ARM_TOL)
  finally:
    body.close()


def test_a_move_the_arm_cannot_finish_is_a_failed_step(quad_world, monkeypatch):
  from pluggybot import tick
  body = _quad(quad_world)
  try:
    life = SimpleNamespace(body=body, model=body.model, data=body.data)
    monkeypatch.setattr(body.mission, "arm_ramp_routine",
                        lambda *a, **kw: tick.result(False))
    verdict = body.run(st._move(life, {"axis": "shoulder", "target": 1.0}))
    assert not verdict["ok"] and "did not reach" in verdict["reason"]
  finally:
    body.close()


def test_a_resting_body_records_no_peer_to_hold_for(quad_world):
  # The drive holds while it has a fresh sighting and the rest reflex lays
  # a holding body down; lying, its nose camera sees a resting peer's arm
  # along the floor, and the two held for each other for the rest of an
  # explore (the pair's recording, #405).
  body = _quad(quad_world)
  try:
    mis = body.mission
    ahead = np.array([[0.4, 0.0, 0.2]])
    mis.posture = qb.LYING
    assert body.watch_for_peers(ahead) is None and mis.peer_seen_m is None
    mis.posture = qb.STANDING
    assert body.watch_for_peers(ahead) == pytest.approx(0.4)
  finally:
    body.close()


def test_the_pack_bills_the_arms_two_motors(quad_world):
  body = _quad(quad_world)
  try:
    pack = body.pack(20.0, 1.0)
    d, acts = body.data, list(body.mission.arm_acts)
    d.ctrl[acts] = 0.0
    mujoco.mj_forward(body.model, d)
    idle = pack.power_draw(d)
    tau = np.array([3.0, -2.0])
    d.ctrl[acts] = tau
    mujoco.mj_forward(body.model, d)
    lim = JointLimits.of(qm.CHOSEN.arm.motor)
    copper = float((1.5 * lim.r_phase[0] * (tau / lim.kt[0]) ** 2).sum())
    shaft = float(np.clip(tau * d.actuator_velocity[acts], 0.0, None).sum())
    assert pack.power_draw(d) - idle == pytest.approx(copper + shaft, rel=1e-9)
  finally:
    body.close()


def test_the_arm_is_carried_across_a_restart(quad_world):
  a, b = _quad(quad_world), _quad(quad_world)
  try:
    a.mission.arm.aim(1.0, -0.5)
    a.mission.arm.target[:] = (1.2, 0.4)
    a.mission.arm.payload = (0.21, (0.0, -0.022))
    b.restore_kept(*a.kept_state())
    assert np.array_equal(b.mission.arm.target, a.mission.arm.target)
    assert np.array_equal(b.mission.arm.goal, a.mission.arm.goal)
    assert b.mission.arm.payload == a.mission.arm.payload
  finally:
    a.close()
    b.close()


# ---- a program's axes --------------------------------------------------------------


def test_a_legged_program_moves_its_own_arm_and_no_other_bodys_axis():
  facts = world_facts(QUAD_HOME)
  lang.compile_procedure('def reach():\n  move("shoulder", 1.0)\n'
                         '  if read("elbow") < 0:\n    move("elbow", -1.0)\n', facts)
  with pytest.raises(st.Refused, match="lift"):
    lang.compile_procedure('def lift():\n  move("lift", 0.1)\n', facts)
  rule = ov.procedure_rule(False)
  assert "  shoulder: -1.2..3.4 rad" in rule and "  elbow -- the arm's elbow" in rule
  assert "lift" not in rule.split("AXES")[1]
  # ...and the rover's world names neither of the quadruped's joints
  home = world_facts("home")
  assert not set(axes.ARM_JOINTS) & (set(home.axes) | set(home.sensors))
