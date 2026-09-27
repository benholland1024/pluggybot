"""The quadruped body (issue #377): the model, the honest actuator, and the
scripted gait the torque tables are measured with. SimNotes, "The quadruped
body"; `scripts/quad_spike.py` flies the tables."""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.legs import model as qm
from pluggybot.legs.actuator import GIM8108_8, JointLimits, driver_gains, envelope
from pluggybot.legs.scripted import Command, VirtualModel

ROOT = Path(__file__).resolve().parents[1]


def _compiled(spec=qm.CHOSEN, **kw):
  model = mujoco.MjModel.from_xml_string(qm.body_xml(spec, **kw))  # scenery=
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return model, data


def test_the_committed_model_is_the_generators():
  text = (ROOT / qm.MODEL_XML).read_text()
  assert text.split("\n", 1)[1] == qm.body_xml(qm.CHOSEN), (
    "stale: uv run python -m pluggybot.legs.model")


def test_the_stand_keyframe_puts_every_foot_on_the_floor_under_its_hip():
  spec = qm.CHOSEN
  model, data = _compiled()
  for leg in qm.LEGS:
    foot = data.site(f"{leg}_foot").xpos
    hip = data.body(f"{leg}_thigh").xpos
    assert abs(foot[2] - spec.foot_r) < 1e-6
    assert abs(foot[0] - hip[0]) < 1e-6 and abs(foot[1] - hip[1]) < 1e-6


def test_the_models_mass_is_the_budgets():
  model, _ = _compiled()
  assert math.isclose(model.body_subtreemass[1], qm.CHOSEN.mass, rel_tol=1e-9)


def test_a_foot_grips_with_its_own_friction_not_the_floors():
  # Friction combines as the pair's MAX unless the foot claims priority
  # (SimNotes, "THE caster lesson"): without it the floor's 1.0 wins and
  # randomising the foot's friction in training would change nothing.
  model, data = _compiled(drive="torque")
  for _ in range(50):
    mujoco.mj_step(model, data)
  feet = {model.geom(f"{leg}_foot").id for leg in qm.LEGS}
  touching = [c for c in data.contact[:data.ncon]
              if c.geom1 in feet or c.geom2 in feet]
  assert touching
  foot_mu = model.geom_friction[next(iter(feet))][0]
  assert foot_mu < 1.0
  for c in touching:
    assert c.friction[0] == foot_mu


def test_the_envelope_falls_to_zero_at_no_load_and_keeps_the_peak_braking():
  sat, peak, noload = np.array([60.0]), np.array([22.0]), np.array([40.0])
  lo, hi = envelope(np.array([0.0]), sat, peak, noload)
  assert hi[0] == 22.0 and lo[0] == -22.0
  lo, hi = envelope(np.array([40.0]), sat, peak, noload)
  assert hi[0] == 0.0 and lo[0] == -22.0
  # Driven backwards at speed, a positive torque brakes: the peak stands.
  lo, hi = envelope(np.array([-40.0]), sat, peak, noload)
  assert hi[0] == 22.0


def test_the_knee_belt_multiplies_torque_divides_speed_and_squares_inertia():
  lim = JointLimits.of(GIM8108_8, knee_ratio=2.0)
  assert lim.peak[2] == 2 * lim.peak[1] and lim.noload[2] == lim.noload[1] / 2
  model, _ = _compiled(qm.CHOSEN.with_(knee_ratio=2.0))
  knee = model.dof_armature[model.jnt_dofadr[model.joint("FL_knee").id]]
  flex = model.dof_armature[model.jnt_dofadr[model.joint("FL_hip_flex").id]]
  assert math.isclose(knee, 4 * flex)


def test_the_scripted_gait_trots_forward_without_falling():
  # The tables' instrument: if it stops walking, every number it made is
  # unreproducible. Two seconds of a 0.5 m/s trot.
  model, data = _compiled(drive="torque")
  vm = VirtualModel(model, data, qm.CHOSEN)
  lim = JointLimits.of(qm.CHOSEN.motor)
  for _ in range(int(2.0 / model.opt.timestep)):
    cmd = Command(gait="trot", vx=0.5 * min(data.time / 0.5, 1.0), period=0.35)
    data.ctrl[:] = lim.clip(vm.torque(cmd), data.qvel[vm.vadr])
    mujoco.mj_step(model, data)
  assert data.qpos[0] > 0.6
  assert abs(data.qpos[2] - qm.CHOSEN.stand_height) < 0.03


# The walking policy as numpy (#377 item 4): the arithmetic, the observation
# mjlab trained on, and what the served process may not import.

def _tiny_policy(path, obs=45, hidden=8, act=12, seed=0):
  """A two-layer policy in the export's format, small enough to check by
  hand."""
  import json
  rng = np.random.default_rng(seed)
  meta = {"joint_names": ",".join(qm.JOINT_NAMES),
          "default_joint_pos": ",".join(str(q) for q in qm.pose_qpos(
            qm.CHOSEN, qm.CHOSEN.stand_height)),
          "action_scale": ",".join(["0.3"] * act),
          "observation_names": "base_ang_vel,projected_gravity,joint_pos,"
                               "joint_vel,actions,command"}
  stiffness, damping = driver_gains(qm.CHOSEN.motor)
  training = {"train_dt": 0.005, "decimation": 4, "stiffness": stiffness,
              "damping": damping, "env_steps": 0, "envs": 1, "gpu": "-",
              "wall": "-"}
  arrays = {"obs_mean": rng.normal(size=obs).astype(np.float32),
            "obs_div": (1 + rng.random(obs)).astype(np.float32),
            "w0": rng.normal(size=(obs, hidden)).astype(np.float32),
            "b0": rng.normal(size=hidden).astype(np.float32),
            "w1": rng.normal(size=(hidden, act)).astype(np.float32),
            "b1": rng.normal(size=act).astype(np.float32),
            "layers": np.array(2), "meta": np.array(json.dumps(meta)),
            "training": np.array(json.dumps(training))}
  np.savez(path, **arrays)
  return arrays


def test_the_policy_is_a_normaliser_and_an_elu_mlp(tmp_path):
  from pluggybot.legs.policy import WalkingPolicy
  a = _tiny_policy(tmp_path / "p.npz")
  policy = WalkingPolicy(tmp_path / "p.npz")
  obs = np.linspace(-2, 2, 45)
  x = (obs - a["obs_mean"].astype(float)) / a["obs_div"].astype(float)
  h = x @ a["w0"].astype(float) + a["b0"]
  h = np.where(h > 0, h, np.exp(h) - 1)
  want = h @ a["w1"].astype(float) + a["b1"]
  assert np.allclose(policy.act(obs), want, atol=1e-9)
  assert policy.period == 0.02


def test_the_observation_is_the_one_mjlab_trained_on(tmp_path):
  # base_ang_vel, projected_gravity, joint_pos - default, joint_vel,
  # last action, command: the order the ONNX metadata names.
  from pluggybot.legs.policy import PolicyDriver, Twist, WalkingPolicy
  _tiny_policy(tmp_path / "p.npz")
  model, data = _compiled()
  drv = PolicyDriver(model, data, WalkingPolicy(tmp_path / "p.npz"))
  obs = drv.observation(Twist(0.4, -0.1, 0.3))
  assert obs.shape == (45,)
  assert np.allclose(obs[3:6], [0, 0, -1])        # standing level
  assert np.allclose(obs[6:18], 0, atol=1e-5)     # the keyframe, to 6 digits
  assert np.allclose(obs[42:45], [0.4, -0.1, 0.3])


def test_the_policy_decides_every_tenth_step_and_the_drivers_run_every_step(tmp_path):
  from pluggybot.legs.policy import PolicyDriver, Twist, WalkingPolicy
  _tiny_policy(tmp_path / "p.npz")
  model, data = _compiled()
  policy = WalkingPolicy(tmp_path / "p.npz")
  calls = []
  act = policy.act
  policy.act = lambda obs: calls.append(1) or act(obs)
  drv = PolicyDriver(model, data, policy)
  assert drv.every == 10                           # 50 Hz on 2 ms steps
  targets, torques = [], []
  for _ in range(25):
    drv.command(Twist())
    targets.append(data.ctrl.copy())
    mujoco.mj_step(model, data)
    torques.append(data.actuator_force.copy())
  assert len(calls) == 3                           # steps 0, 10, 20
  assert np.array_equal(targets[1], targets[9])    # held between decisions
  assert not np.array_equal(targets[9], targets[10])
  assert not np.allclose(torques[1], torques[2])   # the PD tracks the body


def _random_joint_states(model, data, rng):
  mujoco.mj_resetDataKeyframe(model, data, 0)
  data.qpos[7:19] += rng.normal(0, 0.3, 12)
  data.qvel[6:18] = rng.normal(0, 12, 12)
  data.ctrl[:] = data.qpos[7:19] + rng.normal(0, 0.5, 12)
  mujoco.mj_forward(model, data)


def test_the_drivers_dcmotor_is_the_pd_inside_the_envelope():
  # Issue #385 moved the driver's PD and the torque-speed envelope from
  # numpy into MuJoCo's `dcmotor`, whose controller makes a VOLTAGE: its
  # gains are the torque PD's through the motor's K and R, and a wrong
  # sign or a missing back-EMF term is a different motor. At any state
  # inside the no-load speed, the force must be the clipped PD it replaced
  # -- on the nominal pack and on an empty one (`set_bus`).
  from pluggybot.legs.policy import PolicyDriver, WalkingPolicy
  model, data = _compiled()
  stiffness, damping = driver_gains(qm.CHOSEN.motor)
  drv = PolicyDriver(model, data, WalkingPolicy())
  rng = np.random.default_rng(0)
  for bus in (43.2, 36.0):
    drv.set_bus(bus)
    lim = JointLimits.of(qm.CHOSEN.motor, bus_v=bus)
    clipped = 0
    for _ in range(300):
      _random_joint_states(model, data, rng)
      q, qd = data.qpos[7:19], data.qvel[6:18]
      want = lim.clip(stiffness * (data.ctrl - q) - damping * qd, qd)
      inside = np.abs(qd) < lim.noload
      assert np.allclose(data.actuator_force[inside], want[inside],
                         rtol=0, atol=1e-9)
      clipped += int((np.abs(want) >= lim.peak - 1e-9).sum())
    assert clipped > 100                        # the peak clip was exercised


def test_a_policy_walks_only_on_the_drivers_it_was_trained_on(tmp_path):
  import json
  from pluggybot.legs.policy import PolicyDriver, WalkingPolicy
  a = _tiny_policy(tmp_path / "p.npz")
  model, data = _compiled(drive="torque")
  with pytest.raises(ValueError, match="trained"):
    PolicyDriver(model, data, WalkingPolicy(tmp_path / "p.npz"))
  training = json.loads(str(a["training"]))
  training["stiffness"] *= 1.5
  a["training"] = np.array(json.dumps(training))
  np.savez(tmp_path / "q.npz", **a)
  model, data = _compiled()
  with pytest.raises(ValueError, match="trained"):
    PolicyDriver(model, data, WalkingPolicy(tmp_path / "q.npz"))


def test_every_committed_policy_was_trained_on_the_drivers_gains():
  # One definition (`actuator.driver_gains`): the served model's dcmotors
  # run it, `training/` reads it from quadruped.json, and each committed
  # policy recorded what it was trained on.
  import json
  from pluggybot.legs.policy import WalkingPolicy
  gains = driver_gains(qm.CHOSEN.motor)
  body = json.loads((ROOT / qm.CONSTANTS_JSON).read_text())
  assert (body["driver"]["stiffness"], body["driver"]["damping"]) == gains
  files = sorted((ROOT / "models").glob("quadruped_*.npz"))
  assert files
  for path in files:
    policy = WalkingPolicy(path)
    assert (policy.stiffness, policy.damping) == gains, path.name


def test_running_the_policy_imports_no_training_stack():
  # The serving image installs deploy/requirements-serve.txt and nothing
  # else; a policy that needed torch or onnx to run would crash there.
  import subprocess
  import sys
  code = ("import sys, pluggybot.legs.policy, pluggybot.legs.model; "
          "bad = [m for m in ('torch', 'jax', 'warp', 'onnx', 'mjlab') "
          "if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
  out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
  assert out.returncode == 0, out.stdout


# Legged odometry (#377 item 8): the two corrections a ball foot and a
# lagging contact estimate need, pinned on the scripted trot (no policy).

def _trot_odometry(monkeypatch, lag_s):
  from pluggybot.legs import odometry as od
  monkeypatch.setattr(od, "CONTACT_LAG_S", lag_s)
  monkeypatch.setattr(od, "GYRO_NOISE", 0.0)
  monkeypatch.setattr(od, "GYRO_BIAS_DPS", 0.0)
  monkeypatch.setattr(od, "BACKLASH_RAD", 0.0)
  model, data = _compiled(drive="torque")
  vm = VirtualModel(model, data, qm.CHOSEN)
  lim = JointLimits.of(qm.CHOSEN.motor)
  odo = od.LegOdometry(model, data)
  for _ in range(int(4.0 / model.opt.timestep)):
    cmd = Command(gait="trot", vx=0.5 * min(data.time / 0.5, 1.0), period=0.35)
    data.ctrl[:] = lim.clip(vm.torque(cmd), data.qvel[vm.vadr])
    mujoco.mj_step(model, data)
    odo.step()
  err, _ = odo.error()
  return err / odo.distance


def test_odometry_corrects_for_a_rolling_ball_foot(monkeypatch):
  # A ball foot's centre moves while its contact point does not: assumed
  # still, the estimate ran 6.6 % long with perfect contact and no noise.
  assert _trot_odometry(monkeypatch, lag_s=0.0) < 0.015


def test_odometry_survives_a_contact_estimate_31_ms_late(monkeypatch):
  # For 31 ms after each footfall only the lifting pair is flagged planted;
  # averaged in, it made the estimate 27 % of distance wrong.
  assert _trot_odometry(monkeypatch, lag_s=0.031) < 0.04


def test_the_height_scan_is_mjlabs_grid_turned_with_the_heading(tmp_path):
  # The perceptive policy was trained on mjlab's grid (x fastest, 17 x 11
  # points over 1.6 x 1.0 m) scaled by 1/5: a scrambled or unscaled scan
  # shows it a different world. A 10 cm block front-left of the robot must
  # read at exactly the grid points under it.
  import json
  from pluggybot.legs.policy import (SCAN_SIZE_M, PolicyDriver, WalkingPolicy,
                                     scan_offsets)
  a = _tiny_policy(tmp_path / "p.npz", obs=45 + 187)
  meta = json.loads(str(a["meta"]))
  meta["observation_names"] += ",height_scan"
  meta["observation_terms_scale"] = "1,1,1,1,1,1,0.2"
  a["meta"] = np.array(json.dumps(meta))
  np.savez(tmp_path / "p.npz", **a)
  # Its edges between grid lines (x 0.32-0.78, y 0.12-0.48), so which
  # points it covers is geometry, not rounding at an edge.
  block = ('\n    <geom name="block" type="box" size="0.23 0.18 0.05" '
           'pos="0.55 0.3 0.05"/>')
  model, data = _compiled(scenery=block)
  drv = PolicyDriver(model, data, WalkingPolicy(tmp_path / "p.npz"))
  scan = drv.observation(Twist_())[45:] / 0.2
  xy = scan_offsets()
  under = (np.abs(xy[:, 0] - 0.55) < 0.23) & (np.abs(xy[:, 1] - 0.3) < 0.18)
  h0 = data.qpos[2]
  assert len(scan) == 187 and xy.shape == (187, 2)
  assert np.allclose(scan[~under], h0, atol=1e-6)
  assert np.allclose(scan[under], h0 - 0.1, atol=1e-6)
  assert xy[1, 0] - xy[0, 0] > 0 and xy[1, 1] == xy[0, 1]   # x fastest
  assert math.isclose(xy[:, 0].max(), SCAN_SIZE_M[0] / 2, abs_tol=1e-9)


def test_the_policy_reads_the_scan_it_is_given(tmp_path):
  # `quad_spike.py --climb --scan map` hands the driver the D435's map
  # (#388): a driver that kept casting ideal rays would fly the ideal scan
  # under the map's name.
  import json
  from pluggybot.legs.policy import PolicyDriver, WalkingPolicy
  a = _tiny_policy(tmp_path / "p.npz", obs=45 + 187)
  meta = json.loads(str(a["meta"]))
  meta["observation_names"] += ",height_scan"
  meta["observation_terms_scale"] = "1,1,1,1,1,1,0.2"
  a["meta"] = np.array(json.dumps(meta))
  np.savez(tmp_path / "p.npz", **a)
  model, data = _compiled()
  drv = PolicyDriver(model, data, WalkingPolicy(tmp_path / "p.npz"),
                     scan=lambda: np.full(187, 0.25))
  assert np.allclose(drv.observation(Twist_())[45:] / 0.2, 0.25)


def Twist_():
  from pluggybot.legs.policy import Twist
  return Twist()


def test_the_scripted_gait_turns_on_the_spot_through_180_degrees():
  # Its attitude loop once ran about the WORLD's axes: roll and pitch are
  # the heading's, the correction reversed past 90 deg of heading, and a
  # turn on the spot flipped the body at 140 deg.
  model, data = _compiled(drive="torque")
  vm = VirtualModel(model, data, qm.CHOSEN)
  lim = JointLimits.of(qm.CHOSEN.motor)
  lowest = 1.0
  for _ in range(int(4.5 / model.opt.timestep)):
    cmd = Command(gait="trot", yaw_rate=0.8 * min(data.time / 0.5, 1.0), period=0.35)
    data.ctrl[:] = lim.clip(vm.torque(cmd), data.qvel[vm.vadr])
    mujoco.mj_step(model, data)
    lowest = min(lowest, data.qpos[2])
  w, x, y, z = data.qpos[3:7]
  yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  assert abs(yaw) > math.radians(150)          # past the old flip, round to -
  assert lowest > qm.CHOSEN.stand_height - 0.03


def test_the_robot_rests_on_its_belly_with_the_drivers_holding_nothing():
  # A folded leg holds the hips 0.10 m up, so the belly pack hangs below
  # that: lying down, the legs carry nothing (SimNotes, "The belly").
  model, data = _compiled(drive="torque")
  mujoco.mj_resetDataKeyframe(model, data, 1)
  for _ in range(int(1.0 / model.opt.timestep)):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)
  belly = model.geom("belly").id
  touching = {int(g) for pair in data.contact.geom[:data.ncon] for g in pair}
  assert belly in touching
  assert abs(data.qpos[2] - qm.CHOSEN.belly_depth) < 0.005
