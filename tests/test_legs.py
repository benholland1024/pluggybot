"""The quadruped body (issue #377): the model, the honest actuator, and the
scripted gait the torque tables are measured with. SimNotes, "The quadruped
body"; `scripts/quad_spike.py` flies the tables."""

import math
from pathlib import Path

import mujoco
import numpy as np

from pluggybot.legs import model as qm
from pluggybot.legs.actuator import GIM8108_8, JointLimits, envelope
from pluggybot.legs.scripted import Command, VirtualModel

ROOT = Path(__file__).resolve().parents[1]


def _compiled(spec=qm.CHOSEN, **kw):
  model = mujoco.MjModel.from_xml_string(qm.body_xml(spec, **kw))
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
  model, data = _compiled()
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
  model, data = _compiled()
  vm = VirtualModel(model, data, qm.CHOSEN)
  lim = JointLimits.of(qm.CHOSEN.motor)
  for _ in range(int(2.0 / model.opt.timestep)):
    cmd = Command(gait="trot", vx=0.5 * min(data.time / 0.5, 1.0), period=0.35)
    data.ctrl[:] = lim.clip(vm.torque(cmd), data.qvel[vm.vadr])
    mujoco.mj_step(model, data)
  assert data.qpos[0] > 0.6
  assert abs(data.qpos[2] - qm.CHOSEN.stand_height) < 0.03
