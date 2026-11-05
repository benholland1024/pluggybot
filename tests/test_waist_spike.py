"""#560 stage (c)'s instrument (`scripts/waist_spike.py`): each waist candidate
is today's body with a waist put in at the pelvis's top, nothing else -- its
zero pose today's to the micrometre, and its waist turning what rides above
it (the head's sensors, the port, the arms) and nothing below (the IMU, the
legs). Every number the instrument reports (SimNotes, "A waist, measured
before it is built") stands on that."""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import waist_spike as ws  # noqa: E402

from pluggybot.humanoid import model as hm  # noqa: E402

ABOVE = ("lidar", "charge_port", "left_grip", "right_grip")
BELOW = ("imu", "left_foot", "right_foot")


def _places(sm) -> dict:
  m, d = sm.m, sm.d
  # kinematics places bodies; a camera is placed by `mj_camlight`, as the
  # served code places one (`HumanoidBody`, `HumanoidPlaces`)
  mujoco.mj_camlight(m, d)
  out = {n: (d.site_xpos[m.site(n).id].copy(), d.site_xmat[m.site(n).id].reshape(3, 3).copy())
         for n in ABOVE + BELOW}
  for c in ("depth_eye", "color_eye", "nav_eye"):
    out[c] = (d.cam_xpos[m.camera(c).id].copy(), d.cam_xmat[m.camera(c).id].reshape(3, 3).copy())
  return out


def _apart(a, b) -> tuple[float, float]:
  """How far apart two placed frames are, m, and how far turned, rad."""
  turn = (np.trace(a[1].T @ b[1]) - 1.0) / 2.0
  return float(np.linalg.norm(a[0] - b[0])), float(np.arccos(np.clip(turn, -1.0, 1.0)))


def test_each_candidate_is_todays_body_with_a_waist_and_its_waist_turns_only_what_is_above():
  q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  up = np.array([0.0, 0.0, -1.0])
  today = ws.WaistModel("none")
  today.pose(q0, up)
  ref = _places(today)
  for kind, joints in ws.CANDIDATES.items():
    sm = ws.WaistModel(kind)
    assert set(sm.waist_adr) == {ws.WAIST_JOINTS[j] for j in joints}
    sm.pose(q0, up)
    for name, at in _places(sm).items():
      assert max(_apart(at, ref[name])) < 1e-6, (kind, name)
    for joint in sm.waist_adr:
      sm.set_waist({joint: 0.3})
      sm.pose(q0, up)
      moved = _places(sm)
      for name in ABOVE + ("depth_eye", "nav_eye"):
        far, turned = _apart(moved[name], ref[name])
        # moved, or turned where it rides the joint's axis (the LIDAR the yaw's)
        assert far > 0.02 or turned > 0.25, (kind, joint, name)
      for name in BELOW:
        assert max(_apart(moved[name], ref[name])) < 1e-6, (kind, joint, name)
      sm.set_waist({})


def test_the_instruments_copies_of_the_served_pose_and_statue_are_the_served_ones():
  """The instrument's `WaistModel.pose` and `static_legs` copy
  `arms.SelfModel.pose` and `reach.static_legs`, the waist's joints added:
  on today's body they agree with the served ones, or a change to the served
  code has not reached the copies, and the candidates are no longer measured
  as the served reach plans."""
  from pluggybot.humanoid import reach as rc
  from pluggybot.humanoid.arms import SelfModel
  rng = np.random.default_rng(560)
  today, served = ws.WaistModel("none"), SelfModel()
  q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  for _ in range(4):
    q = q0 + rng.normal(0.0, 0.1, len(q0))
    g = np.array([0.0, 0.0, -1.0]) + rng.normal(0.0, 0.15, 3)
    today.pose(q, g)
    served.pose(q, g)
    for name in ("left_grip", "right_grip", "lidar", "charge_port"):
      a = today.d.site_xpos[today.m.site(name).id]
      b = served.d.site_xpos[served.m.site(name).id]
      assert np.abs(a - b).max() < 1e-9, name
  for height, pitch in rc.POSTURES[:4]:
    mine, ok = ws.static_legs(today, height, pitch)
    assert ok
    for name, v in rc.static_legs(height, pitch, served).items():
      assert abs(mine[name] - v) < 1e-6, (height, pitch, name)
