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
  q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in ws.JOINTS])
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
  on the served body -- its `yaw` candidate, built (#560's stage (c3)) --
  they agree with the served ones at any waist, or a change to the served
  code has not reached the copies, and the candidates are no longer measured
  as the served reach plans."""
  from pluggybot.humanoid import reach as rc
  from pluggybot.humanoid.arms import SelfModel
  rng = np.random.default_rng(560)
  yaw, served = ws.WaistModel("yaw"), SelfModel()
  q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in ws.JOINTS])
  waist = hm.JOINT_NAMES.index("waist_yaw")
  for turn in (0.0, 0.4, -0.7):
    q = q0 + rng.normal(0.0, 0.1, len(q0))
    g = np.array([0.0, 0.0, -1.0]) + rng.normal(0.0, 0.15, 3)
    yaw.set_waist({"waist_yaw": turn})
    yaw.pose(q, g)
    served.pose(np.insert(q, waist, turn), g)
    for name in ("left_grip", "right_grip", "lidar", "charge_port"):
      a = yaw.d.site_xpos[yaw.m.site(name).id]
      b = served.d.site_xpos[served.m.site(name).id]
      assert np.abs(a - b).max() < 1e-9, (turn, name)
    for height, pitch in rc.POSTURES[:4]:
      mine, ok = ws.static_legs(yaw, height, pitch)
      assert ok
      for name, v in rc.static_legs(height, pitch, served, waist=turn).items():
        assert abs(mine[name] - v) < 1e-6, (turn, height, pitch, name)


def _named(m, obj, n) -> dict:
  return {mujoco.mj_id2name(m, obj, i): i for i in range(n) if mujoco.mj_id2name(m, obj, i)}


def test_the_built_waist_is_the_yaw_candidate_and_at_its_zero_todays_body():
  """#560's stage (c3) built the `yaw` candidate into the served body
  (`model.HumanoidSpec.waist_yaw`): every body, geom and joint as the
  candidate has it -- place, mass, inertia, size, range, armature, friction
  -- and at the waist's zero every part today's body has where it had it, to
  1e-9 m. What #620 measured is what flies."""
  built = mujoco.MjModel.from_xml_string(hm.body_xml())
  ref = mujoco.MjModel.from_xml_string(ws.waist_xml("yaw"))
  today = mujoco.MjModel.from_xml_string(hm.body_xml(ws.TODAY))
  obj_ = mujoco.mjtObj
  for obj, n, fields in ((obj_.mjOBJ_BODY, "nbody", ("body_pos", "body_quat", "body_mass",
                                                  "body_inertia", "body_ipos", "body_iquat")),
                         (obj_.mjOBJ_GEOM, "ngeom", ("geom_pos", "geom_quat", "geom_size",
                                                  "geom_type", "geom_contype")),
                         (obj_.mjOBJ_JOINT, "njnt", ("jnt_pos", "jnt_axis", "jnt_range"))):
    mine, theirs = _named(built, obj, getattr(built, n)), _named(ref, obj, getattr(ref, n))
    assert set(mine) == set(theirs)
    for name, i in theirs.items():
      for f in fields:
        assert np.array_equal(getattr(built, f)[mine[name]], getattr(ref, f)[i]), (name, f)
  joint = built.joint("waist_yaw")
  assert tuple(joint.range) == hm.joint_range("waist_yaw")
  dof = int(built.jnt_dofadr[joint.id])
  assert built.dof_armature[dof] == ref.dof_armature[int(ref.jnt_dofadr[ref.joint("waist_yaw").id])]
  # ...and at zero, today's
  placed = {}
  for m in (built, today):
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    placed[m] = {(k, name): (pos[i].copy(), mat[i].copy())
                 for k, obj, n, pos, mat in (("b", obj_.mjOBJ_BODY, m.nbody, d.xpos, d.xmat),
                                             ("g", obj_.mjOBJ_GEOM, m.ngeom, d.geom_xpos, d.geom_xmat),
                                             ("s", obj_.mjOBJ_SITE, m.nsite, d.site_xpos, d.site_xmat),
                                             ("c", obj_.mjOBJ_CAMERA, m.ncam, d.cam_xpos, d.cam_xmat))
                 for name, i in _named(m, obj, n).items()}
  new = set(placed[built]) - set(placed[today])
  assert new == {("b", "waist_yaw_link"), ("g", "waist_plate"), ("g", "waist_yaw_motor")}
  for key, (pos, mat) in placed[today].items():
    assert np.abs(placed[built][key][0] - pos).max() < 1e-9, key
    assert np.abs(placed[built][key][1] - mat).max() < 1e-9, key
