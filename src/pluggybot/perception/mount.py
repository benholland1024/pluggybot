"""Where a sensor sits on its robot, off the robot's own drawing (#560's
stage (c3)): its place and turn in the robot's root body's frame, composed
through the bodies between them -- and each joint on the way turned to the
angle its own driver reads, which the caller hands in (the humanoid's waist,
`HumanoidMission.waist_read`). ⚠ NEVER THE SIMULATOR'S ANGLE, and never a
silent zero: a mount with a joint on its way refuses a pose without its
reading (`Mount.pose`). The depth camera and the LIDAR ride the waist;
before it, nothing on the humanoid's head was jointed to its pelvis, and
both mounts were fixed when the sensor was built.
"""

import mujoco
import numpy as np

from pluggybot.telemetry.protocol import ROBOT_ROOT


def _quat_mat(quat) -> np.ndarray:
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
  return mat.reshape(3, 3)


def _turn(axis: np.ndarray, angle: float) -> np.ndarray:
  quat = np.zeros(4)
  mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, dtype=np.float64), float(angle))
  return _quat_mat(quat)


class Mount:
  """A frame on a robot -- `pos` and `mat` in body `body`'s frame -- placed
  in its root body's frame (named `root`; where the world has none, the
  frame of the free body it rides).
  Every body on the way may carry one hinge, turned as its driver reads it;
  `joints` names them, the robot's prefix off (`waist_yaw`)."""

  def __init__(self, model, body: int, pos, mat, root: str) -> None:
    try:
      top = model.body(root).id
    except KeyError:
      top = 0
    prefix = root[:-len(ROBOT_ROOT)] if root.endswith(ROBOT_ROOT) else ""
    self.pos0 = np.array(pos, dtype=np.float64)
    self.mat0 = np.array(mat, dtype=np.float64).reshape(3, 3)
    #: Each body up the way: its place and turn in its parent at zero, and
    #: its hinge -- (bare name, anchor, axis) in its own frame -- or None.
    self._chain = []
    b = int(body)
    while b not in (top, 0):
      hinge = None
      if model.body_jntnum[b]:
        j = int(model.body_jntadr[b])
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and top == 0:
          # ...no such root: the frame of the body that is some robot's
          break
        if model.body_jntnum[b] > 1 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
          raise ValueError(f"body {b} is not jointed to its parent by one hinge")
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        hinge = (name[len(prefix):] if name.startswith(prefix) else name,
                 np.array(model.jnt_pos[j], dtype=np.float64),
                 np.array(model.jnt_axis[j], dtype=np.float64))
      self._chain.append((np.array(model.body_pos[b], dtype=np.float64),
                          _quat_mat(model.body_quat[b]), hinge))
      b = int(model.body_parentid[b])
    #: The joints its pose turns with, their bare names, nearest it first.
    self.joints = tuple(h[0] for _, _, h in self._chain if h is not None)

  def pose(self, q: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(place (3,), turn (3, 3)) in the root's frame, each joint on the way
    at `q[name]`, rad: a joint `q` does not give is a ValueError."""
    pos, mat = self.pos0.copy(), self.mat0.copy()
    for at, rot, hinge in self._chain:
      if hinge is not None:
        name, anchor, axis = hinge
        if q is None or name not in q:
          raise ValueError(f"{name} turns this mount: hand in its angle as its driver "
                           "reads it")
        r = _turn(axis, q[name])
        pos = anchor + r @ (pos - anchor)
        mat = r @ mat
      pos = at + rot @ pos
      mat = rot @ mat
    return pos, mat


def site_mount(model, site: int, root: str) -> Mount:
  """A site's mount (`Mount`)."""
  return Mount(model, int(model.site_bodyid[site]), model.site_pos[site],
               _quat_mat(model.site_quat[site]), root)


def camera_mount(model, cam: int, root: str) -> Mount:
  """A camera's mount (`Mount`)."""
  return Mount(model, int(model.cam_bodyid[cam]), model.cam_pos[cam],
               _quat_mat(model.cam_quat[cam]), root)
