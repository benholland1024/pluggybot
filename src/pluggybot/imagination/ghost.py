"""What the robot thinks is there, on the wire (issue #466, stage 3): the
`imagined` event, from which the site draws the robot's model as a ghost
over the real object.

⚠ IT CARRIES THE ROBOT'S MODEL AND NOTHING ELSE: its parts, where it drew
them and how its imagination moved them along the probe -- never the
world's numbers, which are kept as `Task.secret` is. Its frame is the
robot's MAP: the site draws it in the world's, which the map is anchored to
(the dock's commissioned prior), so the ghost sits off the real object by
the robot's own error of where it is -- part of what it thinks.

A part is one primitive at its body's origin (`compile.py`), so it is a
scene body of one geom: a pose a part, `[x, y, z, qw, qx, qy, qz]` (m,
MuJoCo's order), as a frame's bodies are, and the geom's own `size` and
local `quat` on the scene's conventions (protocol/README.md: a box's FULL
extents, a cylinder's radius and length, its quat already turning a
ThreeJS cylinder onto MuJoCo's axis: `AXIS_FIX`).
"""

from __future__ import annotations

import mujoco

from pluggybot.imagination.compile import PREFIX, compile_scene
from pluggybot.imagination.rollout import Readings
from pluggybot.imagination.scene import parse

#: The ghost's replay is sampled this often, Hz: the stream's frame rate.
HZ = 20.0
#: Positions are rounded to 0.1 mm, as a frame's are; quaternions to 1e-5.
POS_DP, QUAT_DP = 4, 5
#: A cylinder geom's local quat: +90 deg about x, ThreeJS's +y onto MuJoCo's
#: +z -- `telemetry.scene.AXIS_FIX`, held here because importing that
#: module loads the house (its `main`), and nothing the imagination loads may
#: (the fence; a test pins the two equal).
AXIS_FIX = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)


def _pose(d, b) -> list[float]:
  return ([round(float(v), POS_DP) for v in d.xpos[b]]
          + [round(float(v), QUAT_DP) for v in d.xquat[b]])


def message(document: dict, readings: Readings | None, *, robot: str, t: float,
            t0: float | None = None, dt: float = 0.002, what: str = "", extra=None) -> dict:
  """The `imagined` event for `document` (a fitted model, the map's frame):
  its parts and their drawn pose (`rest`), and -- given the fitted world's
  `readings` along a record that began at sim time `t0`, a row every `dt`
  -- every part's pose through that replay at `HZ` (`replay`)."""
  scene = parse(document)
  world = compile_scene(scene, carrying=None)
  m = world.model
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  bodies = {p.id: m.body(f"{PREFIX}{p.id}").id for p in scene.parts}
  parts = []
  for p in scene.parts:
    cylinder = p.shape == "cylinder"
    size = ([round(p.size[0] / 2, POS_DP), round(p.size[1], POS_DP)] if cylinder
            else [round(v, POS_DP) for v in p.size])
    parts.append({"id": p.id, "shape": "cylinder" if cylinder else "box", "size": size,
                  "quat": [round(v, QUAT_DP) for v in (AXIS_FIX if cylinder
                                                       else (1.0, 0.0, 0.0, 0.0))],
                  "on": p.on})
  out = {"type": "imagined", "robot": robot, "t": round(float(t), 3), "what": what,
         "parts": parts, "rest": {pid: _pose(d, b) for pid, b in bodies.items()},
         "joints": [{"id": j.id, "type": j.type, "part": j.part,
                     "at": [round(v, POS_DP) for v in j.at],
                     "axis": [round(v, QUAT_DP) for v in j.axis],
                     "range": None if j.range is None else [round(v, QUAT_DP) for v in j.range]}
                    for j in scene.joints]}
  if readings is not None and t0 is not None and scene.joints:
    every = max(1, round(1.0 / (HZ * dt)))
    moving = sorted({pid for pid in bodies if any(
      _rides(scene, pid, j.part) for j in scene.joints)})
    frames = []
    for k in range(0, len(readings.t), every):
      for j in scene.joints:
        d.qpos[world.joints[j.id][0]] = readings.joints[j.id][k]
      mujoco.mj_kinematics(m, d)
      frames.append([_pose(d, bodies[pid]) for pid in moving])
    out["replay"] = {"t0": round(float(t0), 3), "hz": HZ, "parts": moving, "frames": frames}
  if extra:
    out.update(extra)
  return out


def _rides(scene, pid: str, moved: str) -> bool:
  """Is part `pid` the part `moved`, or on it, however far down?"""
  cur = scene.part(pid)
  while True:
    if cur.id == moved:
      return True
    if cur.on is None:
      return False
    cur = scene.part(cur.on)
