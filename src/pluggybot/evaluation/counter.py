"""The chest on a counter (issue #507, stage 3): #466's drop-handle chest
set on a counter in the house's empty storeroom, and the humanoid standing
in front of it -- its hands' demo world, as `evaluation.probe.set_out` is
the quadruped's probe's. Seeded; no change to the served house (a world of
its own, as `home_quad_chest` is).

The truth stays here -- where the counter and the chest stand, where the
knob truly is, whether the pads truly hold it -- OUR staging and grading.
The robot finds the knob by its tag (`HumanoidBody.spot`) and judges its
grip by its gripper's own reading (`Body.gripped`).

  set_out     chest k drawn (`chest.draw(k)`), the counter and the chest
              where they stand, and where the humanoid starts
  world       the house with the humanoid, the counter and the chest put in
  knob_world  where the knob truly is (our grading's)
  pads_hold   whether the gripper's pads truly touch the knob (ours)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from pluggybot.activity import chest as ch

HOME_XML = "models/home_world.xml"
#: Where the humanoid starts: the storeroom, empty (`home.world`'s x 22-28,
#: y -6-0), facing +x, as #466's probe did.
START = (24.0, -3.0)
#: The counter: its top a counter's height, m (#507's demos are at a
#: counter's height, 0.94 m; a kitchen's is 0.90-0.94), its depth and its
#: width, and how far its front edge stands ahead of the robot's soles.
COUNTER_H, COUNTER_D, COUNTER_W = 0.90, 0.60, 1.20
#: A set-out draws: the knob's plumb place this far ahead of the soles'
#: middle and this far to its left (in front of the left shoulder, which
#: hangs 0.18 m out), and the chest turned this far off square (m, rad).
#: ⚠ The knob then swings back under its bracket (0.57 rad on its pin), 2-3
#: cm further. Drawn 0.42-0.50 m it hung where a crouch's arm met the
#: counter; hanging under 0.37 m, at 0.99 m up, no grip its arm holds folds
#: the hand back that near (two set-outs of ten drawn from 0.34).
KNOB_AHEAD = (0.37, 0.43)
KNOB_LEFT = (0.12, 0.24)
TURNED = math.radians(12.0)
#: The chest's front this far back from the counter's edge, at least, m:
#: its knob hangs 0.10 m in front of its front, over the counter's top.
EDGE_BACK = 0.12
#: `set_out(k)`'s seed is this plus k, and the demo's set-out.
SETOUT_SEED = 5070
DEMO = 0
#: The cameras' near plane, m: the quadruped's (`legs.world.NEAR_M`); at
#: MuJoCo's own, scaled from the house's pinned extent (0.37 m), a hand
#: near the head was cut.
NEAR_M = 0.10


@dataclass(frozen=True)
class SetOut:
  """Chest k on the counter: its hidden parameters (`chest.draw(k)`), the
  chest's frame (x, y on the counter, yaw), the counter's front edge's
  middle (x, y) and its heading, and the humanoid's start (x, y, heading)."""
  k: int
  lid: ch.Lid
  chest: tuple[float, float, float]
  counter: tuple[float, float, float]
  start: tuple[float, float, float]


def _stance_ahead() -> float:
  """How far the soles' middle is ahead of the pelvis at the stand, m, off
  the robot's drawing: where a set-out's distances are measured from."""
  from pluggybot.humanoid.arms import SelfModel
  from pluggybot.humanoid import model as hm
  sm = SelfModel()
  sm.pose(np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES]),
          np.array([0.0, 0.0, -1.0]))
  return float(-sm.d.qpos[0])


def set_out(k: int) -> SetOut:
  """Chest `k` on the counter (seeded): the robot at `START` facing +x, the
  knob's plumb place `KNOB_AHEAD` ahead of its soles and `KNOB_LEFT` to its
  left, the chest turned up to `TURNED`, the counter square to the robot
  with its edge `EDGE_BACK` ahead of the chest's front."""
  if k < 0:
    raise ValueError(f"a set-out is a whole number from 0, not {k}")
  rng = np.random.default_rng(SETOUT_SEED + k)
  sx = START[0] + _stance_ahead()
  ahead, left = float(rng.uniform(*KNOB_AHEAD)), float(rng.uniform(*KNOB_LEFT))
  yaw = float(rng.uniform(-TURNED, TURNED))
  kx, ky = sx + ahead, START[1] + left
  off = ch.HINGE + ch.PIN + ch.KNOB_OFF                  # the plumb knob, chest frame
  c, s = math.cos(yaw), math.sin(yaw)
  chest = (kx - (c * off[0] - s * off[1]), ky - (s * off[0] + c * off[1]), yaw)
  # The chest's front edge (its frame's x = 0 face) nearest the robot.
  front = min(chest[0] + c * 0.0 - s * dy for dy in (-ch.BOX_W / 2, ch.BOX_W / 2))
  counter = (front - EDGE_BACK, START[1] + left, 0.0)
  return SetOut(k=k, lid=ch.draw(k), chest=chest, counter=counter,
                start=(START[0], START[1], 0.0))


def world(so: SetOut, policy=None):
  """A set-out's world: the house with the humanoid standing at its start,
  the counter and the chest on it. Returns the model, its data and the
  humanoid's body (`HumanoidBody`), its stance commissioned where it
  stands."""
  from pluggybot.body import body_for
  from pluggybot.humanoid import model as hm
  spec = mujoco.MjSpec.from_file(HOME_XML)
  frame = spec.worldbody.add_frame()
  frame.pos = [so.start[0], so.start[1], 0.0]
  frame.quat = [math.cos(so.start[2] / 2), 0.0, 0.0, math.sin(so.start[2] / 2)]
  spec.attach(hm.attachable(), prefix="", frame=frame)
  cx, cy, cyaw = so.counter
  body = spec.worldbody.add_body(name="counter", pos=[cx + COUNTER_D / 2, cy, 0.0],
                                 quat=[math.cos(cyaw / 2), 0.0, 0.0, math.sin(cyaw / 2)])
  body.add_geom(name="counter", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[COUNTER_D / 2, COUNTER_W / 2, COUNTER_H / 2], pos=[0, 0, COUNTER_H / 2],
                rgba=[0.86, 0.85, 0.82, 1])
  ch.attach_chest(spec, so.lid, so.chest[:2], so.chest[2], z=COUNTER_H)
  spec.visual.map.znear = NEAR_M / spec.stat.extent
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  humanoid = body_for(model, data, policy=policy)
  humanoid.start_at(*so.start)
  return model, data, humanoid


def knob_world(model, data, name: str = "chest") -> np.ndarray:
  """Where the knob truly is, its centre: our grading's."""
  return data.xpos[model.body(f"{name}_knob_body").id].copy()


def pads_hold(model, data, side: str, name: str = "chest") -> bool:
  """Whether both of the gripper's pads truly touch the knob: our grading
  of the grip the robot judged by its own reading."""
  from pluggybot.humanoid import model as hm
  knob = model.geom(f"{name}_knob").id
  touching = set()
  for i in range(data.ncon):
    c = data.contact[i]
    for jaw in hm.JAWS:
      pad = model.geom(f"{side}_{jaw}_pad").id
      if {int(c.geom1), int(c.geom2)} == {knob, pad}:
        touching.add(jaw)
  return len(touching) == 2
