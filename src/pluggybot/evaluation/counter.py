"""The chest on a counter (issue #507, stage 3): #466's drop-handle chest
set on a counter in the house's empty storeroom, and the humanoid standing
in front of it -- its hands' demo world. Seeded; no change to the served
house (a world of its own).

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
#: ...its generator's sidecar (`home.world`): the house's visual hints, its
#: zones and its boards, which the site draws the house by.
HOME_META = "models/home_world.meta.json"
#: A set-out's world on the wire (#530): the house, the humanoid where the
#: set-out starts it, the counter and its prop -- a world of its own, never
#: the served one.
WORLD = "home_humanoid_counter"
#: Where the humanoid starts: the storeroom, empty (`home.world`'s x 22-28,
#: y -6-0), facing +x, as #466's probe did.
START = (24.0, -3.0)
#: The counter: its top's height, m, its depth and its width. A table's
#: height for a 1.4 m body (#493, Ben): at a kitchen's 0.90 the chest's lid
#: carried its knob 0.52 m ahead and 1.21 m up by the grader's 57 deg -- the
#: arm's full stretch, however the hand and the handle turned together --
#: and 62 deg was out of every set-out's reach (SimNotes, "The humanoid
#: opens the chest").
COUNTER_H, COUNTER_D, COUNTER_W = 0.72, 0.60, 1.20
#: A set-out draws: the knob's plumb place this far ahead of the soles'
#: middle and this far to its left (in front of the left shoulder, which
#: hangs 0.18 m out), and the chest turned this far off square (m, rad).
#: ⚠ The knob then swings back under its bracket (0.57 rad on its pin), 2-3
#: cm further. Its reach's bounds (#493): further, the lid's arc leaves the
#: arm; nearer, no grip folds the hand back onto the shut knob; and 0.12 m
#: left, the left shoulder's roll at its stop, the arm could not move its
#: hand 3 cm toward the body's middle.
KNOB_AHEAD = (0.33, 0.37)
KNOB_LEFT = (0.16, 0.24)
TURNED = math.radians(12.0)
#: The chest's front this far back from the counter's edge, at least, m:
#: its knob hangs 0.10 m in front of its front, over the counter's top.
EDGE_BACK = 0.12
#: `set_out(k)`'s seed is this plus k, and the demo's set-out.
SETOUT_SEED = 5070
DEMO = 0
#: The cameras' near plane, m: at MuJoCo's own, scaled from the house's
#: pinned extent (0.37 m), a hand near the head was cut.
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


def _placed(knob, off, yaw: float) -> tuple[float, float, float]:
  """A prop's frame (x, y, yaw) that puts its point `off` (its own frame)
  at `knob` (x, y), turned `yaw`."""
  c, s = math.cos(yaw), math.sin(yaw)
  return knob[0] - (c * off[0] - s * off[1]), knob[1] - (s * off[0] + c * off[1]), yaw


def _drawn(k: int, seed: int, ahead=KNOB_AHEAD, left=KNOB_LEFT, turned: float = TURNED):
  """A check's set-out k (#525): where its handle stands -- `ahead` of the
  soles' middle and `left` of it -- and how far its prop is turned, drawn
  off the stream (`seed`, k). ⚠ A STREAM OF ITS OWN, never its prop's
  hidden draw's: off one seed with it, the furthest cube was always the
  heaviest, and its weight could be read off where it stood (#525)."""
  if k < 0:
    raise ValueError(f"a set-out is a whole number from 0, not {k}")
  rng = np.random.default_rng((seed, k))
  sx = START[0] + _stance_ahead()
  fore, side = float(rng.uniform(*ahead)), float(rng.uniform(*left))
  return (sx + fore, START[1] + side), float(rng.uniform(-turned, turned)), rng


@dataclass(frozen=True)
class DrawerSetOut:
  """Drawer k in its cabinet on the counter (#525): its hidden parameters
  (`drawer.draw(k)`), the cabinet's frame, the counter's front edge's middle
  and heading, and the humanoid's start."""
  k: int
  runners: object
  cabinet: tuple[float, float, float]
  counter: tuple[float, float, float]
  start: tuple[float, float, float]

  def attach(self, spec) -> None:
    from pluggybot.activity import drawer as dr
    dr.attach_cabinet(spec, self.runners, self.cabinet[:2], self.cabinet[2], z=COUNTER_H)


#: `drawer_set_out(k)`'s stream is this and k (`_drawn`).
DRAWER_SEED = 5251


def drawer_set_out(k: int) -> DrawerSetOut:
  """Drawer `k` (seeded): its knob where a chest's knob stands
  (`KNOB_AHEAD`, `KNOB_LEFT`), the cabinet turned up to `TURNED`, the
  counter's edge `EDGE_BACK` ahead of the cabinet's front."""
  from pluggybot.activity import drawer as dr
  knob, yaw, _ = _drawn(k, DRAWER_SEED)
  cab = _placed(knob, dr.KNOB_AT, yaw)
  s = math.sin(yaw)
  front = min(cab[0] - s * dy for dy in (-dr.BOX_W / 2, dr.BOX_W / 2))
  return DrawerSetOut(k=k, runners=dr.draw(k), cabinet=cab,
                      counter=(front - EDGE_BACK, knob[1], 0.0), start=(START[0], START[1], 0.0))


@dataclass(frozen=True)
class DoorSetOut:
  """Door k on its cupboard on the counter (#525): its hidden parameters
  (`door.draw(k)`), the cupboard's frame, the counter's front edge's middle
  and heading, and the humanoid's start."""
  k: int
  hinge: object
  cupboard: tuple[float, float, float]
  counter: tuple[float, float, float]
  start: tuple[float, float, float]

  def attach(self, spec) -> None:
    from pluggybot.activity import door as dr
    dr.attach_cupboard(spec, self.hinge, self.cupboard[:2], self.cupboard[2], z=COUNTER_H)


#: `door_set_out(k)`'s stream is this and k (`_drawn`); how far its cupboard is turned
#: at most, rad -- its door swings toward the robot -- and where its lever's
#: tip stands, this far ahead of the soles' middle and this far to its
#: left, m: the tip comes 7 cm nearer and 9 cm to the left as the door
#: opens 50 deg, and from a chest knob's 0.33-0.37 ahead, ajar 5 deg, no
#: posture put the grip on it again (6 doors of 8); 0.12-0.13 left, the
#: press ran out of reach 1-3 cm down (#525).
DOOR_SEED, DOOR_TURNED = 5253, math.radians(8.0)
DOOR_AHEAD, DOOR_LEFT = (0.40, 0.44), (0.15, 0.20)


def door_set_out(k: int) -> DoorSetOut:
  """Door `k` (seeded): its lever's tip `DOOR_AHEAD` and `DOOR_LEFT`, the
  cupboard turned up to `DOOR_TURNED`, the counter's edge `EDGE_BACK` ahead
  of the cupboard's front."""
  from pluggybot.activity import door as dr
  tip, yaw, _ = _drawn(k, DOOR_SEED, DOOR_AHEAD, DOOR_LEFT, DOOR_TURNED)
  box = _placed(tip, dr.KNOB_AT, yaw)
  s = math.sin(yaw)
  front = min(box[0] - s * dy for dy in (-dr.BOX_W / 2, dr.BOX_W / 2))
  return DoorSetOut(k=k, hinge=dr.draw(k), cupboard=box,
                    counter=(front - EDGE_BACK, tip[1], 0.0), start=(START[0], START[1], 0.0))


@dataclass(frozen=True)
class CubeSetOut:
  """Cube k on the counter and the mat beside it (#525): its hidden
  parameters (`cube.draw(k)`), where the cube rests (x, y, its turn) and
  where the mat lies (x, y), the counter's front edge's middle and heading,
  and the humanoid's start."""
  k: int
  load: object
  cube: tuple[float, float, float]
  mat: tuple[float, float]
  counter: tuple[float, float, float]
  start: tuple[float, float, float]

  def attach(self, spec) -> None:
    from pluggybot.activity import cube as cb
    x0 = min(self.cube[0], self.mat[0]) - BOARD_MARGIN
    x1 = max(self.cube[0], self.mat[0]) + BOARD_MARGIN
    y0 = min(self.cube[1], self.mat[1]) - BOARD_MARGIN
    y1 = max(self.cube[1], self.mat[1]) + BOARD_MARGIN
    board = spec.worldbody.add_body(name="cube_board", pos=[(x0 + x1) / 2, (y0 + y1) / 2, COUNTER_H])
    board.add_geom(name="cube_board", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[(x1 - x0) / 2, (y1 - y0) / 2, BOARD_H / 2], pos=[0, 0, BOARD_H / 2],
                   rgba=[0.55, 0.45, 0.36, 1])
    cb.attach_props(spec, self.load, self.cube[:2], self.mat, self.cube[2], z=COUNTER_H + BOARD_H)


#: The cube and the mat lie on a board on the counter, this high, m, and
#: this much wider than them all round: on the counter's own top, 8 cm under
#: a chest's knob, nothing reached the mat 0.34-0.40 m ahead and only the
#: left hand tilted 45 deg the cube (#525; SimNotes, "The humanoid stands,
#: reaches and grasps": its envelope reaches 0.35-0.55 m ahead at 0.80-0.94).
BOARD_H, BOARD_MARGIN = 0.06, 0.08
#: `cube_set_out(k)`'s stream is this and k (`_drawn`); where the cube rests -- this
#: far ahead of the soles' middle and this far to its left, m -- and how far
#: it is turned at most, rad; where the mat lies from it: this far to its
#: left and this far further ahead, m -- and how far behind the counter's
#: edge the nearer of the two lies, m. IN SIGHT AND OUT OF REACH WHERE THE
#: ROBOT STANDS, so it STEPS to its work (`step`): standing put, its sight
#: and its reach boxed a cube into 4 cm -- nearer than 0.29 m the head
#: camera did not see it (5 of 8), past 0.34 the carry 6-10 cm ahead left
#: every posture (6 of 8) (#525).
CUBE_SEED, CUBE_AHEAD, CUBE_LEFT, CUBE_TURNED = 5255, (0.40, 0.46), (0.13, 0.17), math.radians(30.0)
MAT_LEFT, MAT_AHEAD, CUBE_EDGE = (-0.01, 0.01), (0.06, 0.08), 0.04


def cube_set_out(k: int) -> CubeSetOut:
  """Cube `k` (seeded): resting `CUBE_AHEAD` and `CUBE_LEFT`, turned up to
  `CUBE_TURNED`, the mat `MAT_LEFT` to its left and `MAT_AHEAD` further
  ahead, the counter's edge `CUBE_EDGE` in front of the nearer of the two."""
  from pluggybot.activity import cube as cb
  at, yaw, rng = _drawn(k, CUBE_SEED, CUBE_AHEAD, CUBE_LEFT, CUBE_TURNED)
  mat = (at[0] + float(rng.uniform(*MAT_AHEAD)), at[1] + float(rng.uniform(*MAT_LEFT)))
  front = min(at[0], mat[0]) - BOARD_MARGIN
  return CubeSetOut(k=k, load=cb.draw(k), cube=(at[0], at[1], yaw), mat=mat,
                    counter=(front - CUBE_EDGE, (at[1] + mat[1]) / 2, 0.0),
                    start=(START[0], START[1], 0.0))


def world(so, policy=None):
  """A set-out's world: the house with the humanoid standing at its start,
  the counter and the prop on it -- the chest (`SetOut`), or a check's
  (`DrawerSetOut`'s and the others', #525: its own `attach`). Returns the
  model, its data and the humanoid's body (`HumanoidBody`), its stance
  commissioned where it stands."""
  from pluggybot.body import body_for
  model = model_of(so)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  humanoid = body_for(model, data, policy=policy)
  humanoid.start_at(*so.start)
  return model, data, humanoid


def model_of(so):
  """A set-out's world compiled (`world`'s), no body built: what its scene
  is read off (#530)."""
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
  if isinstance(so, SetOut):
    ch.attach_chest(spec, so.lid, so.chest[:2], so.chest[2], z=COUNTER_H)
  else:
    so.attach(spec)
  spec.visual.map.znear = NEAR_M / spec.stat.extent
  return spec.compile()


def scene(model, cameras: dict | None = None) -> dict:
  """A set-out's world as the site draws it (`telemetry.scene`, named
  `WORLD`): the house's hints, and `cameras` -- `{role: camera}`, the head
  camera a person sees the robot's view through (#530)."""
  import json
  from pathlib import Path
  from pluggybot.telemetry.scene import scene_dict
  meta = json.loads(Path(HOME_META).read_text())
  return scene_dict(model, WORLD, meta=meta, cameras=cameras)


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
