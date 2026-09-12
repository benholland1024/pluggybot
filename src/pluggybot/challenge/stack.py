"""The three-block tower: the first predicate-graded CHALLENGE (issue #120).

A challenge is a job nobody wrote the robot a script for. It is graded the way
every task is graded -- code measures the world, never the robot's account of
it (economy/scoring.py) -- and what makes it a challenge rather than a task is
that its success criteria were written down BEFORE the robot saw it and say
nothing about method. docs/Challenges.md is the decision and the reasoning;
this module is the worked example.

THE CRITERIA, written first. A run passes when, at the moment the robot says
it is done AND again `HOLD_S` sim-seconds later:

  1. the three blocks form ONE tower `LAYERS` high: sorted by height, each
     block rests on the one below it --
  2. "rests on" meaning its centre sits one block pitch above the other's
     (within `PITCH_TOL_M`; a settled stack measures 25.9 mm, not 26.0)
  3. and no further than `REST_OFFSET_M` from it sideways, so it is ON the
     block and not beside it;
  4. the tower is FREE-STANDING: every block touches the floor or another
     block and nothing else. A block in the jaws is a block being held, and
     a tower the robot is holding up is not a tower.

Everything else is the robot's business: which block goes where, whether it
uses the claw at all, how long it takes. The grader knows only the blocks'
poses and contacts, and `sample_stack` reads nothing off the errand's report.

Perception ladder (TaskPattern.md §3): written for TIER 1. When the tower is
offered the blocks carry AprilTags and are found by the same `TagDetector`
that finds the rack; until then they are untagged props, because the grader
reads `xpos` and is tier-agnostic, and a 20 mm tag's decode range is measured
against the attempt (issues #58, #166), not before it.

The hold is the issue's own "still standing after 10 s": a tower whose centre
of mass is past its support passes every geometric check the instant it is
let go and is on the floor 0.2-0.3 s later (measured). Props are the
challenge's own, added through MjSpec so no committed world moves. ⚠ The
blocks carry `GRIP_SOLIMP` and it is load-bearing: on default soft contact a
2 + 4 mm lean crept over at 16.9 s (docs/Challenges.md §4, SimNotes "The grip
that leaked"); on the hard contact a tower stands iff its mass is over its base.
"""

import math

import mujoco
import numpy as np

from pluggybot.rack.coupling import GRIP_SOLIMP

#: The blocks. Same body as the hub world's `pickup` block, because the claw
#: was measured against that one (70 mm jaw span, 26 mm cube).
BLOCK_HALF = 0.013
BLOCK_MASS = 0.06
BLOCK_FRICTION = 1.2
BLOCKS = ("block_0", "block_1", "block_2")
#: Where they start: open floor east of the rack, a block's throw apart, on
#: the same line as the pickup block so the claw's approach runway is clear.
START_XY = ((1.10, 0.30), (1.10, 0.05), (1.10, -0.20))

# ---- the criteria (see the module docstring; numbered the same) -----------
LAYERS = 3                        # 1
PITCH_M = 2 * BLOCK_HALF          # 2: one block's edge...
PITCH_TOL_M = 0.005               #    ...25.9 mm measured on a settled stack
REST_OFFSET_M = BLOCK_HALF        # 3: centre within half an edge, sideways
HOLD_S = 10.0                     # 4: an overhung block is down inside 0.31 s
#: Only these may touch a block in a free-standing tower: the floor, and the
#: other blocks.
SUPPORT_GEOMS = ("floor",)


def add_blocks(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Add the three blocks to a world spec, at their start poses."""
  for name, (x, y) in zip(BLOCKS, START_XY):
    body = spec.worldbody.add_body()
    body.name = name
    body.pos = [x, y, BLOCK_HALF]
    body.add_freejoint()
    geom = body.add_geom()
    geom.name = f"{name}_box"
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [BLOCK_HALF] * 3
    geom.mass = BLOCK_MASS
    geom.friction = [BLOCK_FRICTION, 0.005, 0.0001]
    # Hard contact, or a stack creeps over (module docstring).
    geom.solimp = [float(v) for v in GRIP_SOLIMP.split()] + [0.9, 2.0]
    geom.rgba = [0.90, 0.60, 0.20, 1.0]
  return spec


def world_with_blocks(path: str = "models/hub_world.xml") -> mujoco.MjModel:
  return add_blocks(mujoco.MjSpec.from_file(path)).compile()


def place(model, data, name: str, xyz, yaw: float = 0.0) -> None:
  """Put a block somewhere, at rest. A test's hand, not the robot's."""
  jid = model.body(name).jntadr[0]
  adr, dof = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
  data.qpos[adr:adr + 3] = xyz
  data.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  data.qvel[dof:dof + 6] = 0.0


def _rests_on(upper: np.ndarray, lower: np.ndarray) -> bool:
  dz = float(upper[2] - lower[2])
  dxy = float(math.hypot(upper[0] - lower[0], upper[1] - lower[1]))
  return abs(dz - PITCH_M) <= PITCH_TOL_M and dxy <= REST_OFFSET_M


def measure(model, data) -> dict:
  """One snapshot of the world, as the grader sees it.

  Reads block poses off `xpos` and their contacts off `data.contact`, and
  nothing else -- not the claw's command, not the errand's result, not the
  robot's narration. `layers` is the tallest tower standing on the FLOOR:
  a chain of blocks from the lowest up, each resting on the last, so two
  blocks balanced on a third count as one layer, not three.
  """
  blocks = list(BLOCKS)
  pos = {b: np.array(data.xpos[model.body(b).id], dtype=float) for b in blocks}
  gids = {model.geom(f"{b}_box").id: b for b in blocks}
  support = {model.geom(g).id for g in SUPPORT_GEOMS}
  touching_other = set()
  for i in range(data.ncon):
    c = data.contact[i]
    for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
      if g in gids and other not in gids and other not in support:
        touching_other.add(gids[g])
  # The tower: the tallest chain that starts on the FLOOR, each block resting
  # on the one before it. Every floor block is tried as a base, so a spare
  # block lying beside the tower does not hide it.
  chain, offsets = [], []
  for base in blocks:
    if abs(float(pos[base][2]) - BLOCK_HALF) > PITCH_TOL_M:
      continue
    this, offs = [base], []
    while True:
      above = [b for b in blocks if b not in this and _rests_on(pos[b], pos[this[-1]])]
      if not above:
        break
      offs.append(math.hypot(*(pos[above[0]][:2] - pos[this[-1]][:2])))
      this.append(above[0])
    if len(this) > len(chain):
      chain, offsets = this, offs
  top = chain[-1] if chain else max(blocks, key=lambda b: pos[b][2])
  return {
    "t": float(data.time),
    "blocks": len(blocks),
    "layers": len(chain),
    "offsetMm": round(max(offsets) * 1000, 1) if offsets else 0.0,
    "heightMm": round((float(pos[top][2]) + BLOCK_HALF) * 1000, 1),
    "freeStanding": not touching_other,
    "touchedBy": sorted({model.geom(other).name for i in range(data.ncon)
                        for other in _foreign(data.contact[i], gids, support)}),
  }


def _foreign(contact, gids: dict, support: set) -> list:
  out = []
  for g, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
    if g in gids and other not in gids and other not in support:
      out.append(other)
  return out


def measurements(at_done: dict | None, settled: dict | None) -> dict:
  """The evaluator's input: the snapshot when the robot called it done, the
  snapshot after the hold, and how long the hold was. Absences stay absent --
  `eval_stack` turns them into a failure, not a pass."""
  done = at_done or {}
  end = settled or {}
  hold = (end["t"] - done["t"]) if "t" in done and "t" in end else None
  return {
    "layersAtDone": done.get("layers"),
    "freeStandingAtDone": done.get("freeStanding"),
    "layers": end.get("layers"),
    "freeStanding": end.get("freeStanding"),
    "offsetMm": end.get("offsetMm"),
    "heightMm": end.get("heightMm"),
    "holdS": round(hold, 1) if hold is not None else None,
    "touchedBy": sorted(set(done.get("touchedBy", ())) | set(end.get("touchedBy", ()))),
  }


def eval_stack(m: dict) -> tuple[bool, dict, str]:
  """The pre-declared predicate, applied twice: at the call and after the
  hold. Pure -- a dict of measurements in, a verdict out."""
  metrics = {
    "layersAtDone": m.get("layersAtDone"), "layers": m.get("layers"),
    "freeStandingAtDone": m.get("freeStandingAtDone"),
    "freeStanding": m.get("freeStanding"),
    "offsetMm": m.get("offsetMm"), "heightMm": m.get("heightMm"),
    "holdS": m.get("holdS"), "touchedBy": list(m.get("touchedBy") or ()),
  }
  # A missing measurement is not a passing one (scoring.py's rule): a run
  # that never sampled the world, or sampled it once, fails here.
  for key in ("layersAtDone", "layers", "freeStandingAtDone", "freeStanding",
              "holdS"):
    if metrics[key] is None:
      return False, metrics, "the tower was never measured"
  done, end = int(metrics["layersAtDone"]), int(metrics["layers"])
  hold = float(metrics["holdS"])
  if done < LAYERS:
    return False, metrics, f"{done} of {LAYERS} blocks stacked at the call"
  if not (metrics["freeStandingAtDone"] and metrics["freeStanding"]):
    touched = ", ".join(metrics["touchedBy"]) or "something"
    return False, metrics, (f"{LAYERS} blocks stacked, but {touched} was "
                            "holding the tower up")
  # A short hold is the HARNESS's fault, and it still fails: a verdict on a
  # tower nobody waited for is a verdict on a tower-shaped moment.
  if hold + 1e-6 < HOLD_S:
    return False, metrics, (f"measured {hold:.1f} s after the call, "
                            f"not the {HOLD_S:.0f} s the challenge asks for")
  if end < LAYERS:
    return False, metrics, (f"a tower of {LAYERS} at the call, {end} "
                            f"{'block' if end == 1 else 'blocks'} high "
                            f"{hold:.0f} s later -- it fell")
  return True, metrics, (f"{LAYERS} blocks stacked, free-standing, "
                         f"{float(metrics['heightMm']):.0f} mm tall, still "
                         f"standing {hold:.0f} s after the call "
                         f"({float(metrics['offsetMm']):.1f} mm of lean)")


def sample_stack(life, errand, result: dict, before: dict) -> dict:
  """The scoring seam's sampler: `before` is the snapshot taken when the
  robot called it done, and the world is measured again NOW -- after the
  hold. `result`, the errand's own account, is deliberately unread."""
  return measurements(before, measure(life.model, life.data))
