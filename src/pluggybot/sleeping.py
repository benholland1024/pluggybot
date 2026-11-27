"""MuJoCo 3.10's island sleeping in every world `home_spec` and
`counter.model_of` build (#779; SimNotes, "Bodies at rest asleep"): a tree
that stays still for `mjMINAWAKE` steps sleeps with every tree it is
constrained to -- its contacts, kinematics, dynamics and integration
skipped -- until a touch, a write or a force wakes it, in that step.

⚠ EVERY ROBOT'S TREE IS `never` (`enable`): a robot asleep would read its
sensors stale. ⚠ The flag acts when the `MjData` is made (static bodies'
poses are computed then, and never again): set in the spec, before it
compiles. A world compiled elsewhere -- an instrument's own -- does not sleep.

⚠ THE SLEEP STATE IS STATE, and MuJoCo's own state calls leave it out: a
restore wakes every island unless it is put back (`kept`, `put`), and the
arrays MuJoCo derives from it rebuilt (`update`, MuJoCo's `mj_updateSleep`,
which the Python bindings do not expose)."""

from __future__ import annotations

import mujoco
import numpy as np

#: `tree_asleep` of a tree fully awake: MuJoCo's `-(1 + mjMINAWAKE)`. Below
#: zero a tree is awake and counts up to -1, ready to sleep; at zero or above
#: it is asleep, the value the next tree of its island's cycle.
AWAKE = -(1 + mujoco.mjMINAWAKE)


def on(model) -> bool:
  """Does this world sleep?"""
  return bool(model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP))


def robot_roots(spec) -> list:
  """Every robot's root body in a world's spec, by name: `pluggybot` and a
  second robot's `<prefix>pluggybot` (⚠ `MjsBody.parent` of the world body
  crashes MuJoCo 3.10's bindings: never walked)."""
  from pluggybot.telemetry.protocol import ROBOT_ROOT
  return [b for b in spec.bodies
          if b.name == ROBOT_ROOT or b.name.endswith("_" + ROBOT_ROOT)]


def enable(spec) -> None:
  """Sleeping on in a world's spec at MuJoCo's own tolerance, every robot
  `never`; everything else keeps the compiler's policy (a tree an actuator
  moves never sleeps: the stations' plugs, on their balancers)."""
  spec.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  for body in robot_roots(spec):
    body.sleep = mujoco.mjtSleepPolicy.mjSLEEP_NEVER


def update(model, data) -> None:
  """MuJoCo's `mj_updateSleep`: the arrays and counts it derives from
  `tree_asleep` -- which trees and bodies are awake, and the awake bodies',
  parents' and dofs' indices -- static bodies `mjS_STATIC` but a mocap
  body's tree, which is always awake. Read against MuJoCo's own
  (`tests/test_sleeping.py`)."""
  awake = (np.asarray(data.tree_asleep) < 0).astype(np.int32)
  data.tree_awake[:] = awake
  data.ntree_awake = int(awake.sum())
  tree = np.asarray(model.body_treeid)
  mocap = np.asarray(model.body_mocapid)[np.asarray(model.body_rootid)] >= 0
  static = np.where(mocap, int(mujoco.mjtSleepState.mjS_AWAKE),
                    int(mujoco.mjtSleepState.mjS_STATIC))
  body = np.where(tree < 0, static, awake[np.maximum(tree, 0)]).astype(np.int32)
  data.body_awake[:] = body
  ind = np.flatnonzero(body != int(mujoco.mjtSleepState.mjS_ASLEEP))
  data.body_awake_ind[:ind.size] = ind
  data.nbody_awake = int(ind.size)
  parent = np.flatnonzero(body[np.asarray(model.body_parentid)]
                          != int(mujoco.mjtSleepState.mjS_ASLEEP))
  parent = parent[parent > 0]
  data.parent_awake_ind[:parent.size] = parent
  data.nparent_awake = int(parent.size)
  dof_body = np.asarray(model.dof_bodyid)
  dof = np.flatnonzero((tree[dof_body] >= 0)
                       & (body[dof_body] == int(mujoco.mjtSleepState.mjS_AWAKE)))
  data.dof_awake_ind[:dof.size] = dof
  data.nv_awake = int(dof.size)


def keys(model) -> list[str]:
  """Each tree's key, in the world's order: its root body's name, else
  `#<body id>`."""
  return [model.body(int(b)).name or f"#{int(b)}" for b in model.tree_bodyadr]


def kept(model, data) -> tuple[list[str], np.ndarray]:
  """The sleep state by name: each tree's key and its `tree_asleep`, a
  sleeping tree's the position of the next tree of its cycle in that list."""
  return keys(model), np.array(data.tree_asleep, dtype=np.int32)


def put(model, data, names, values) -> None:
  """Put a `kept` sleep state back into a world, by name, and rebuild what
  MuJoCo derives from it. A tree the save lacks is awake; a sleeping cycle
  with a tree this world lacks is woken whole."""
  here = {k: i for i, k in enumerate(keys(model))}
  names, values = list(names), np.asarray(values, dtype=np.int64)
  asleep = np.full(model.ntree, AWAKE, dtype=np.int32)
  for i, name in enumerate(names):
    t = here.get(name)
    if t is None:
      continue
    v = int(values[i])
    if v < 0:
      asleep[t] = v
      continue
    cycle, j = [], i
    while True:                    # the island's cycle, all here or none
      cycle.append(j)
      j = int(values[j])
      if not 0 <= j < len(names) or len(cycle) > len(names):
        cycle = None
        break
      if j == i:
        break
    if cycle is not None and all(names[c] in here for c in cycle):
      asleep[t] = here[names[v]]
  data.tree_asleep[:] = asleep
  update(model, data)


def wake_all(model, data) -> None:
  """Every tree awake, its countdown begun again."""
  data.tree_asleep[:] = AWAKE
  update(model, data)
