"""MuJoCo 3.10's island sleeping in every world `home_spec` and
`counter.model_of` build (#779; SimNotes, "Bodies at rest asleep"): a tree
that stays still for `mjMINAWAKE` steps sleeps with every tree it is
constrained to -- its contacts, kinematics, dynamics and integration
skipped -- until a touch, a write or a force wakes it, in that step.

⚠ EVERY ROBOT'S TREE IS `never` (`enable`, its root named by the builder):
a robot asleep would read its sensors stale. ⚠ The flag acts when the
`MjData` is made (static bodies' poses are computed then, and never again):
set before it, as the builders set it in the spec. A world compiled
elsewhere -- an instrument's own -- does not sleep.

⚠ THE SLEEP STATE IS STATE, and MuJoCo's own state calls leave it out: a
restore wakes every island unless it is put back (`kept`, `put`), and the
arrays MuJoCo derives from it rebuilt (`update`, MuJoCo's `mj_updateSleep`,
which the Python bindings do not expose)."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence

import mujoco
import numpy as np

#: `tree_asleep` of a tree fully awake: MuJoCo's `-(1 + mjMINAWAKE)`. Below
#: zero a tree is awake and counts up to -1, ready to sleep; at zero or above
#: it is asleep, the value the next tree of its island's cycle.
AWAKE = -(1 + mujoco.mjMINAWAKE)


def on(model: mujoco.MjModel) -> bool:
  """Does this world sleep?"""
  return bool(model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP))


def enable(spec: mujoco.MjSpec, roots: Iterable[str]) -> None:
  """Sleeping on in a world's spec at MuJoCo's own tolerance, the trees of
  `roots` (every robot's root body, by name) `never`; everything else keeps
  the compiler's policy (a tree an actuator moves never sleeps: the
  stations' plugs, on their balancers)."""
  spec.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  for name in roots:
    body = spec.body(name)
    if body is None:
      raise ValueError(f"no robot root {name!r} in this world")
    body.sleep = mujoco.mjtSleepPolicy.mjSLEEP_NEVER


#: The policies under which a tree never sleeps: a save's sleep is never put
#: on such a tree (`put`), as MuJoCo never puts one to sleep.
NEVER = (int(mujoco.mjtSleepPolicy.mjSLEEP_NEVER), int(mujoco.mjtSleepPolicy.mjSLEEP_AUTO_NEVER))


def update(model: mujoco.MjModel, data: mujoco.MjData) -> None:
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
  body = np.where(mocap, int(mujoco.mjtSleepState.mjS_AWAKE),
                  int(mujoco.mjtSleepState.mjS_STATIC)).astype(np.int32)
  dynamic = tree >= 0
  body[dynamic] = awake[tree[dynamic]]
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


def keys(model: mujoco.MjModel) -> list[str]:
  """Each tree's key, in the world's order: its root body's name, else
  `#<body id>` -- an id the world's fingerprint holds in place, since a
  restore puts nothing back into a changed world (`continuation.restore`)."""
  return [model.body(int(b)).name or f"#{int(b)}" for b in model.tree_bodyadr]


def kept(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[list[str], np.ndarray]:
  """The sleep state by name: each tree's key and its `tree_asleep`, a
  sleeping tree's the position of the next tree of its cycle in that list --
  as the next step will begin from it. ⚠ A write since the last step that
  wakes a sleeping tree (its pose moved, a velocity or a force set: an
  offer's set-out, a put-away) is only seen by the next step's kinematics,
  so it is applied here on a copy, as that step's `mj_kinematics` applies it
  (0.7 ms): saved as it stood, the tree came back asleep at the written pose,
  its kinematics forwarded to it, and nothing ever woke it."""
  asleep = np.array(data.tree_asleep, dtype=np.int32)
  if on(model) and (asleep >= 0).any():
    scratch = copy.copy(data)
    mujoco.mj_kinematics(model, scratch)
    asleep = np.array(scratch.tree_asleep, dtype=np.int32)
  return keys(model), asleep


def put(model: mujoco.MjModel, data: mujoco.MjData, names: Sequence[str],
        values: np.ndarray) -> None:
  """Put a `kept` sleep state back into a world, by name, and rebuild what
  MuJoCo derives from it. A tree the save lacks is awake; a sleeping cycle
  with a tree this world lacks, or one this world never lets sleep, is woken
  whole."""
  here = {k: i for i, k in enumerate(keys(model))
          if int(model.tree_sleep_policy[i]) not in NEVER}
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


def wake_all(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """Every tree awake, its countdown begun again."""
  data.tree_asleep[:] = AWAKE
  update(model, data)
