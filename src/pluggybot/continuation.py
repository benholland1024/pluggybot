"""A restart is a continuation (issue #345): the world's state on the volume,
and the world put back from it.

The served world's process ends -- a deploy, a crash, an out-of-memory
kill, the `PLUGGY_MAX_SIM_TIME` ceiling -- and the next one used to build
the world from its XML: both robots at their spawn poses on a full pack
with an empty map, every module back on its bay, and the job in hand
failed. The volume kept the robot's WRITING (thoughts, ledger, boards, the
task board) and nothing it had DONE. So a robot at 5 % in minute 59 woke
at 100 %, which is the buffer the sixth quality measures, refilled free.

This module keeps the rest: every body by name (robots, modules, cubes,
the mouse), the sim clock, each robot's pack, believed pose, occupancy grid,
height map, rack belief, clocks and whether it is dead, the world's
activities and the producer's schedule. `Keeper` writes it every
`SAVE_EVERY_S` and at shutdown; `restore` puts it back after `begin()` and
before the day routine runs. docs/Webserver.md, "A restart is a
continuation", is the story; the rules are here and in the tests.

⚠ SIM TIME CONTINUES. `data.time` comes back with the bodies, so every
absolute stamp -- the survival clock, the unminded clock, a dead robot's
stand-up timer, an offer's deadline -- means what it meant. Rebasing each
stamp to a fresh zero instead is how one missed stamp becomes a robot that
lies dead for an hour. `max_sim_time` is a RUN's budget from where it
starts (`HubLifecycle._day_routine`).

⚠ BODIES ARE MATCHED BY NAME, never by state-vector layout: a built tool
re-hung in another order, or a new build's world, reorders `qpos`. 11 of
room_hub's 24 joints are unnamed, so a joint's key is its body's name and
its index there. And the solver's WARM START is kept: MEASURED, without
it the same step diverges by 1e-12 and a parity check means nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from pluggybot.mission.mission import MissionAborted

#: Bumped when a field changes meaning; an older file loads, a newer one
#: refuses (the task board's rule, for the same volume).
FORMAT = 1
#: Sim seconds between saves on the physics seam. A crash rewinds the
#: bodies at most this far; the ledger, the boards and the task board are
#: written on every event and do not rewind. MEASURED, a save of the home
#: pair with both maps built is ~50 ms of the physics thread and 1.3 MB
#: (`write`): 0.1 % of the minute, on a pair that runs at 0.96x real time.
SAVE_EVERY_S = 60.0
#: Restores in a row that died before the next save. A world whose saved
#: state crashes the process would otherwise be restored into the same
#: crash for ever; at this many the next start is a fresh one, and says so.
MAX_RESUMES = 3


def fingerprint(model) -> str:
  """The WORLD's geometry, hashed: every body, joint and geom with its static
  pose and size. A new build that moved a wall answers differently, and then
  the bodies and the maps are not put back -- a robot restored into a wall
  that was not there when it was saved is not a continuation."""
  h = hashlib.sha256()
  for arr in (model.body_parentid, model.body_pos, model.body_quat,
              model.jnt_type, model.jnt_bodyid, model.geom_type,
              model.geom_bodyid, model.geom_size, model.geom_pos,
              model.geom_quat):
    h.update(np.ascontiguousarray(arr).tobytes())
  for i in range(model.nbody):
    h.update(model.body(i).name.encode() + b"\0")
  return h.hexdigest()[:16]


# ---- the physics, by name ----------------------------------------------------

_QLEN = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}
_VLEN = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}


def _joint_keys(model) -> list[tuple[str, int, int, int, int]]:
  """(key, qposadr, qlen, dofadr, vlen) for every joint: its name, else its
  body's name and its index among that body's joints."""
  out, seen = [], {}
  for j in range(model.njnt):
    body = int(model.jnt_bodyid[j])
    k = seen.get(body, 0)
    seen[body] = k + 1
    kind = int(model.jnt_type[j])
    key = model.joint(j).name or f"{model.body(body).name or body}#{k}"
    out.append((key, int(model.jnt_qposadr[j]), _QLEN.get(kind, 1),
                int(model.jnt_dofadr[j]), _VLEN.get(kind, 1)))
  return out


def _cat(parts) -> np.ndarray:
  return np.concatenate(parts) if parts else np.zeros(0)


def physics(model, data) -> tuple[dict, dict[str, np.ndarray]]:
  """MuJoCo's integration state, keyed by name: (index, arrays)."""
  joints = _joint_keys(model)
  q = [data.qpos[a:a + n] for _, a, n, _, _ in joints]
  v = [slice(d, d + n) for _, _, _, d, n in joints]
  acts = [(model.actuator(i).name, int(model.actuator_actadr[i]),
           int(model.actuator_actnum[i])) for i in range(model.nu)]
  mocap = [model.body(b).name for b in range(model.nbody)
           if model.body_mocapid[b] >= 0]
  index = {
    "t": float(data.time),
    "joints": [[k, n, vn] for k, _, n, _, vn in joints],
    "actuators": [[name, max(num, 0)] for name, _, num in acts],
    "mocap": mocap,
    "eq": [model.eq(i).name for i in range(model.neq)],
    "bodies": [model.body(i).name for i in range(model.nbody)],
  }
  arrays = {
    "qpos": _cat(q),
    "qvel": _cat([data.qvel[s] for s in v]),
    "warm": _cat([data.qacc_warmstart[s] for s in v]),
    "qfrc": _cat([data.qfrc_applied[s] for s in v]),
    "ctrl": np.array(data.ctrl, dtype=float),
    "act": _cat([data.act[a:a + n] for _, a, n in acts if n > 0]),
    "mocap_pos": np.array([data.mocap_pos[model.body(b).mocapid[0]] for b in mocap])
                 .reshape(-1, 3),
    "mocap_quat": np.array([data.mocap_quat[model.body(b).mocapid[0]] for b in mocap])
                  .reshape(-1, 4),
    "eq_active": np.array(data.eq_active, dtype=np.uint8),
    "xfrc": np.array(data.xfrc_applied, dtype=float),
  }
  return index, arrays


def put_physics(model, data, index: dict, arrays: dict) -> dict:
  """Put a `physics` state into a world, by name. Returns what matched: a
  body in the file and not the world is left out, one in the world and not
  the file keeps where the XML put it."""
  here = {k: (a, n, d, vn) for k, a, n, d, vn in _joint_keys(model)}
  qi = vi = 0
  matched = missed = 0
  for key, n, vn in index["joints"]:
    slot = here.get(key)
    if slot is not None and slot[1] == n and slot[3] == vn:
      a, _, d, _ = slot
      data.qpos[a:a + n] = arrays["qpos"][qi:qi + n]
      data.qvel[d:d + vn] = arrays["qvel"][vi:vi + vn]
      data.qacc_warmstart[d:d + vn] = arrays["warm"][vi:vi + vn]
      data.qfrc_applied[d:d + vn] = arrays["qfrc"][vi:vi + vn]
      matched += 1
    else:
      missed += 1
    qi, vi = qi + n, vi + vn
  ai = 0
  for i, (name, num) in enumerate(index["actuators"]):
    try:
      act = model.actuator(name)
    except KeyError:
      ai += num
      continue
    data.ctrl[act.id] = arrays["ctrl"][i]
    if num and int(model.actuator_actnum[act.id]) == num:
      adr = int(model.actuator_actadr[act.id])
      data.act[adr:adr + num] = arrays["act"][ai:ai + num]
    ai += num
  for i, name in enumerate(index["mocap"]):
    try:
      mid = int(model.body(name).mocapid[0])
    except KeyError:
      continue
    if mid >= 0:
      data.mocap_pos[mid] = arrays["mocap_pos"][i]
      data.mocap_quat[mid] = arrays["mocap_quat"][i]
  for i, name in enumerate(index["eq"]):
    if name and i < len(arrays["eq_active"]):
      try:
        data.eq_active[model.eq(name).id] = arrays["eq_active"][i]
      except KeyError:
        pass
  for i, name in enumerate(index["bodies"]):
    if name and i < len(arrays["xfrc"]):
      try:
        data.xfrc_applied[model.body(name).id] = arrays["xfrc"][i]
      except KeyError:
        pass
  data.time = float(index["t"])
  mujoco.mj_forward(model, data)
  return {"joints": matched, "jointsMissed": missed}


# ---- the file ----------------------------------------------------------------


@dataclass
class Snapshot:
  """One saved world: `meta` is JSON, `arrays` the numbers too big for it."""

  meta: dict
  arrays: dict[str, np.ndarray] = field(default_factory=dict)
  path: Path | None = None

  @property
  def t(self) -> float:
    return float(self.meta["physics"]["t"])

  def robot(self, root: str) -> tuple[dict | None, dict[str, np.ndarray]]:
    """One robot's state and its arrays (the `<root>/` keys, unprefixed)."""
    pre = f"{root}/"
    return (self.meta.get("robots", {}).get(root),
            {k[len(pre):]: v for k, v in self.arrays.items() if k.startswith(pre)})


def capture(lives, fingerprint_: str, resumes: int = 0) -> Snapshot:
  """The world as it stands, off the lifecycles that share it."""
  first = lives[0]
  index, arrays = physics(first.model, first.data)
  robots = {}
  for life in lives:
    state, own = life.kept_state()
    robots[life.root] = state
    arrays.update({f"{life.root}/{k}": v for k, v in own.items()})
  world = {"activities": ({a.name: a.kept_state() for a in first.activities}
                          if first.activities is not None else {}),
           "producer": (first.producer.kept_state()
                        if first.producer is not None else None)}
  meta = {"format": FORMAT, "world": first.world, "fingerprint": fingerprint_,
          "savedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "resumes": int(resumes), "physics": index, "robots": robots,
          "state": world}
  return Snapshot(meta, arrays)


def write(snap: Snapshot, path: str | os.PathLike) -> Path:
  """Atomically: a crash mid-write leaves the previous save, never half.
  An `.npz` `np.load` reads, deflated at level 1: MEASURED on the home
  pair's two mapped grids, 44 ms for 1.4 MB against `savez_compressed`'s
  level 6 at 175 ms for 1.05 -- and this is the physics thread's time."""
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  tmp = target.with_name(target.name + ".tmp")
  members = {"meta": np.array(json.dumps(snap.meta)), **snap.arrays}
  with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                       compresslevel=1) as zf:
    for name, arr in members.items():
      with zf.open(name + ".npy", "w", force_zip64=True) as fh:
        np.lib.format.write_array(fh, np.asanyarray(arr), allow_pickle=False)
  # on the disk before it replaces the last good one, or a host crash can
  # leave a renamed file with nothing in it
  with open(tmp, "rb") as fh:
    os.fsync(fh.fileno())
  os.replace(tmp, target)
  snap.path = target
  return target


def read(path: str | os.PathLike) -> Snapshot:
  with np.load(path, allow_pickle=False) as npz:
    meta = json.loads(str(npz["meta"]))
    arrays = {k: npz[k] for k in npz.files if k != "meta"}
  version = int(meta.get("format", 0))
  if not 1 <= version <= FORMAT:
    raise ValueError(f"{path}: world state format {version!r}, expected "
                     f"1..{FORMAT}")
  return Snapshot(meta, arrays, Path(path))


@dataclass
class Loaded:
  """What `load` found: the snapshot to carry on from, or None and why."""

  snapshot: Snapshot | None
  why: str = ""


def load(path: str | os.PathLike | None, world: str) -> Loaded:
  """The saved world to carry on from, or `Loaded(None, why)`.

  ⚠ COUNTS ITSELF BEFORE IT IS TRUSTED. `resumes` is written back up by one
  here and down to 0 by the next save, so a process that dies before
  saving again is seen by the next one: at `MAX_RESUMES` the world starts
  fresh rather than be put back into whatever killed it.
  """
  if path is None or not Path(path).exists():
    return Loaded(None)
  try:
    snap = read(path)
  except Exception as e:                    # noqa: BLE001 -- see below
    # ANY failure to read is a fresh start, never a crash: raised here it
    # kills the process before `MAX_RESUMES` can count, and `restart:
    # unless-stopped` loops on it (an empty or torn zip raises EOFError or
    # BadZipFile -- found in review). The next save replaces the file.
    return Loaded(None, f"the saved world could not be read "
                        f"({type(e).__name__}: {e})")
  if snap.meta.get("world") != world:
    return Loaded(None, f"the saved world is {snap.meta.get('world')!r}, "
                        f"not {world!r}")
  resumes = int(snap.meta.get("resumes", 0))
  if resumes >= MAX_RESUMES:
    return Loaded(None, f"the world was put back {resumes} times from the "
                        "same saved moment and never got past it")
  snap.meta["resumes"] = resumes + 1
  try:
    write(snap, path)
  except OSError as e:
    print(f"world state: could not count this restart ({e})")
  return Loaded(snap)


def restore(lives, snap: Snapshot) -> dict:
  """Put a saved world back into freshly built lifecycles: after every
  `begin()` (built tools re-hung, the bench's mass set) and before any day
  routine is driven.

  The clock always continues. The bodies, the poses and the maps are put
  back only where the world is the one they were saved in (`fingerprint`);
  elsewhere the robots start from their start poses carrying their packs,
  their clocks and their jobs, and are told why.
  """
  first = lives[0]
  same = snap.meta.get("fingerprint") == first.world_fingerprint
  if same:
    matched = put_physics(first.model, first.data, snap.meta["physics"], snap.arrays)
  else:
    first.data.time = snap.t
    mujoco.mj_forward(first.model, first.data)
    matched = {}
  for life in lives:
    # the run's budget counts from here: `begin` set it from 0, and a
    # robot the save never had starts fresh on this clock too
    life.max_sim_time += snap.t
  why = "" if same else "the world itself changed since then"
  state = snap.meta.get("state", {})
  if first.activities is not None:
    kept = state.get("activities", {})
    for act in first.activities:
      if act.name in kept:
        act.restore_kept(kept[act.name])
  if first.producer is not None and state.get("producer"):
    first.producer.restore_kept(state["producer"])
  restored = []
  for life in lives:
    kept, arrays = snap.robot(life.root)
    if kept is None:
      continue
    life.restore_kept(kept, arrays, in_place=same, why=why)
    restored.append(life.root)
  return {"inPlace": same, "t": snap.t, "robots": restored, **matched}


# ---- the keeper ----------------------------------------------------------------


class Keeper:
  """Saves the world on the physics seam every `every_s`, and on request.

  On the LAST robot's hooks: one world, one save, and a pair's step runs
  every robot's bookkeeping in order (`tick.run_many`), the world's
  activities with the first -- hooked on the first, the second robot was
  saved a step behind its own body. ⚠ A STOP IS RAISED ON THE SEAM, never
  from a signal handler:
  a SIGTERM lands between any two bytecodes -- inside a ledger write, half
  way through a death -- and a state saved from there is half one thing.
  `request_stop` only sets a flag; the next step boundary raises
  `MissionAborted`, which ends the day the way `stop_when` does.
  """

  def __init__(self, lives, path: str | os.PathLike,
               every_s: float = SAVE_EVERY_S) -> None:
    self.lives = list(lives)
    self.path = Path(path)
    self.every_s = float(every_s)
    self.fingerprint = self.lives[0].world_fingerprint
    self.saves = 0
    self.last_error: str | None = None
    self.last_save_s: float | None = None
    self.stop_requested = ""
    self._next: float | None = None
    for life in self.lives:
      life.continuing = True
    self.lives[-1].mission.step_hooks.append(self.step_hook)

  def busy(self) -> bool:
    """A robot mid stand-up is half stood up: its settle drive steps the
    sim with the pose moved and the pack not yet refilled."""
    return any(life._standing_up for life in self.lives)

  def step_hook(self) -> None:
    if self.busy():
      return
    if self.stop_requested:
      raise MissionAborted(f"stopping: {self.stop_requested}")
    t = float(self.lives[0].data.time)
    if self._next is None:
      self._next = t + self.every_s
    elif t >= self._next:
      self._next = t + self.every_s
      self.save()

  def request_stop(self, why: str) -> None:
    self.stop_requested = why

  def save(self) -> bool:
    """Write the world now. A failed save is reported and survived: the
    world must never stop because its disk did."""
    t0 = time.monotonic()
    try:
      write(capture(self.lives, self.fingerprint), self.path)
    except Exception as e:                  # noqa: BLE001 -- see docstring
      self.last_error = f"{type(e).__name__}: {e}"
      print(f"world state: could not save ({self.last_error})")
      return False
    self.saves += 1
    self.last_save_s = time.monotonic() - t0
    return True


def resumed_line(where: tuple[float, float] | None, frac: float,
                 why: str = "") -> str:
  """The History line a restart writes (issue #345): what it was, never a
  new day."""
  if why:
    return (f"the world restarted and could not put me back where I was "
            f"({why}); I start from the start pose with the pack at {frac:.0%}")
  x, y = where if where is not None else (math.nan, math.nan)
  return (f"the world restarted; I carried on from ({x:.1f}, {y:.1f}) with "
          f"the pack at {frac:.0%}")
