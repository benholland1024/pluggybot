"""The robot's model against the truth (issue #466, stage 3): OUR grading,
by code that knows a chest's hidden parameters. The robot never sees any of
it.

Each parameter is graded apart, never summed, against TWO references: the
world's own value and the best-expressible reference's (`chest.
reference_document`, the true numbers written in the language). The
reference against the world is "the language can't say it"; the model
against the reference, "the robot didn't find it":

  joint      which joint moves the lid (the nearest joint above the part
             nearest the true lid's middle), its type, its axis (deg off the
             true one) and its line (mm along the box and up, in the map:
             the robot's own frame), and its range
  lid        the part it moves: its middle (mm off) and its size (mm off)
  static     what gravity and any spring put on the lid at `ANGLES_DEG`,
             the handle hanging free, N*m shutting: a spring is never graded
             apart from gravity (they correlate 0.98-0.998 over the sweeps,
             #469)
  moment     the lid's first moment about its own hinge, kg*m: what the
             torques see of its mass (#469: never its mass)
  friction, damping   the lid's hinge's
  catch      its release, N, and what it holds the lid with, N*m (the
             release times its lever)
  assumed    the lid's mass and its second moment about its hinge, which
             the arm's torques cannot see: what the model ASSUMED

`reference_template` is the fitter's own instrument: the reference with its
lid's dynamics left unknown, whose fit is the fitter's error alone -- the
structure and the geometry true -- and, its hinge moved, what a geometry
error leaks into the dynamics (#466's decision 2).
"""

from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from pluggybot.activity import chest as ch
from pluggybot.evaluation import imagined as im
from pluggybot.imagination.compile import Imagined, PREFIX, compile_scene
from pluggybot.imagination.scene import Template, parse, parse_template

#: The static torque curve's angles, deg (#466).
ANGLES_DEG = (11.0, 34.0, 57.0)
#: The fitter's own ranges on the reference: past every drawn chest's
#: (`chest.RANGES`, `CATCH_RANGE`), each end at least half as far again.
#: ⚠ The hidden weight's mass is unknown too: left at its true number, the
#: fitter alone was handed a hidden parameter (the review); its place is
#: geometry, the reference's.
REFERENCE_RANGES = {"lid.mass": (0.05, 1.5), "lump.mass": (0.001, 0.3),
                    "hinge.stiffness": (0.0, 0.3), "hinge.slack": (0.0, 180.0),
                    "hinge.damping": (0.0, 0.4), "hinge.friction": (0.0, 0.25),
                    "hinge.release": (0.2, 8.0)}


def reference_template(lid: ch.Lid, setting: im.Setting, pin: float,
                       hinge_off_mm=(0.0, 0.0)) -> Template:
  """The best-expressible reference with its lid's dynamics unknown over
  `REFERENCE_RANGES` (its catch's release too, where it has one), its hinge
  moved `hinge_off_mm` (along the box, into it; and up)."""
  doc = ch.reference_document(lid, (setting.chest_x, setting.chest_y), setting.chest_yaw,
                              pin=pin)
  doc = copy.deepcopy(doc)
  c, s = math.cos(setting.chest_yaw), math.sin(setting.chest_yaw)
  for j in doc["joints"]:
    if j["id"] == "hinge":
      along, up = hinge_off_mm
      j["at"] = [j["at"][0] + along * c, j["at"][1] + along * s, j["at"][2] + up]
  for name, (lo, hi) in REFERENCE_RANGES.items():
    pid, field = name.split(".")
    if field == "release":
      for c_ in doc.get("catches", []):
        c_["release"] = {"between": [lo, hi]}
      continue
    for item in doc["parts"] + doc["joints"]:
      if item.get("id") == pid and field in ("mass", "stiffness", "slack", "damping",
                                             "friction"):
        item[field] = {"between": [lo, hi]}
  return parse_template(doc)


# ---- the worlds -------------------------------------------------------------------

def _subtree(model, body: int) -> list[int]:
  out = []
  for b in range(model.nbody):
    x = b
    while x not in (0, body):
      x = int(model.body_parentid[x])
    if x == body:
      out.append(b)
  return out


def _hang(model, data, joints: list[int], passes: int = 3) -> None:
  """Each hinge in `joints` turned to where gravity holds it still and
  pushes it back either way (a handle hanging free): a few passes over
  where one hangs from another, one where it does not."""
  for _ in range(passes if len(joints) > 1 else 1):
    for jid in joints:
      q, dof = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

      def f(phi: float) -> float:
        data.qpos[q] = phi
        mujoco.mj_forward(model, data)
        return float(data.qfrc_passive[dof] - data.qfrc_bias[dof])
      grid = np.linspace(-math.pi, math.pi, 145)
      vals = [f(p) for p in grid]
      root = None
      for i in range(len(grid) - 1):
        if vals[i] > 0.0 >= vals[i + 1]:
          lo, hi = float(grid[i]), float(grid[i + 1])
          for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > 0.0 else (lo, mid)
          root = (lo + hi) / 2
          break
      data.qpos[q] = 0.0 if root is None else root
  mujoco.mj_forward(model, data)


class Lid:
  """One world's lid as graded: its hinge (`joint`, a model joint id), the
  sign that opens it, its body and the hinges hanging below it."""

  def __init__(self, model, joint: int) -> None:
    self.model = model
    self.joint = joint
    lo, hi = model.jnt_range[joint]
    self.opens = -1.0 if (bool(model.jnt_limited[joint]) and hi <= 0.0 < -lo) else 1.0
    self.body = int(model.jnt_bodyid[joint])
    below = set(_subtree(model, self.body)) - {self.body}
    self.hanging = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in below
                    and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]

  @property
  def hinge(self) -> bool:
    return self.model.jnt_type[self.joint] == mujoco.mjtJoint.mjJNT_HINGE

  def static(self, angles_deg=ANGLES_DEG) -> list[float] | None:
    """What gravity and the spring put on the lid at each angle, N*m
    shutting, its handles hanging free; None for a lid on a slide."""
    if not self.hinge:
      return None
    m = self.model
    d = mujoco.MjData(m)
    q, dof = int(m.jnt_qposadr[self.joint]), int(m.jnt_dofadr[self.joint])
    out = []
    for a in angles_deg:
      d.qpos[:] = m.qpos0
      d.qvel[:] = 0.0
      d.qpos[q] = self.opens * math.radians(a)
      _hang(m, d, self.hanging)
      out.append(-self.opens * float(d.qfrc_passive[dof] - d.qfrc_bias[dof]))
    return out

  def line(self) -> tuple[np.ndarray, np.ndarray]:
    """The hinge's point and axis in the world (its drawn pose)."""
    m = self.model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return d.xanchor[self.joint].copy(), d.xaxis[self.joint].copy()

  def moments(self) -> tuple[float, float, float]:
    """(first moment, kg*m shutting; second moment, kg*m^2; mass, kg) of
    the lid and all it carries about its hinge, shut, its handles hanging
    free (as `static` hangs them: a handle drawn where it was seen, swung,
    put the reference's first moment 0.0008 kg*m off the world's)."""
    m = self.model
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    _hang(m, d, self.hanging)
    p, a = self.line()
    first, second = im.mass_moments(m, d, self.body, p, a)
    mass = float(sum(m.body_mass[b] for b in _subtree(m, self.body)))
    return -self.opens * first, second, mass


def lid_of(world: Imagined, true_middle) -> tuple[Lid | None, str | None]:
  """The model's lid: the nearest joint above the part nearest the true
  lid's middle (map), and that part's document id. (None, part) where no
  joint moves it."""
  m = world.model
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  best, part = None, None
  for b in range(1, m.nbody):
    name = m.body(b).name
    if not name.startswith(PREFIX):
      continue
    dist = float(np.linalg.norm(d.xipos[b] - np.asarray(true_middle)))
    if best is None or dist < best[0]:
      best, part = (dist, b), name[len(PREFIX):]
  if best is None:
    return None, None
  b = best[1]
  while b != 0:
    joints = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) == b]
    if joints:
      return Lid(m, joints[0]), part
    b = int(m.body_parentid[b])
  return None, part


def _true_middle(setting: im.Setting) -> np.ndarray:
  c, s = math.cos(setting.chest_yaw), math.sin(setting.chest_yaw)
  p = ch.HINGE + np.array([-ch.BOX_D / 2, 0.0, ch.LID_T / 2])
  return np.array([setting.chest_x + c * p[0] - s * p[1],
                   setting.chest_y + s * p[0] + c * p[1], p[2]])


def _deg_between(a, b) -> float:
  """The angle between two lines, deg (0-90: a line has no direction)."""
  cos = abs(float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))
  return math.degrees(math.acos(min(1.0, cos)))


def _off_line(p, axis, q, true_axis, yaw) -> tuple[float, float]:
  """Where a hinge's line sits off the true one, mm: along the box (+ in)
  and up, square to the true axis, at the true line's point `q`."""
  a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
  r = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
  closest = np.asarray(p) + (r @ a) * a                # the model's line nearest q
  off = closest - np.asarray(q)
  u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
  return 1000 * float(off @ u), 1000 * float(off[2])


def _catch(doc: dict, joint_id: str | None, scene_joint) -> dict | None:
  """The model's catch on its lid's joint: its release (N) and what it
  holds the lid with (N*m, the release times its point's lever)."""
  for c in doc.get("catches", []):
    if c.get("joint") == joint_id:
      from pluggybot.imagination.scene import lever
      lev = lever(scene_joint, np.asarray(c["at"], dtype=float) / 1000.0)
      return {"release": float(c["release"]), "holdNm": float(c["release"]) * lev}
  return None


def grade(document: dict, lid: ch.Lid, setting: im.Setting, pin: float) -> dict:
  """A model (its fitted document, the map's frame) graded against the
  world's chest and the best-expressible reference, both set at `setting`
  in the map, each handle starting `pin` rad on its pin (the module
  docstring). Absent is None, never 0."""
  world = im.truth_world(lid, setting, pin)
  ref = im.reference_world(lid, setting, pin)
  scene = parse(document)
  model = compile_scene(scene, carrying="module_claw")
  w_lid = Lid(world.model, world.model.joint("chest_hinge").id)
  r_lid = Lid(ref.model, ref.model.joint(f"{PREFIX}hinge").id)
  m_lid, part = lid_of(model, _true_middle(setting))
  q_w, a_w = w_lid.line()
  w_static, r_static = w_lid.static(), r_lid.static()
  w_first, w_second, w_mass = w_lid.moments()
  r_first, r_second, r_mass = r_lid.moments()
  truth = {"static": w_static, "moment": w_first, "second": w_second, "mass": w_mass,
           "friction": lid.friction, "damping": lid.damping, "release": lid.catch_n,
           "holdNm": lid.catch_n * ch.PIN_R, "rangeDeg": math.degrees(ch.HINGE_RANGE[1])}
  reference = {"static": r_static, "moment": r_first, "second": r_second, "mass": r_mass,
               "friction": lid.friction, "damping": lid.damping,
               "release": lid.catch_n if lid.catch_n > 0 else None,
               "holdNm": lid.catch_n * ch.PIN_R if lid.catch_n > 0 else None,
               "rangeDeg": math.degrees(ch.HINGE_RANGE[1])}
  out = {"truth": truth, "reference": reference, "lidPart": part, "model": None}
  if m_lid is None:
    return out
  jid = model.model.joint(m_lid.joint).name[len(PREFIX):]
  sj = scene.joint(jid)
  p, a = m_lid.line()
  along, up = _off_line(p, a, q_w, a_w, setting.chest_yaw)
  lp = scene.part(sj.part)
  w_size = (ch.BOX_D, ch.BOX_W, ch.LID_T)
  first, second, mass = m_lid.moments()
  catch = _catch(document, jid, sj)
  rng = None if sj.range is None else max(abs(v) for v in sj.range)
  out["model"] = {
    "joint": jid, "type": sj.type, "axisDeg": _deg_between(a, a_w) if sj.type == "hinge"
    else None, "hingeAlongMm": along, "hingeUpMm": up,
    "rangeDeg": None if rng is None else (math.degrees(rng) if sj.type == "hinge" else None),
    "lidMiddleMm": 1000 * float(np.linalg.norm(np.asarray(lp.pos) - _true_middle(setting))),
    "lidSizeMm": [1000 * (x - y) for x, y in zip(sorted(lp.size, reverse=True),
                                                 sorted(w_size, reverse=True))],
    "static": m_lid.static(), "moment": first, "second": second, "mass": mass,
    "friction": sj.friction, "damping": sj.damping,
    "release": None if catch is None else catch["release"],
    "holdNm": None if catch is None else catch["holdNm"]}
  return out


def errors(graded: dict) -> dict:
  """The graded model's errors, model less each reference, by parameter:
  `vsWorld`, `vsReference`, and the reference's own against the world
  (`language`). None wherever either side is absent."""
  m, t, r = graded.get("model"), graded["truth"], graded["reference"]

  def diff(a, b):
    if a is None or b is None:
      return None
    if isinstance(a, list):
      return [x - y for x, y in zip(a, b)]
    return a - b
  keys = ("static", "moment", "second", "mass", "friction", "damping", "release", "holdNm",
          "rangeDeg")
  return {"vsWorld": {k: None if m is None else diff(m.get(k), t.get(k)) for k in keys},
          "vsReference": {k: None if m is None else diff(m.get(k), r.get(k)) for k in keys},
          "language": {k: diff(r.get(k), t.get(k)) for k in keys}}


def ranges_graded(template_raw: dict, joint_id: str | None, part_id: str | None,
                  lid: ch.Lid) -> dict:
  """The author's ranges on the lid against the truth, each: does it hold
  the truth, and how wide it is (hi - lo, and hi / lo where lo is more than
  0). The lid's mass against its board and its hidden weight; its hinge's
  spring, slack, damping and friction; its catch's release. None where the
  author wrote no range there."""
  truth = {"mass": lid.lid_kg + lid.lump_kg, "stiffness": lid.stiffness,
           "slack": math.degrees(lid.springref), "damping": lid.damping,
           "friction": lid.friction, "release": lid.catch_n}
  out = {}

  def grade_one(v, true):
    if not (isinstance(v, dict) and "between" in v):
      return None
    lo, hi = (float(x) for x in v["between"])
    return {"lo": lo, "hi": hi, "holds": lo <= true <= hi, "width": hi - lo,
            "ratio": hi / lo if lo > 0 else None}
  for p in template_raw.get("parts", []):
    if p.get("id") == part_id:
      out["mass"] = grade_one(p.get("mass"), truth["mass"])
  for j in template_raw.get("joints", []):
    if j.get("id") == joint_id:
      for f in ("stiffness", "slack", "damping", "friction"):
        out[f] = grade_one(j.get(f), truth[f])
  for c in template_raw.get("catches", []):
    if c.get("joint") == joint_id:
      out["release"] = grade_one(c.get("release"), truth["release"])
  return out


# ---- over the set-outs: intervals ----------------------------------------------------

#: The bootstrap's resamples and seed.
BOOT_N, BOOT_SEED = 2000, 466


def median_interval(values, level: float = 0.95) -> dict | None:
  """The median of `values` with its bootstrap interval (`BOOT_N`
  resamples, a fixed seed), and how many there were; None for none."""
  v = np.asarray([x for x in values if x is not None], dtype=float)
  if not len(v):
    return None
  rng = np.random.default_rng(BOOT_SEED)
  boots = np.median(rng.choice(v, size=(BOOT_N, len(v)), replace=True), axis=1)
  lo, hi = np.percentile(boots, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
  return {"n": int(len(v)), "median": float(np.median(v)), "lo": float(lo), "hi": float(hi),
          "p90": float(np.percentile(v, 90))}


def share_interval(flags, level: float = 0.95) -> dict | None:
  """A share of True with its Wilson interval; None for none."""
  f = [bool(x) for x in flags if x is not None]
  n = len(f)
  if not n:
    return None
  k = sum(f)
  z = 1.959963984540054 if level == 0.95 else float(
    __import__("scipy.stats", fromlist=["norm"]).norm.ppf((1 + level) / 2))
  p = k / n
  den = 1 + z * z / n
  mid = (p + z * z / (2 * n)) / den
  half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return {"n": n, "k": k, "share": p, "lo": max(0.0, mid - half), "hi": min(1.0, mid + half)}
