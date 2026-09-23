"""The physics bench (issues #215, #227): find an unknown mass and record it.

The second predicate-graded challenge (docs/Challenges.md §8), open in
METHOD: a workbench against the lab's east wall and two cubes on the floor
in front of it, a KNOWN mass and an UNKNOWN one. The cubes are the tower's
blocks (challenge/stack.py: 26 mm, `GRIP_SOLIMP`, a tag on every face --
ids 23 and 24, `tags.MASS_TAG_IDS`) because that cube is the one grasp the
claw has been proven on; only the mass differs. No errand does the job: the
robot writes the procedure (#166), may build the apparatus (#168), and
records what it found in its science record (#217, the `record` verb).

THE CRITERIA, written before the robot saw the job and saying nothing
about how (Challenges.md §3):

  1. The finding is ON THE RECORD: one line under `findings/mass_bench`
     whose quantity names the unknown, written AFTER the job was claimed.
     A line recorded before the claim is a memory, not a measurement, and
     the newest qualifying line is the one graded -- a corrected figure is
     a new `record`, as the findings rule says.
  2. It is a MASS: a number in kilograms (or grams, converted); a line with
     no number is not a finding and `record` refused it already.
  3. It is RIGHT to within `TOLERANCE` of the true mass, relative:
     `|reported - true| / true <= TOLERANCE`. The truth is read off the
     model's own `body_mass` at grading and nowhere else.
  4. No hold. The thing graded is a written record, and a record does not
     fall over; the tower's hold is a criterion of the tower.

THE TRUTH IS HIDDEN. The unknown is drawn from a BANK (`masses.json`,
`questions.json`'s pattern: rotated on the board's own sequence number,
nothing random) when the offer is made, written into the world's
`body_mass` -- the model AND the spec, so the workshop's recompile carries
it -- and kept in `Task.secret`, which reaches no context. `KNOWN_MASS_KG`
is the job's own statement and is told to the robot in the offer;
`UNKNOWN_MASS_KG` is only what the cube weighs between offers, and nothing
may read it as a truth. A bank entry within `TOLERANCE` of the known mass is
refused at load: the job would then be answerable without weighing anything.

THE HONEST SENSOR IS THE REAL PART'S: the lift is an igus lead screw under
a position servo, and what it pushes with at rest is the weight it carries
-- `read("lift.force")`, `procedure/axes.py`, with a load cell's noise.
MEASURED (the probe behind this module, 2026-09-20): a cube in the claw's
jaws, lifted and settled, moves the lift's force by exactly `dm * g` (to
1 mN across 0.05..0.40 kg; the empty claw reads 6.40 N), and the jaws hold
0.40 kg -- `MAX_KG` is that, and the bank stays under it.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco

from pluggybot.challenge.stack import block_xml
from pluggybot.rack.tags import MASS_TAG_IDS

#: The bench: EXACTLY ONE BOX (hint `table`, the height is the TOP), a
#: 1.2 x 0.6 m worktop at 0.8 m. Its long side runs along the wall.
BENCH_HALF = (0.30, 0.60, 0.40)
BENCH_RGBA = "0.55 0.47 0.38 1"

KNOWN_MASS_KG = 0.100
#: ⚠ A PLACEHOLDER, NOT THE ANSWER -- see the module docstring.
UNKNOWN_MASS_KG = 0.150

MASSES = ("mass_known", "mass_unknown")
KNOWN, UNKNOWN = MASSES
#: Where the two cubes START, from the bench's centre: on the floor a metre
#: in front of it (the bench faces -x, into the lab) and half a metre apart,
#: on no route the robot needs. The job is graded wherever they end up.
MASS_OFFSETS = ((-1.0, -0.5), (-1.0, 0.5))

#: Criterion 3: how far off, relative to the true mass, still passes. One
#: in ten: wide enough that a single careful reading on the lift's own
#: sensor clears it (a pair of readings with `axes.LOAD_NOISE_N` of noise
#: is ~4 g of scatter, 5 % of the lightest bank entry) and narrow enough
#: that the known mass, or any other bank entry, is a wrong answer.
TOLERANCE = 0.10
#: The heaviest cube the claw holds (measured, module docstring); the bank
#: refuses an entry above it, because a job that cannot be attempted is
#: not a challenge.
MAX_KG = 0.40
#: Where a finding for this job is recorded, and the word its quantity
#: must carry (criterion 1). The offer says both.
FINDINGS_TOPIC = "findings/mass_bench"
QUANTITY_WORD = "unknown"

BANK_PATH = Path(__file__).with_name("masses.json")
BANK_ENV = "PLUGGY_MASSES"
BANK_VERSION = 1


def bench_xml(bench_xy: tuple[float, float], prefix: str = "lab") -> str:
  """MJCF for the bench and its two masses, for a generator."""
  bx, by = bench_xy
  hx, hy, hz = BENCH_HALF
  out = (f'    <body name="{prefix}_bench" pos="{bx:.4f} {by:.4f} {hz:.4f}">\n'
         f'      <geom name="{prefix}_bench_geom" type="box" '
         f'size="{hx:.4f} {hy:.4f} {hz:.4f}" rgba="{BENCH_RGBA}"/>\n'
         f'    </body>')
  for name, (dx, dy), tag, mass in zip(MASSES, MASS_OFFSETS, MASS_TAG_IDS,
                                      (KNOWN_MASS_KG, UNKNOWN_MASS_KG)):
    out += "\n" + block_xml(name, bx + dx, by + dy, tag, mass=mass)
  return out


def bench_center(model, prefix: str = "lab") -> tuple[float, float]:
  bid = model.body(f"{prefix}_bench").id
  return (float(model.body_pos[bid][0]), float(model.body_pos[bid][1]))


# ---- the bank -----------------------------------------------------------------


@dataclass(frozen=True)
class MassBank:
  """The unknown masses the house may set out, in kg, in rotation order."""

  masses: tuple[float, ...]
  version: int = BANK_VERSION

  @classmethod
  def load(cls, path: str | os.PathLike | None = None) -> "MassBank":
    """Read the bank, refusing an entry the job could not be honest with:
    one the claw cannot lift, or one within `TOLERANCE` of the known mass
    (then "it weighs what the other one weighs" would be paid)."""
    p = Path(path) if path is not None else Path(os.environ.get(BANK_ENV) or BANK_PATH)
    doc = json.loads(p.read_text())
    masses = []
    for entry in doc.get("masses", ()):
      kg = float(entry["kg"])
      if not 0.0 < kg <= MAX_KG:
        raise ValueError(f"{p}: {entry.get('id', kg)} is {kg} kg -- the claw "
                         f"holds at most {MAX_KG} kg")
      if abs(kg - KNOWN_MASS_KG) / kg <= TOLERANCE:
        raise ValueError(f"{p}: {entry.get('id', kg)} is within {TOLERANCE:.0%} "
                         f"of the known {KNOWN_MASS_KG} kg -- a copied answer "
                         "would pass")
      masses.append(kg)
    if not masses:
      raise ValueError(f"{p}: the bank is empty")
    return cls(masses=tuple(masses), version=int(doc.get("version", 0)))

  def pick(self, n: int) -> float:
    """The nth mass, wrapping -- `QuestionBank.pick`'s rule: rotation on
    the board's own sequence number, deterministic, and it has to be."""
    return self.masses[int(n) % len(self.masses)]


_BANK: MassBank | None = None


def default_bank() -> MassBank:
  global _BANK
  if _BANK is None or os.environ.get(BANK_ENV):
    _BANK = MassBank.load()
  return _BANK


# ---- the world ----------------------------------------------------------------


def unknown_mass(model) -> float:
  """The TRUTH, off the model's own mass table. Read at grading and by the
  probe that sets it; shown to nothing."""
  return float(model.body_mass[model.body(UNKNOWN).id])


def set_unknown_mass(model, data, kg: float, spec=None) -> float:
  """Make the unknown cube weigh `kg`: `body_mass` and the inertia scaled
  with it, the derived constants recomputed, and the SPEC's geom too where
  one is held -- the workshop's recompile (#168) rebuilds from the spec,
  and a mass set on the model alone would revert to the placeholder the
  moment a tool was hung. ⚠ `mj_setConst` is given a scratch MjData: it
  writes `qpos0` into whatever data it is handed."""
  bid = model.body(UNKNOWN).id
  kg = float(kg)
  ratio = kg / float(model.body_mass[bid])
  model.body_mass[bid] = kg
  model.body_inertia[bid] *= ratio
  mujoco.mj_setConst(model, mujoco.MjData(model))
  if spec is not None:
    spec.body(UNKNOWN).geoms[0].mass = kg
  return kg


# ---- the record, and the grade -----------------------------------------------


_UNITS = {"kg": 1.0, "g": 0.001, "kilogram": 1.0, "kilograms": 1.0,
          "gram": 0.001, "grams": 0.001, "": 1.0}


def reported(findings, since_t: float) -> dict | None:
  """Criterion 1: the newest finding under `FINDINGS_TOPIC` whose quantity
  names the unknown and which was recorded AFTER `since_t` (the claim),
  as kilograms -- or None. `findings` is `ThoughtFiles.findings()`:
  parsed rows, oldest first, each with its `topic` and `t`."""
  hit = None
  for f in findings:
    if f.get("topic") != FINDINGS_TOPIC:
      continue
    if QUANTITY_WORD not in str(f.get("quantity", "")).lower():
      continue
    if round(float(f.get("t", 0.0)), 3) <= round(float(since_t), 3):
      continue
    scale = _UNITS.get(str(f.get("unit", "")).lower())
    if scale is None:
      continue
    hit = {"kg": float(f["value"]) * scale, "unit": f.get("unit", ""),
           "method": f.get("method", ""), "t": float(f.get("t", 0.0))}
  return hit


def sample_mass(life, errand, result: dict | None = None,
                before: dict | None = None) -> dict:
  """MEASURE, in the registry's shape (`scoring.SAMPLERS`): the truth off
  the world and the finding off the record -- and nothing off `result`,
  the robot's account of what it did. `errand.task_id` names the task,
  which says when the claim was, so a finding from before it does not
  count; the grade on the seam hands a namespace carrying just that."""
  task = life.tasks.get(getattr(errand, "task_id", "")) if life.tasks is not None else None
  since = (float(task.claimed_t) if task is not None and task.claimed_t is not None
           else 0.0)
  finding = reported(life.thoughts.findings(), since)
  try:
    truth = unknown_mass(life.model)
  except KeyError:
    truth = None
  return {"truth": truth, "reported": finding["kg"] if finding else None,
          "method": finding["method"] if finding else "",
          "recordedAt": finding["t"] if finding else None}


def eval_mass(m: dict) -> tuple[bool, dict, str]:
  """JUDGE, pure. Criteria 1-3 over the sampler's reading. The reason
  line says right or wrong and by how much only in the pass/fail sense --
  never the truth, never the error, which together with the reported
  value would give the truth away (`secret` on the row)."""
  truth = m.get("truth")
  reported_kg = m.get("reported")
  metrics = {"reported": reported_kg, "truth": truth, "error": None,
             "method": m.get("method") or "", "tolerance": TOLERANCE}
  if truth is None or truth <= 0:
    return False, metrics, "the bench was not read"
  if reported_kg is None:
    # WHAT THE GRADE READS, said at the one moment it matters (issue #264):
    # a deployed robot measured the cube, wrote it as a NOTE under another
    # topic, read "nothing since the claim" as a timing fault and learned
    # the wrong lesson. The offer already says all of this.
    return False, metrics, (
      "no finding for the unknown mass was recorded since the claim -- the "
      f"grade reads `record` lines under topic {FINDINGS_TOPIC.split('/', 1)[1]} "
      f"whose quantity says '{QUANTITY_WORD}', in kg or g; notes are not read")
  error = abs(float(reported_kg) - float(truth)) / float(truth)
  metrics["error"] = round(error, 4)
  ok = error <= TOLERANCE
  return ok, metrics, (f"recorded {float(reported_kg):.3f} kg for the unknown mass -- "
                       f"{'within' if ok else 'not within'} {TOLERANCE:.0%} of the true mass")
