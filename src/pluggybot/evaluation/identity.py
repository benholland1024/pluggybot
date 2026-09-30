"""Which build produced a stream (issue #132; docs/Evaluation.md §5).

The served world is an observatory: one continuous run, read by its rows.
An observation nobody can attribute is unusable, so the telemetry header
carries this block -- the commit, the data files' and the world's hashes,
the arm, the mind, the pack, the body -- and the observatory never pools
across it.
"""

import hashlib
import os
import re
import subprocess
from pathlib import Path

from pluggybot.economy import cadence, energy, metabolism, questions, scoring

#: The five files that each change the regime, resolved exactly as the sim
#: resolves them -- the env override wins -- so the header hashes the file
#: the run actually read.
DATA_FILES: dict[str, tuple[Path, str]] = {
  "rewards": (scoring.TABLE_PATH, scoring.TABLE_ENV),
  "cadence": (cadence.CADENCE_PATH, cadence.CADENCE_ENV),
  "energy": (energy.ENERGY_PATH, energy.ENERGY_ENV),
  "metabolism": (metabolism.METABOLISM_PATH, metabolism.METABOLISM_ENV),
  "questions": (questions.BANK_PATH, questions.BANK_ENV),
}

#: Where the serving image bakes its git sha (issue #132). `.git` is
#: dockerignored, so a container has no repo to ask -- and `repo_commit`
#: answering `unknown` in production is precisely the unattributable
#: observatory this variable exists to end.
COMMIT_ENV = "PLUGGY_COMMIT"

REPO = Path(__file__).resolve().parents[3]


def world_hash(world: str) -> str:
  """sha256 over the world's XML, every file it `<include>`s (recursively)
  and every asset it names by `file=` -- the WORLD is part of the regime.

  Learned from issue #110: the MSAA fix changed one attribute in the robot
  model and every scripted day after it is a different (and now
  repeatable) trajectory, while the five data files were untouched -- so a
  rollup keyed on the data files alone would have pooled pre-fix and
  post-fix runs under one series name.
  """
  from pluggybot.lifecycle import world_config
  root = REPO / world_config(world)["model"]
  digest = hashlib.sha256()
  seen: set[Path] = set()

  def visit(path: Path) -> None:
    if path in seen or not path.exists():
      return
    seen.add(path)
    text = path.read_bytes()
    digest.update(path.name.encode() + b"\0" + text + b"\0")
    for ref in re.findall(rb'file="([^"]+)"', text):
      visit(path.parent / ref.decode())

  visit(root)
  # ...and the body put in at load (issue #387), which the XML does not
  # name: the quadruped's generated model, its drivers' gains and the
  # policies it walks and gets up on -- a retrained policy is a new regime
  for rel in BODY_FILES[world_config(world)["body"]]:
    visit(REPO / rel)
  return digest.hexdigest()


#: The files a body put into a world at load is made of (`world_hash`).
BODY_FILES = {"quadruped": ("models/quadruped.xml", "models/quadruped.json",
                            "models/quadruped_policy.npz",
                            "models/quadruped_getup.npz")}


def body_identity(world: str) -> dict:
  """WHICH BODY a world's robots have (issue #387): its name and the sha256
  of each policy it runs, by role."""
  from pluggybot.lifecycle import world_config
  body = world_config(world)["body"]
  from pluggybot.legs.body import policies
  walk, getup = policies()
  return {"name": body, "policies": {"walk": walk.sha256, "getup": getup.sha256}}


def data_hashes(world: str) -> dict[str, str]:
  """sha256 of each data file as the sim would load it right now, plus the
  world's own hash: together they are the regime a series is defined by."""
  out = {}
  for name, (default, env) in DATA_FILES.items():
    path = Path(os.environ.get(env) or default)
    out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
  out["world"] = world_hash(world)
  return out


def repo_commit() -> str:
  """Which build this is, as a short sha.

  `$PLUGGY_COMMIT` wins over git because the SERVING IMAGE has no `.git`
  (it is in `.dockerignore`, and copying the history to name a commit
  would be an odd trade) -- the sha is baked at image build and the
  Dockerfile fails the build without one, so a deployed container can say
  what it is rather than reporting `unknown` for ever. Locally the
  variable is unset and this is the git call it always was.
  """
  baked = os.environ.get(COMMIT_ENV, "").strip()
  if baked:
    return baked
  try:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True,
                          cwd=Path(__file__).parent).stdout.strip()
  except Exception:  # noqa: BLE001 -- no git, no repo: still a header
    return "unknown"


def build_identity(world: str, *, arm: str, model: str | None = None,
                   backend: str | None = None, pack_wh: float | None = None,
                   reserve_wh: float | None = None,
                   deadline_s: float | None = None,
                   rung: str | None = None,
                   origin: str | None = None,
                   hashes: dict | None = None,
                   commit: str | None = None,
                   constitutions: dict | None = None,
                   eyes: str | None = None,
                   body: dict | None = None) -> dict:
  """WHICH BUILD produced a stream (issue #132; docs/Evaluation.md §5).

  The deployed world is an observatory rather than an experiment -- one
  uncontrolled continuous run whose numbers never enter a results table --
  but an observation nobody can attribute is not weaker data, it is
  unusable data: a week under one build and the week after it under
  another wear one name, and nothing separates them afterwards.
  """
  return {
    "commit": commit if commit is not None else repo_commit(),
    "dataHashes": dict(hashes if hashes is not None else data_hashes(world)),
    "arm": arm,
    # WHICH MIND, and it is two fields because they answer different
    # questions: `Qwen/Qwen3-4B-Instruct-2507` is the model and
    # `huggingface` is the road it took to get there -- the same id served
    # locally is a different regime (docs/Overseer.md §6).
    "model": model,
    "backend": backend,
    # The three world parameters that have each been shown to move
    # behaviour: the pack size (Evaluation.md §5, "an experimental
    # parameter, not a comfort setting"), the return-trip margin every
    # errand must leave behind, and the decision deadline -- which is not
    # a data file, so no hash catches it, and which decides how much of a
    # day the model decided at all (issue #117).
    "packWh": pack_wh, "reserveWh": reserve_wh, "deadlineS": deadline_s,
    # ...and WHICH RUNG, where there is a ladder (issue #142): A0 hides the
    # survival clock A1 restores, so two rungs are two regimes.
    #
    # ⚠ ABSENT rather than null on an arm with no ladder, which is the one
    # place this block departs from `model`/`backend`. Those answer a
    # question every arm has an answer to ("which mind" -- none, on
    # `scripted`); "which rung" is not a question `guarded` has an answer
    # to, and a `"rung": null` beside it invites a reader to look for a
    # ladder that does not exist. It also keeps a `guarded` header -- the
    # deployed world's -- byte-identical to the one #132 shipped.
    **({"rung": rung} if rung else {}),
    # ...and WHICH ORIGIN the agent's event map started from (issue #127),
    # on exactly the rung's terms: ABSENT where the arm has no map, so a
    # `guarded` header stays byte-identical to #132's and an `autonomous`
    # one flown at `none` stays byte-identical to #142's.
    **({"origin": origin} if origin and origin != "none" else {}),
    # ...and WHICH CONSTITUTION each robot was told it is (issue #263),
    # per robot ROOT -- `{root: {name, sha}}` -- because a pair may be
    # given two, and that pair is the experiment the library exists for.
    # The name is the library file, the sha its content, so an edit to a
    # file under the same name is a new period too. ABSENT where the
    # caller passed none, on the rung's terms: the fixtures and every
    # header before this carry no such block.
    **({"constitutions": {root: dict(c) for root, c in constitutions.items()}}
       if constitutions else {}),
    # ...and WHICH MODEL LOOKED (issue #275): the id that was handed the
    # pictures the robot took. Today the MIND'S OWN (the deployed model
    # takes an image on the request), so it reads the same as `model` --
    # and it is a field of its own because a captioner in front of a
    # text-only mind would be a different regime under the same `model`:
    # what the robot "saw" then depends on it. ABSENT where the arm cannot
    # look, on the rung's terms.
    **({"eyes": eyes} if eyes else {}),
    # ...and WHICH BODY (issue #387): the quadruped and the policies it
    # walks and gets up on (`body_identity`).
    **({"body": dict(body)} if body else {}),
  }
