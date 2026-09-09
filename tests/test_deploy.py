"""Serving-image guards (rooftop-media-2026 #20).

The deploy image is deliberately NOT the dev environment: it installs the
handful of packages `scripts/serve.py` actually needs rather than
uv-syncing a project whose torch is a ~3 GB CUDA wheel with no place on a
GPU-less server. That saving is only safe while two things stay true, and
both fail silently -- the suite stays green and the container dies on the
server, minutes into a mission, where nobody is watching:

  - the pins in `deploy/requirements-serve.txt` still match `uv.lock`, so
    the box runs the versions this repo is tested against;
  - the serve path still needs nothing the image leaves out. One `import
    torch` in a module the lifecycle touches is enough, and it costs
    nothing locally.

The second one is checked by RUNNING the serve path with those packages
made unimportable, not by scanning imports. Scanning is what the first
attempt did, and it passed while the first real container died at
`HubMission.__init__`: the apriltag detector is imported inside a function.
The forbidden set is derived from pyproject minus the image's own
requirements, so a new heavy dependency is covered without anyone
remembering to list it here.
"""

import os
import re
import subprocess
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQS = ROOT / "deploy" / "requirements-serve.txt"


def _norm(dist: str) -> str:
  """PEP 503 normalisation. Not cosmetic: pyproject says
  `pupil-apriltags` and the installed metadata says `pupil_apriltags`, and
  the first version of this file compared them raw -- which made the
  forbidden set below silently EMPTY, so the guard passed with the pin
  deleted. A test that cannot fail is decor."""
  return re.sub(r"[-_.]+", "_", dist).lower()


def _pins() -> dict[str, str]:
  """`{distribution: version}` from the image's requirements file."""
  out = {}
  for line in REQS.read_text().splitlines():
    line = line.split("#")[0].strip()
    if line:
      name, version = line.split("==")
      out[_norm(name.strip())] = version.strip()
  return out


def _lock_versions() -> dict[str, str]:
  lock = tomllib.loads((ROOT / "uv.lock").read_text())
  return {_norm(p["name"]): p["version"] for p in lock["package"]}


def _project_dependencies() -> set[str]:
  pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
  deps = pyproject["project"]["dependencies"]
  # "mujoco>=3.10.0" -> "mujoco"
  return {_norm(re.split(r"[<>=!~\[]", d)[0].strip()) for d in deps}


def test_pins_match_the_lockfile():
  """The server runs the versions the suite runs. A drifted pin is a
  different MuJoCo on the box than in every test that cleared it."""
  lock = _lock_versions()
  for dist, version in _pins().items():
    assert dist in lock, f"{dist} is pinned for deploy but not in uv.lock"
    assert version == lock[dist], (
      f"deploy pins {dist}=={version}, uv.lock has {lock[dist]}"
      " -- regenerate deploy/requirements-serve.txt")


def test_pins_are_project_dependencies():
  """A deploy pin that pyproject does not declare is a package nothing
  local ever exercises."""
  assert set(_pins()) <= _project_dependencies()


def _forbidden_modules() -> dict[str, tuple[str, ...]]:
  """`{top-level module: distributions}` for every project dependency the
  image leaves out -- derived, so a new heavy dependency is covered without
  anyone remembering to list it here."""
  omitted = _project_dependencies() - set(_pins())
  found = {mod: tuple(dists) for mod, dists in packages_distributions().items()
           if omitted.intersection(_norm(d) for d in dists)}
  # Every omitted dependency must resolve to at least one module, or the
  # set this test blocks is quietly smaller than the set the image omits.
  covered = {_norm(d) for dists in found.values() for d in dists}
  assert omitted <= covered, (
    f"no importable module found for {sorted(omitted - covered)}"
    " -- is the dev venv synced (`uv sync`)?")
  return found


# Run the REAL serve path with the omitted distributions made unimportable.
# A module-level import scan is not enough and was measured not to be: the
# detector is imported inside `hub.tags._shared_detector`, so the scan came
# back clean and the first container died at `HubMission.__init__`. Blocking
# the imports and then actually flying the robot catches a lazy import
# wherever it hides.
_BLOCKED_MISSION = """
import sys

class Blocked:
  \"\"\"Stand in for the packages the deploy image does not install.\"\"\"
  def __init__(self, names): self.names = names
  def find_spec(self, name, path=None, target=None):
    if name.split(".")[0] in self.names:
      raise ImportError(f"{name} is not installed in the deploy image")
    return None

sys.meta_path.insert(0, Blocked(set(sys.argv[1].split(","))))

import mujoco
from pluggybot.mind import overseer as ov
from pluggybot.lifecycle import (
  HubLifecycle, board_book, errands_for, world_config,
)
from pluggybot.telemetry.publisher import WsPublisher

cfg = world_config("home")
model = mujoco.MjModel.from_xml_path(cfg["model"])
data = mujoco.MjData(model)
book = board_book("home", state=None)
# The overseer is built and its client is resolved (issue #15): `anthropic`
# is a real runtime dependency of the serve path, and `Menu.for_world` drags
# in the stroke library and the drawing stack behind it. Enabled explicitly
# rather than off $PLUGGY_OVERSEER, so this exercises the path the deploy
# runs and not the one it happens to be configured for today.
boss, journal = ov.build("home", book, enabled=True)
assert boss.client is not None, boss.usage.errors
life = HubLifecycle(model, data, battery_wh=cfg["battery_wh"],
                    rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                    low_battery_wh=cfg["low_battery_wh"], world="home",
                    overseer=boss, journal=journal,
                    errands=errands_for("draw", "home", book), boards=book)
activities = cfg["activities"](model, data)
life.mission.step_hooks.append(activities.step_hook(model, data))
# Port 1 is nothing: the publisher's retry loop is the point, not a peer.
pub = WsPublisher(model, data, "ws://127.0.0.1:1",
                  model_name=cfg["model_name"],
                  status_fn=life.telemetry_status, grid=life.mission.grid,
                  activities=activities, boards=book)
life.mission.step_hooks.append(pub.step_hook)
try:
  life.mission.start_at(*cfg["start"])
  life.mission.start_discovery()
  life.mission._spin()          # lidar, tag detection, grid, frame building
finally:
  pub.close()
  life.mission.close()
print("ok")
"""


def test_serve_path_runs_without_the_packages_the_image_omits():
  """torch, ultralytics, SB3 and the apriltag GENERATOR belong to training,
  dataset generation and world generation -- none of which run on the
  serving box. If the mission stack starts needing one, this fails here
  rather than on the server, ten minutes into a mission, at night."""
  forbidden = _forbidden_modules()
  proc = subprocess.run(
    [sys.executable, "-c", _BLOCKED_MISSION, ",".join(sorted(forbidden))],
    cwd=ROOT, capture_output=True, text=True,
    env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "osmesa")})
  assert proc.returncode == 0, (
    "the serve path needs a package the deploy image does not install"
    " (deploy/requirements-serve.txt):\n" + proc.stderr[-2000:])
  assert "ok" in proc.stdout


@pytest.mark.parametrize("path", ["Dockerfile", "deploy/entrypoint.sh",
                                  "deploy/requirements-serve.txt"])
def test_deploy_files_present(path):
  assert (ROOT / path).is_file()


def test_entrypoint_is_executable():
  """COPYed into the image as-is; a lost +x bit is an exec-format failure
  on the server, not here."""
  assert (ROOT / "deploy" / "entrypoint.sh").stat().st_mode & 0o111


def _entrypoint_argv(tmp_path, **env) -> list[str]:
  """What `deploy/entrypoint.sh` hands `serve.py`, for a given environment.

  The REAL script under `sh`, with a fake `python` first on PATH that prints
  its argv -- the same discipline `_commit_guard` uses on the Dockerfile's
  guard line. A restated copy of the env-to-flag mapping would keep passing
  after somebody changed the real one, which is the only failure worth
  catching here.
  """
  bin_dir = tmp_path / "bin"
  bin_dir.mkdir(parents=True)
  fake = bin_dir / "python"
  fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
  fake.chmod(0o755)
  proc = subprocess.run(
    ["sh", str(ROOT / "deploy" / "entrypoint.sh")],
    cwd=ROOT, capture_output=True, text=True,
    env={"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}", **env})
  assert proc.returncode == 0, proc.stderr
  return proc.stdout.split()


def test_the_image_can_be_told_which_arm_to_fly(tmp_path):
  """Issue #142. The image takes ENVIRONMENT, never flags, so `$PLUGGY_ARM`
  has to reach `serve.py` through the entrypoint or the capability does not
  exist where it matters."""
  argv = _entrypoint_argv(tmp_path, PLUGGY_ARM="autonomous", PLUGGY_RUNG="A1")
  assert "--arm" in argv and argv[argv.index("--arm") + 1] == "autonomous"
  assert "--rung" in argv and argv[argv.index("--rung") + 1] == "A1"
  # ...and unset means the deployment behaves exactly as it did: no arm
  # named, so `$PLUGGY_OVERSEER` decides and the world is `guarded`.
  bare = _entrypoint_argv(tmp_path / "bare")
  assert "--arm" not in bare and "--rung" not in bare


def test_the_journal_is_not_trapped_behind_the_overseer_flag(tmp_path):
  """⚠ The trap issue #142 walked into. `--journal` and `--overseer-budget`
  used to be set only inside the `$PLUGGY_OVERSEER` branch -- so a
  `$PLUGGY_ARM` that turned a mind on WITHOUT that variable would have
  silently lost the robot's journal, on a volume where it is world state.

  Both are inert without an overseer (`overseer.build` returns `(None,
  None)` and neither path is read), so the safe shape is to pass them
  whenever they are set."""
  argv = _entrypoint_argv(tmp_path, PLUGGY_ARM="guarded",
                          PLUGGY_JOURNAL="/var/lib/pluggybot/journal.json",
                          PLUGGY_OVERSEER_BUDGET="30")
  assert "--overseer" not in argv, "the arm is what asked for a mind here"
  assert argv[argv.index("--journal") + 1] == "/var/lib/pluggybot/journal.json"
  assert argv[argv.index("--overseer-budget") + 1] == "30"


# ---- the build's identity (issue #132) ----------------------------------------


def _commit_guard() -> str:
  """The Dockerfile's own guard line, as `sh` would see it.

  Read out of the file rather than restated here: a copy would keep passing
  after somebody softened the real one, which is the whole failure mode
  this guard exists to prevent, one level up.
  """
  for line in (ROOT / "Dockerfile").read_text().splitlines():
    if line.startswith("RUN [ \"$PLUGGY_COMMIT\""):
      return line[len("RUN "):]
  raise AssertionError("the Dockerfile no longer guards $PLUGGY_COMMIT")


def test_the_image_refuses_to_build_without_a_commit():
  """A DEPLOYED CONTAINER MUST BE ABLE TO SAY WHICH BUILD IT IS (issue #132).

  `.git` is dockerignored, so `repo_commit()` inside the image can only read
  `$PLUGGY_COMMIT` -- and a default that silently stays `unknown` is not a
  smaller problem than no field at all: it produces months of observatory
  data nobody can attribute, from a container that looks perfectly healthy.
  So the BUILD is red instead, which is the osmesa smoke test's argument.

  Run under `sh` rather than through `docker build`, because docker is not
  on every box the suite runs on and a test that skips itself is decoration.
  What is actually asserted is the line the image executes, both ways.
  """
  guard = _commit_guard()
  unset = subprocess.run(["sh", "-c", guard], capture_output=True, text=True,
                         env={"PLUGGY_COMMIT": "unknown"})
  assert unset.returncode != 0, "an image with no commit built happily"
  assert "PLUGGY_COMMIT" in unset.stderr, \
    "the failure does not say what to pass"

  ok = subprocess.run(["sh", "-c", guard], capture_output=True, text=True,
                      env={"PLUGGY_COMMIT": "1a2b3c4"})
  assert ok.returncode == 0, "a real sha was rejected"


def test_the_baked_commit_is_what_the_sim_reports():
  """...and the other end of it: the variable the Dockerfile bakes is the
  one `repo_commit()` reads, in preference to a git call that cannot work
  inside the image."""
  dockerfile = (ROOT / "Dockerfile").read_text()
  assert "ARG PLUGGY_COMMIT" in dockerfile
  assert "ENV PLUGGY_COMMIT=$PLUGGY_COMMIT" in dockerfile

  from pluggybot.evaluation.record import COMMIT_ENV, repo_commit
  assert COMMIT_ENV == "PLUGGY_COMMIT"
  before = os.environ.get(COMMIT_ENV)
  try:
    os.environ[COMMIT_ENV] = "1a2b3c4"
    assert repo_commit() == "1a2b3c4"
    # ⚠ Blank is not a commit. An empty variable is the classic mis-deploy
    # ($PLUGGYWORLD_TOKEN's lesson), and reporting "" as an identity is
    # worse than falling back to the repo.
    os.environ[COMMIT_ENV] = "   "
    assert repo_commit() != "   "
  finally:
    if before is None:
      os.environ.pop(COMMIT_ENV, None)
    else:
      os.environ[COMMIT_ENV] = before
