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
from pluggybot.hub import overseer as ov
from pluggybot.hub.lifecycle import (
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
