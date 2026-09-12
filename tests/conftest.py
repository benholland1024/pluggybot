from pathlib import Path
import mujoco
import pytest


WORLD_MODEL_PATH = Path(__file__).parent.parent / "models" / "world.xml"
PLAYGROUND_MODEL_PATH = Path(__file__).parent.parent / "models" / "playground.xml"


@pytest.fixture(scope="module")
def world_model():
  return mujoco.MjModel.from_xml_path(str(WORLD_MODEL_PATH))

@pytest.fixture
def world_data(world_model):
  return mujoco.MjData(world_model)

@pytest.fixture(scope="module")
def playground_model():
  return mujoco.MjModel.from_xml_path(str(PLAYGROUND_MODEL_PATH))

@pytest.fixture
def playground_data(playground_model):
  return mujoco.MjData(playground_model)

# ---- endurance: flown proofs that no longer gate the default run ------------
# ⚠ OPT-IN, NOT A MARKER EXPRESSION (issue #158). `addopts = "-m 'not
# endurance'"` does not compose: pytest keeps the LAST `-m`, so the everyday
# `-m "not slow"` would silently switch these back ON. A flag plus a skip
# applied at collection cannot be defeated by another `-m`.
#
# What qualifies: a whole-mission run whose regressable claim is ALSO pinned
# by a fast unit test in the same file -- so the flown version proves the
# integration (the deferral produces a charge and a completed errand on real
# physics) rather than the rule, and the rule is what actually regresses.
# Ben, 2026-09-12: while the design is still moving, a generous pack is
# assumed to fund any single errand and a battery death costs a heart rather
# than the world, so paying twenty minutes per issue to fly those proofs is
# the wrong trade. Run them deliberately before a release or after touching
# the mission loop:  MUJOCO_GL=egl uv run pytest -q --endurance -m endurance
ENDURANCE_OPT = "--endurance"


def pytest_addoption(parser):
  parser.addoption(ENDURANCE_OPT, action="store_true", default=False,
                   help="also run the flown endurance proofs (minutes each)")


def pytest_collection_modifyitems(config, items):
  if config.getoption(ENDURANCE_OPT):
    return
  skip = pytest.mark.skip(reason=f"endurance proof: opt in with {ENDURANCE_OPT}")
  for item in items:
    if "endurance" in item.keywords:
      item.add_marker(skip)
