"""Ladder A of issue #264: a hand-written solution exists for the tower,
and the verbs it stands on keep their rules.

The rules, each pinned without a mission (docs/Testing.md):

  1. `pick`/`place` are on the vocabulary, refuse without the claw, refuse
     a `pick` while holding and a `place` while empty, and fail out loud on
     a tag the eye cannot see -- every verdict measured, none reported.
  2. `place`'s ok is read off the WORLD after the retreat: a cube resting
     beside the target fails with the distance, one on it passes.
  3. `HubMission.spot`'s pixel-ray range: at a known tag height the cube's
     position is good to a few millimetres where PnP's range is not.
  4. `ClawTool.calibrate_from_body` is taken at the DEPLOYED reach (arm
     tucked or not, one offset), off the body and never the belief.
  5. `tuck_routine` leaves the claw clear of the lidar's front-stop cone --
     and the premise: at `APPROACH_LIFT` it is not.
  6. The solution parses and validates against the home world's facts, and
     nothing under `mind/` imports `challenge.solutions`.

The integration -- the tower stacked BY THE CLAW from the rack, graded on
the seam with the hold -- is the endurance flight at the bottom, and
`scripts/stack.py` is the same flight with a filmstrip.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot import tick
from pluggybot.challenge import solutions, stack
from pluggybot.lifecycle import world_facts
from pluggybot.mission.mission import HubMission
from pluggybot.procedure import lang
from pluggybot.procedure import steps as st
from pluggybot.rack.coupling import HUB_STATION_YS
from pluggybot.rack.swap import ARM_EXT, HubSwap
from pluggybot.robot import world_spec
from pluggybot.tools.gripper import (APPROACH_LIFT, MODULE_DRIVE_LIFT, ClawTool,
                                     PLACE_RETREAT_M)

SRC = Path(__file__).parent.parent / "src" / "pluggybot"


# ---- 1. the verbs' rules, with a fake claw ---------------------------------


_KEEP = object()


def _fake_claw(held=None, after=_KEEP):
  """A claw whose routines step nothing and whose `held` answers as told:
  `after` is what it holds once the routines have run."""
  state = {"held": held}

  def routine(*a, **kw):
    if after is not _KEEP:
      state["held"] = after
    return tick.result({"gripped_before_lift": True})
  return SimpleNamespace(held=lambda: state["held"], calibrate_from_body=lambda: (0.29, -0.05),
                         held_hang=lambda g: (0.0, 0.0, -0.016),
                         drive_over_routine=lambda *a, **kw: tick.result(True),
                         pick_up_routine=routine, place_on_routine=routine,
                         set_lift_routine=lambda *a, **kw: tick.result(None))


def _run(fn, life, args):
  return tick.run(SimpleNamespace(_step_once=lambda *a: None), fn(life, args))


def test_pick_and_place_are_verbs_with_a_measured_doc():
  assert {"pick", "place"} <= set(st.VERBS)
  assert st.VERBS["pick"].args == {"tag": st.VERBS["place"].args["tag"]}
  assert "ok when" in st.VERBS["pick"].doc and "ok when" in st.VERBS["place"].doc


def test_the_verbs_refuse_without_the_claw_and_say_so(monkeypatch):
  monkeypatch.setattr(st, "_claw", lambda life: None)
  life = SimpleNamespace()
  assert _run(st._pick, life, {"tag": 21}) == {"ok": False, "reason": "the claw is not on the fork"}
  assert _run(st._place, life, {"tag": 21}) == {"ok": False, "reason": "the claw is not on the fork"}


def test_pick_refuses_a_full_hand_and_place_an_empty_one(monkeypatch):
  monkeypatch.setattr(st, "_claw", lambda life: _fake_claw(held="block_1_box"))
  r = _run(st._pick, SimpleNamespace(), {"tag": 20})
  assert not r["ok"] and r["reason"] == "already holding block_1_box"
  monkeypatch.setattr(st, "_claw", lambda life: _fake_claw(held=None))
  r = _run(st._place, SimpleNamespace(), {"tag": 20})
  assert not r["ok"] and r["reason"] == "nothing in the jaws to place"


def test_a_tag_the_eye_never_decodes_fails_out_loud(monkeypatch):
  monkeypatch.setattr(st, "_claw", lambda life: _fake_claw())
  monkeypatch.setattr(st, "_spot_routine", lambda life, tag: tick.result(None))
  r = _run(st._pick, SimpleNamespace(mission=SimpleNamespace(pose=(0, 0, 0))), {"tag": 21})
  assert not r["ok"] and r["reason"] == "tag 21 is not a cube this robot can see from here"
  # ...and a tag that is not a cube at all never even looks (the claw only
  # knows the shape of the blocks and the bench's masses)
  assert tick.run(SimpleNamespace(_step_once=lambda *a: None),
                  st._spot_routine(SimpleNamespace(), 7)) is None


def test_pick_is_measured_off_the_jaws_not_the_command(monkeypatch):
  monkeypatch.setattr(st, "_spot_routine", lambda life, tag: tick.result(
    {"centre": (1.0, 0.0, 0.013), "lateral": 0.0, "range": 0.8, "half": 0.013,
     "layer": 0, "toward": (1.0, 0.0), "xyz": (1.0, 0.0, 0.013)}))
  life = SimpleNamespace(mission=SimpleNamespace(pose=(0.0, 0.0, 0.0)))
  monkeypatch.setattr(st, "_claw", lambda life: _fake_claw(held=None, after="block_1_box"))
  assert _run(st._pick, life, {"tag": 21})["ok"]
  monkeypatch.setattr(st, "_claw", lambda life: _fake_claw(held=None, after=None))
  r = _run(st._pick, life, {"tag": 21})
  assert not r["ok"] and r["holding"] is None


# ---- 2. place reads the world after the retreat ---------------------------


@pytest.fixture(scope="module")
def blocks_world():
  spec = stack.add_blocks(world_spec("models/hub_world.xml"))
  model = spec.compile()
  return model


def _place_verdict(model, monkeypatch, stacked: bool, jaws_empty: bool = True):
  data = mujoco.MjData(model)
  stack.place(model, data, "block_0", (1.0, 0.0, stack.BLOCK_HALF))
  stack.place(model, data, "block_1",
              (1.0, 0.0, 3 * stack.BLOCK_HALF + 0.0005) if stacked
              else (1.025, 0.0, stack.BLOCK_HALF))
  for _ in range(100):
    mujoco.mj_step(model, data)
  monkeypatch.setattr(st, "_claw", lambda life: _fake_claw(
    held="block_1_box", after=None if jaws_empty else "block_1_box"))
  monkeypatch.setattr(st, "_spot_routine", lambda life, tag: tick.result(
    {"centre": (1.0, 0.0, 0.013), "lateral": 0.0, "range": 0.8, "half": 0.013,
     "layer": 0, "toward": (-1.0, 0.0), "xyz": (0.987, 0.0, 0.013)}))
  life = SimpleNamespace(model=model, data=data,
                         mission=SimpleNamespace(pose=(0.0, 0.0, 0.0),
                                                 swap=SimpleNamespace(
                                                   _drive_until_routine=lambda *a, **k: tick.result(None),
                                                   _run_routine=lambda *a, **k: tick.result(None))))
  return _run(st._place, life, {"tag": 20})


def test_a_cube_resting_beside_the_target_fails_with_the_distance(blocks_world, monkeypatch):
  r = _place_verdict(blocks_world, monkeypatch, stacked=False)
  assert not r["ok"] and r["placed"] == "block_1_box"
  assert 24.0 < r["offsetMm"] < 27.5 and abs(r["aboveMm"]) < 1.0
  assert r["reason"].startswith("released, but it rests 2") and "not on it" in r["reason"]


def test_a_cube_resting_on_the_target_passes_and_a_held_one_does_not(blocks_world, monkeypatch):
  r = _place_verdict(blocks_world, monkeypatch, stacked=True)
  assert r["ok"] and abs(r["aboveMm"] - 26.0) < 1.0 and r["offsetMm"] < 1.0
  r = _place_verdict(blocks_world, monkeypatch, stacked=True, jaws_empty=False)
  assert not r["ok"] and r["reason"] == "the jaws did not let go"


# ---- 3. the pixel-ray range -------------------------------------------------


@pytest.fixture(scope="module")
def home_model():
  return world_spec("models/home_world.xml").compile()


def test_spot_at_a_known_height_beats_pnp_range(home_model):
  """From 0.8 m the block tag is ~24 px wide: PnP's range scatters by
  ±10-15 mm (half a pixel is 2 %), the ray through its centre pixel cut
  at the tag's known height does not. Both are the same decode."""
  from pluggybot.home import world as home
  data = mujoco.MjData(home_model)
  m = HubMission(home_model, data, viewer=None, realtime=False)
  bx, by = home.TOWER_XY[1]
  m.start_at(bx + 0.82, by, math.pi)
  data.ctrl[m.swap.lift_act] = 0.06
  for _ in range(400):
    mujoco.mj_step(home_model, data)
  # the body yaws ~0.6 deg settling off `start_at`, which the reckoner
  # cannot see; the claim is about the SENSOR, so the belief is re-synced
  # to the truth first (8.8 mm across at 0.8 m otherwise)
  q = m.swap.root_qadr
  yaw = 2 * math.atan2(float(data.qpos[q + 6]), float(data.qpos[q + 3]))
  r = m.swap.reckoner
  r.x = float(data.qpos[q]) - 0.08 * math.cos(yaw)
  r.y = float(data.qpos[q + 1]) - 0.08 * math.sin(yaw)
  r.theta = yaw
  pnp = m.spot(21)
  ray = m.spot(21, at_height=stack.BLOCK_HALF)
  assert pnp is not None and ray is not None, "the tag did not decode from 0.82 m"
  true = data.xpos[home_model.body("block_1").id]
  face = (float(true[0]) + stack.BLOCK_HALF, float(true[1]))   # the east face, seen from +x
  err_ray = math.hypot(ray["xyz"][0] - face[0], ray["xyz"][1] - face[1])
  err_pnp = math.hypot(pnp["xyz"][0] - face[0], pnp["xyz"][1] - face[1])
  assert err_ray < 0.004, (err_ray, err_pnp)
  assert abs(ray["xyz"][1] - face[1]) < 0.003
  assert err_ray <= err_pnp + 0.001
  assert ray["toward"][0] < -0.99      # the eye looks west at it


# ---- 4. and 5. the claw's own calibration and driving configuration -----


@pytest.fixture(scope="module")
def claw_on_fork():
  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)
  swap.place_at_standoff(HUB_STATION_YS[3])
  swap.pick()
  return model, data, swap


def test_the_body_calibration_is_one_offset_whatever_the_arm_is_doing(claw_on_fork):
  model, data, swap = claw_on_fork
  claw = ClawTool(model, data, swap)
  data.ctrl[swap.arm_act] = ARM_EXT
  swap._run(1.0, 0.0)
  out = claw.calibrate_from_body()
  believed = ClawTool(model, data, swap).calibrate()     # the reckoner is exact here
  data.ctrl[swap.arm_act] = 0.0
  swap._run(1.0, 0.0)
  tucked = claw.calibrate_from_body()
  assert abs(out[0] - tucked[0]) < 0.004 and abs(out[1] - tucked[1]) < 0.002, (out, tucked)
  assert abs(out[0] - believed[0]) < 0.010 and abs(out[1] - believed[1]) < 0.003
  data.ctrl[swap.arm_act] = ARM_EXT
  swap._run(1.0, 0.0)


def test_tucked_the_claw_is_outside_the_front_stop_cone_and_at_approach_lift_it_is_not(claw_on_fork):
  from pluggybot.behavior.navigation import FRONT_STOP_RANGE
  from pluggybot.perception.lidar import Lidar
  model, data, swap = claw_on_fork
  claw = ClawTool(model, data, swap)
  lidar = Lidar(model)

  def front_min():
    angles, ranges = lidar.scan(data)
    front = ranges[np.abs(angles) < 0.35]
    return float(front.min()) if front.size else math.inf
  swap.run(claw.tuck_routine())
  assert float(data.ctrl[swap.arm_act]) == 0.0
  assert abs(float(data.ctrl[swap.lift_act]) - MODULE_DRIVE_LIFT) < 1e-9
  assert front_min() > FRONT_STOP_RANGE
  # the premise: 36 mm lower, the claw's body sits in the cone and the
  # reflex would back the robot away from itself
  swap.run(claw.set_lift_routine(APPROACH_LIFT, settle=1.0))
  assert front_min() < FRONT_STOP_RANGE
  swap.run(claw.tuck_routine())


def test_place_on_retreats_a_grip_length():
  assert 0.25 <= PLACE_RETREAT_M <= 0.35


# ---- 6. the solution itself ------------------------------------------------


def test_the_tower_solution_compiles_against_the_home_world():
  facts = world_facts("home")
  for src in (solutions.TOWER, solutions.TOWER_AT_THE_ROW, solutions.WEIGH):
    lang.compile_procedure(src, facts)
  verbs = [s[1] for s in lang.parse(solutions.TOWER).body if s[0] == "verb"]
  assert verbs[0] == "fetch" and verbs[-1] == "stow"
  assert verbs.count("pick") == 2 and verbs.count("place") == 2
  weigh = lang.parse(solutions.WEIGH)
  assert ("verb", "pick", {"tag": ("num", 24)}, 17) in weigh.body and "lift.force" in solutions.WEIGH
  assert solutions.PROCEDURES == {"stack_tower": solutions.TOWER, "find_mass": solutions.WEIGH}
  from pluggybot.economy.tasks import KINDS
  assert all(KINDS[k].discharge == "procedure" for k in solutions.PROCEDURES)


def test_the_mind_never_imports_the_solutions():
  for path in (SRC / "mind").rglob("*.py"):
    for node in ast.walk(ast.parse(path.read_text())):
      names = ([a.name for a in node.names] if isinstance(node, ast.Import)
               else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
      assert not any("solutions" in n for n in names), path
  for path in (SRC / "mind").rglob("*.py"):
    assert "solutions" not in path.read_text(), path


# ---- the flown proof, on demand ------------------------------------------


def _from_the_rack(tmp_path, feature: str):
  import sys
  sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
  import solve as demo
  from pluggybot import lifecycle as lc
  from pluggybot.lifecycle import world_config
  life, _ = demo.build_life(False, str(tmp_path))
  if feature == "mouse":
    acts = lc.home_activities(life.model, life.data)
    life.mission.step_hooks.append(acts.step_hook(life.model, life.data))
    life.activities = acts
  m = life.mission
  m.start_at(*world_config("home")["start"])
  m.start_discovery()
  m._spin()
  source = {"tower": solutions.TOWER, "bench": solutions.WEIGH, "mouse": None}[feature]
  return life, demo.run(life, feature, source)


@pytest.mark.endurance
def test_the_tower_is_stacked_by_the_claw_from_the_rack_and_graded(tmp_path):
  """Ladder A, whole: `solutions.TOWER` run as a `procedure:` errand from
  the living-room rack, the claw fetched and stowed, `done`, the grade on
  the seam with its hold -- the verdict a robot would be paid for. ~170 s
  wall, behind --endurance: every rule it stands on is pinned above."""
  life, out = _from_the_rack(tmp_path, "tower")
  proc = out["errand"]["procedure"]
  assert proc["ok"] and proc["completed"] == proc["total"] == 15, proc
  assert out["errand"]["stowed"] and proc["toolsHung"]
  places = [s for s in proc["steps"] if s["verb"] == "place"]
  assert all(p["ok"] and p["offsetMm"] < stack.REST_OFFSET_M * 1000 for p in places)
  grade = out["grade"]
  assert grade["ok"] and grade["points"] > 0, grade
  assert grade["touchedDuringHold"] == []
  assert life.mission.swap.module_state("module_claw")["hung"]


@pytest.mark.endurance
def test_the_unknown_mass_is_weighed_on_the_lift_and_the_finding_graded(tmp_path):
  """Ladder A for the bench: `solutions.WEIGH` from the rack -- the claw
  to the lab, a tare, the cube lifted, `lift.force` read, set down, home
  -- the finding recorded off the procedure's `mass` and graded against
  the hidden truth. ~150 s wall; the sensor (`lift.force`), the grade and
  the verbs are pinned fast (tests/test_bench.py, above)."""
  life, out = _from_the_rack(tmp_path, "bench")
  proc = out["errand"]["procedure"]
  assert proc["ok"] and proc["completed"] == proc["total"], proc
  assert out["errand"]["stowed"]
  assert 0.05 < proc["locals"]["mass"] < 0.40
  grade = out["grade"]
  assert grade["ok"] and grade["points"] > 0 and grade["reported"] == round(proc["locals"]["mass"], 3), grade
  assert "truth" not in grade


@pytest.mark.endurance
def test_a_feed_act_reaches_the_cage_and_the_mouse_eats(tmp_path):
  """Ladder A for the mouse: the `care` action's own program flown from
  the rack; the act is judged off the cage's count (`landed`), and the
  mouse's state off the activity. ~45 s wall; `cage_program`'s legs and
  the activity's table are pinned fast (tests/test_mouse.py)."""
  life, out = _from_the_rack(tmp_path, "mouse")
  proc = out["errand"]["procedure"]
  assert proc["ok"] and proc["completed"] == proc["total"] == 9, proc
  care = out["care"]
  assert care["landed"] >= 1 and care["ok"]
  assert (care["before"], care["after"]) == ("resting", "eating")
  assert life.cage.state == "eating"
