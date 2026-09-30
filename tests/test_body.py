"""The body interface (issue #380, `src/pluggybot/body.py`): the day loop
and the procedure layer reach the machine only through `Body`, and a stub
body carries a lifecycle's bookkeeping with no robot in the world at all.
(The quadruped implements every member: `tests/test_quadruped.py`.)"""

import ast
import inspect
from pathlib import Path

import pytest

from pluggybot import body as body_mod
from pluggybot.body import Body, StubBody, members
from pluggybot.lifecycle import HubLifecycle, world_config

SRC = Path(__file__).resolve().parents[1] / "src" / "pluggybot"

#: The machine's own modules -- the quadruped's (`legs/`: its model,
#: controller, reckoning and the scan its policy reads, #388), the navigation
#: it is built on (`navigator.py`, #387: a body's map, LIDAR and drive), the
#: coupling and the tools it works, and the one place a body is chosen
#: (`body.body_for`). EVERY OTHER MODULE in `src/` is on the loop's side of
#: the seam, a new one included: it reaches a robot's body only as
#: `<life>.body.<member>`, a member being a name `Body` declares.
BODY_SIDE = ("rack/", "tools/", "body.py", "legs/", "navigator.py")


def loop_side() -> list[str]:
  return sorted(rel for rel in (str(p.relative_to(SRC)) for p in SRC.rglob("*.py"))
                if not rel.startswith(BODY_SIDE))
#: ...and never the machine itself: its objects, its classes and the
#: criteria its coupling is read by. Its constants may be imported -- a
#: number is not a reach -- but nothing that acts on or senses the body.
BODY_ATTRS = frozenset({"mission", "swap", "odometry", "lidar", "tags", "walker",
                        "getup", "drivers"})
BODY_NAMES = frozenset({
  "QuadMission", "QuadBody", "QuadStepper", "PolicyDriver", "ArmDriver", "Drivers",
  "LegOdometry", "DepthCamera", "Lidar", "TagDetector", "module_power_contact",
  "bay_standoff", "charge_standoff", "swap_trace", "charge_trace", "gave_up"})


def reaches(source: str) -> list[str]:
  """Every way `source` reaches a body other than through `Body`: a name
  read off `.body` that the interface does not declare, one of the body's
  own objects as an attribute, or one of its classes or criteria by name."""
  known = members()
  bad = []
  for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Attribute):
      if (isinstance(node.value, ast.Attribute) and node.value.attr == "body"
          and node.attr not in known):
        bad.append(f"{node.lineno}: .body.{node.attr} is not a Body member")
      if node.attr in BODY_ATTRS:
        bad.append(f"{node.lineno}: .{node.attr} is the body's own")
    elif isinstance(node, ast.Name) and node.id in BODY_NAMES:
      bad.append(f"{node.lineno}: {node.id} is the body's own")
    elif isinstance(node, ast.ImportFrom):
      bad += [f"{node.lineno}: imports {a.name}, the body's own"
              for a in node.names if a.name in BODY_NAMES]
  return bad


def test_the_loop_and_the_procedure_verbs_reach_the_body_only_through_it():
  side = loop_side()
  assert {"lifecycle.py", "procedure/steps.py", "procedure/axes.py",
          "procedure/lang.py", "pair.py", "tick.py"} <= set(side)
  bad = [f"{rel}:{why}" for rel in side for why in reaches((SRC / rel).read_text())]
  assert not bad, "reaches past the body interface:\n  " + "\n  ".join(bad)


def test_the_fence_fails_on_every_way_past_it():
  """A fence that cannot fail is decor: each way past the interface is
  caught, and the way through it is not."""
  src = '''
from pluggybot.legs.body import QuadMission, bay_standoff
def f(self, life):
  self.body.drive_to_routine(1, 2)
  life.mission.swap.module_state("m")
  self.body.mission.pose
  life.body.go_to_routine(1, 2)
  module_power_contact(self.model, self.data, "m", "")
'''
  got = reaches(src)
  assert any("imports QuadMission" in b for b in got)
  assert any("imports bay_standoff" in b for b in got)
  assert any(".body.drive_to_routine is not a Body member" in b for b in got)
  assert any(".mission is the body's own" in b for b in got)
  assert any(".swap is the body's own" in b for b in got)
  assert any(".body.mission is not a Body member" in b for b in got)
  assert any("module_power_contact is the body's own" in b for b in got)
  assert not any("go_to_routine" in b for b in got)


def test_every_member_is_documented_where_it_is_declared():
  """The interface is named in one module and documented AT ITS DEFINITION:
  every method and property has a docstring, every attribute a `#:`
  comment on the line above it."""
  lines = inspect.getsource(Body).splitlines()
  undocumented = []
  for node in ast.parse(inspect.getsource(Body)).body[0].body:
    if isinstance(node, ast.FunctionDef):
      if ast.get_docstring(node) is None:
        undocumented.append(node.name)
    elif isinstance(node, ast.AnnAssign):
      if not lines[node.lineno - 2].strip().startswith("#:"):
        undocumented.append(node.target.id)
  assert not undocumented, undocumented
  assert len(members()) > 50        # a count, so a lost declaration shows


def test_the_stub_implements_every_member():
  stub = StubBody()
  missing = sorted(m for m in members() if not hasattr(stub, m))
  assert not missing, missing


def test_a_world_with_no_robot_of_ours_is_refused_a_body():
  """`body_for` is the one place a body is chosen: legs, by this robot's
  names -- and a world with none under them is refused, never given a
  body that is not there."""
  model, data = StubBody.world()
  with pytest.raises(ValueError, match="no body"):
    body_mod.body_for(model, data)


# ---- the stub: a lifecycle's bookkeeping with no robot ----------------------------


def stub_life(world: str = "home_quad", body: StubBody | None = None, **kw) -> HubLifecycle:
  """A lifecycle on a `StubBody`, in a world with no robot in it: for a test
  that needs the loop's bookkeeping -- the mind, the economy, the record --
  and no physics. `world` names the job catalogue, the energy table and the
  map's extent; the physics is the stub's floor."""
  cfg = world_config(world)
  body = body or StubBody(rack=cfg["rack"], grid_bounds=cfg["grid_bounds"])
  kw.setdefault("battery_wh", cfg["battery_wh"])
  kw.setdefault("low_battery_wh", cfg["low_battery_wh"])
  return HubLifecycle(body.model, body.data, realtime=False, world=world,
                      body=body, **kw)


def test_a_day_runs_on_the_stub_charge_and_all():
  """The loop's whole shape on a body that does nothing physical: the
  start, the explore, a charge trip -- the dock, the hold, the undock --
  and the end. Its seams tick as the stub's clock passes."""
  body = StubBody(draw_w=0.0)
  life = stub_life(body=body)
  life.battery.energy_wh = life.low_battery_wh * 0.5          # it must charge
  summary = life.run(start=(0.5, 3.0, 0.0), max_sim_time=120.0)
  assert summary["charge_cycles"] == 1 and summary["battery"] >= 0.9
  assert not body.docked and not body.on_charger, "undocked, and held no longer"
  assert summary["sim_time"] > 0.0 and summary["dead"] is None


def test_the_stub_fetches_holds_and_stows_as_a_test_says():
  body = StubBody()
  assert body.run(body.fetch_tool_routine(0.125, "module_pen")) == "arrived"
  assert body.tool_powered("module_pen") and body.module_state("module_pen")["on_fork"]
  assert body.seated_on("module_pen") == body.handle.root
  assert body.run(body.stow_tool_routine(0.125, "module_pen")) == "arrived"
  assert body.module_state("module_pen") == {"on_fork": False, "hung": True, "bay": 2}
  t0 = float(body.data.time)
  body.run(body.hold_routine(0.5))
  assert float(body.data.time) == pytest.approx(t0 + 0.5)
