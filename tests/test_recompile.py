"""The recompile seam (issue #168, slice C): a tool appears in a running
world, and every holder of the old world follows it.

No served world carries the built-tool rail (the quadruped's rack has none,
and its body refuses a recompile: #405; #407 re-homes the workshop on
legs), so what is pinned here is the seam's own rules -- off a spec, a stub
world, or the served world's refusal:

  1. THE SEAM REFUSES OUT LOUD, IN ITS ORDER: a world compiled without its
     spec, a world without the rail (the served world), and where a rail
     stands, a robot mid-errand or holding a module, or a bay off the rail;
     a hand-built module by name, wherever asked.
  2. A RETIRE TAKES A BUILT TOOL'S ACTUATORS WITH IT, and the ids after it
     shift while the state follows by name -- why every rebind resolves by
     name.
  3. NO STALE HOLDER, two ways. At run time: after a recompile, nothing
     reachable from ANY lifecycle in the world holds the old model or data,
     and the one physics loop steps the new world. Statically: every class
     in `src/` that assigns `self.model` or `self.data` defines `rebind`, or
     is on the roster below with a reason.
  4. A PAIR IS ONE WORLD (issue #315): one spec both lifecycles keep, one
     rack, and a refusal that names the robot in the way or whose tool it is.
"""

import ast
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot import tick
from pluggybot.body import STUB_WORLD, StubBody
from pluggybot.lifecycle import HubLifecycle, world_config
from pluggybot.procedure import axes
from pluggybot.rack import coupling
from pluggybot.rack.coupling import BUILT_STATION_YS, HUB_STATION_YS
from pluggybot.robot import SECOND, pair_model_name, world_spec
from pluggybot.workshop import seam, validate
from pluggybot.workshop.seam import SeamRefused
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_workshop import SCOOP

#: A second built tool, for the tests that need two.
SCOOP2 = {**SCOOP, "name": "scoop2"}
WORLD = "home_quad"

SRC = Path(__file__).parent.parent / "src" / "pluggybot"

#: Classes that assign `self.model` / `self.data` and do NOT rebind, each
#: with the reason. Adding one here is a decision made in the test.
TRANSIENT_HOLDERS = {
  # `self.model` is the LLM's id, not a MuJoCo model
  "mind/overseer.py:Overseer",
}


def _holders(root, depth: int = 4) -> list[tuple[str, object]]:
  """Every (path, object) reachable from `root` through attributes that is
  a MuJoCo model or data -- what a stale reference would be."""
  out, seen = [], set()

  def walk(obj, path, d):
    if id(obj) in seen or d > depth:
      return
    seen.add(id(obj))
    if isinstance(obj, (mujoco.MjModel, mujoco.MjData)):
      out.append((path, obj))
      return
    if isinstance(obj, (str, bytes, int, float, np.ndarray, type)) or obj is None:
      return
    items = []
    if isinstance(obj, dict):
      items = [(f"{path}[{k!r}]", v) for k, v in obj.items()]
    elif isinstance(obj, (list, tuple, set)):
      items = [(f"{path}[{i}]", v) for i, v in enumerate(obj)]
    elif hasattr(obj, "__dict__"):
      items = [(f"{path}.{k}", v) for k, v in vars(obj).items()
               if not k.startswith("__")]
    for p, v in items:
      walk(v, p, d + 1)
  walk(root, "life", 0)
  return out


@pytest.fixture(autouse=True)
def _clean_registries():
  before_a, before_s = set(axes.AXES), set(axes.SENSORS)
  yield
  for k in set(axes.AXES) - before_a:
    del axes.AXES[k]
  for k in set(axes.SENSORS) - before_s:
    del axes.SENSORS[k]


def _stub_world():
  """The stub's world compiled from a spec, as a served world is, its
  textures resolved where a served world's are."""
  spec = mujoco.MjSpec.from_string(STUB_WORLD)
  spec.modelfiledir = str(Path(world_config(WORLD)["model"]).parent.resolve())
  model = spec.compile()
  return spec, model, mujoco.MjData(model)


def _stub_pair():
  """Two stub lifecycles over ONE world and its spec, each told of the
  other, sharing one rack: `pair.build_pair`'s wiring with no robot."""
  spec, model, data = _stub_world()
  cfg = world_config(WORLD)
  a = stub_life(body=StubBody(model, data, rack=cfg["rack"],
                              grid_bounds=cfg["grid_bounds"]), spec=spec)
  b = stub_life(body=StubBody(model, data, handle=SECOND, rack=cfg["rack"],
                              grid_bounds=cfg["grid_bounds"]),
                spec=spec, handle=SECOND, robot_name="Rowan")
  b.rack_inventory = a.rack_inventory
  a.peers, b.peers = [b], [a]
  a.state = b.state = "DECIDE"
  return a, b


def _railed(life) -> HubLifecycle:
  """A rail stood in on a stub world with its spec: the seam's checks past
  the rail's own run as they would where one hangs."""
  life.has_built_rack = True
  life.state = "DECIDE"
  return life


# ---- 1. the refusals ---------------------------------------------------------


def test_the_served_world_has_no_rail_and_the_seam_says_so():
  cfg = world_config(WORLD)
  spec = world_spec(cfg["model"], body=cfg["body"])
  model = spec.compile()
  life = HubLifecycle(model, mujoco.MjData(model), realtime=False, world=WORLD,
                      spec=spec)
  assert life.has_built_rack is False
  with pytest.raises(SeamRefused, match="no built-tool rack"):
    life.hang_tool(validate.check(SCOOP), 0)


def test_the_seam_refuses_in_its_order():
  """Without the spec, nothing can hang; with it and a rail, a robot
  mid-errand or holding a module is in the way, and a bay off the rail is
  no bay."""
  tool = validate.check(SCOOP)
  with pytest.raises(SeamRefused, match="without its spec"):
    _railed(stub_life()).hang_tool(tool, 0)
  spec, model, data = _stub_world()
  life = _railed(stub_life(body=StubBody(model, data), spec=spec))
  life.state = "USE_TOOL"
  with pytest.raises(SeamRefused, match="you are mid-errand -- a tool is hung "
                                        "between errands"):
    life.hang_tool(tool, 0)
  life.state = "DECIDE"
  life.body.holding = "module_pen"
  with pytest.raises(SeamRefused, match="your fork holds module_pen: stow it first"):
    life.hang_tool(tool, 0)
  life.body.holding = None
  with pytest.raises(SeamRefused, match="no bay 3; the built-tool rail has 3"):
    life.hang_tool(tool, 3)
  assert life.model is model, "a refusal recompiled the world"


@pytest.mark.parametrize("module", seam.HAND_BUILT)
def test_a_hand_built_module_is_never_retired(module):
  """The originals are permanent (issue #277): the seam refuses them by
  name at the spec, and the lifecycle refuses before it touches the spec
  -- what every offered job is written against cannot be deleted by the
  robot it is offered to. Shown to fail by dropping `HAND_BUILT` from
  `seam.retire`: the pen goes, peg, face and all."""
  cfg = world_config(WORLD)
  spec = world_spec(cfg["model"], body=cfg["body"])
  there = spec.body(module) is not None
  with pytest.raises(SeamRefused, match="original modules"):
    seam.retire(spec, module)
  assert (spec.body(module) is not None) is there
  life = stub_life()
  with pytest.raises(SeamRefused, match="original modules"):
    life.retire_tool(module)
  assert (module in life.rack_inventory) is there


# ---- 2. the spec surgery ------------------------------------------------------


def test_a_retire_takes_the_actuators_and_the_state_still_follows_by_name(tmp_path):
  """MEASURED: deleting a built tool moves every body after it in the
  tree, and the moved body is exactly where it was. This is why every
  rebind resolves by name."""
  spec, _, _ = _stub_world()
  spec.modelfiledir = str(tmp_path)
  for raw, bay in ((SCOOP, 0), (SCOOP2, 1)):
    seam.attach(spec, validate.check(raw), bay, (0.0, 0.0), 0.0, model_dir=tmp_path)
  model = spec.compile()
  data = mujoco.MjData(model)
  for _ in range(500):
    mujoco.mj_step(model, data)
  second = model.body("module_scoop2").id
  mujoco.mj_forward(model, data)
  pose = data.xpos[second].copy()
  gone = seam.retire(spec, "module_scoop")
  assert gone["actuators"] == ["module_scoop_tilt"]
  model2, data2 = seam.recompile(spec, model, data)
  names = {mujoco.mj_id2name(model2, mujoco.mjtObj.mjOBJ_BODY, i)
           for i in range(model2.nbody)}
  assert "module_scoop" not in names and model2.actuator("module_scoop2_tilt").id >= 0
  assert model2.body("module_scoop2").id < second
  assert np.allclose(data2.xpos[model2.body("module_scoop2").id], pose)


def test_the_rail_is_past_the_first_racks_bays_in_one_index_space():
  """The rail's stations index `STATION_YS` after the first rack's
  (issue #277): the space `rack_inventory` and the bay switches use."""
  first = len(HUB_STATION_YS)
  assert coupling.STATION_YS[first:] == BUILT_STATION_YS
  for k in range(len(BUILT_STATION_YS)):
    assert coupling.built_bay_index(k) == first + k
    assert coupling.is_built_bay(first + k) and not coupling.is_built_bay(k)
  with pytest.raises(ValueError):
    coupling.built_bay_index(len(BUILT_STATION_YS))


def test_the_tag_png_is_written_once(tmp_path):
  p1 = seam.write_tag_png(seam.tag_id_for_bay(2), tmp_path)
  stamp = p1.stat().st_mtime_ns
  p2 = seam.write_tag_png(seam.tag_id_for_bay(2), tmp_path)
  assert p1 == p2 and p2.stat().st_mtime_ns == stamp
  assert [p.name for p in tmp_path.iterdir()] == [p1.name], "a temp file was left"


# ---- 3. no stale holder ------------------------------------------------------


def test_a_recompile_rebinds_every_lifecycle_and_the_loop_steps_the_new_world():
  """#315's blocker 1, on two stub lifecycles over one world: robot A
  recompiles while B holds still, both under ONE `tick.run_many` loop.

    - every lifecycle in the world is rebound, not just the one that ran
      the recompile (`_recompile`);
    - the loop's own `mj_step` reads the world off the stepper each step,
      so it steps the NEW one -- captured once at loop start, `data.time`
      stops advancing while every rebound holder reads the new world;
    - THE RUNTIME FENCE: nothing reachable from either lifecycle holds the
      old world (shown to fail by dropping `self.body.rebind` from
      `HubLifecycle.rebind`);
    - the wire says so once, in the scene fixture's shape: the sidecar's
      zones and the PAIR's name.
  """
  a, b = _stub_pair()
  old_model, old_data = a.model, a.data
  events = []
  a.on_event.append(events.append)

  def builder(n):
    yield from (a.body.STILL for _ in range(n))
    builder.ms = a._recompile(reason="tool")
    yield from (a.body.STILL for _ in range(n))

  def still(n):
    yield from (b.body.STILL for _ in range(2 * n))

  n = 50
  t0 = float(a.data.time)
  tick.run_many([(a.body.stepper, builder(n)), (b.body.stepper, still(n))])
  assert a.model is not old_model and a.data is not old_data
  assert b.model is a.model and b.data is a.data
  for life in (a, b):
    stale = [p for p, o in _holders(life) if o is old_model or o is old_data]
    assert stale == [], (life.root, stale)
  assert float(a.data.time) - t0 == pytest.approx(2 * n * a.model.opt.timestep)
  [changed] = [e for e in events if e.get("type") == "scene_changed"]
  meta = json.loads(Path(world_config(WORLD)["meta"]).read_text())
  assert changed["scene"]["model"] == pair_model_name(world_config(WORLD)["model_name"])
  assert changed["scene"]["zones"] and len(changed["scene"]["zones"]) == len(meta["zones"])


def test_every_holder_of_the_world_can_be_rebound():
  """THE STATIC FENCE. A class that stores `self.model` or `self.data` is a
  class that can go stale; it defines `rebind`, or it is on the roster
  above with a reason -- and a roster entry names a class that holds one,
  or it is stale. A grep would pass on a comment; the syntax tree does
  not."""
  missing, holders = [], set()
  for path in sorted(SRC.rglob("*.py")):
    tree = ast.parse(path.read_text(), filename=str(path))
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
      holds = False
      for node in ast.walk(cls):
        if not isinstance(node, ast.Assign):
          continue
        for t in node.targets:
          for el in (t.elts if isinstance(t, ast.Tuple) else [t]):
            if isinstance(el, ast.Attribute) and isinstance(el.value, ast.Name) \
                and el.value.id == "self" and el.attr in ("model", "data"):
              holds = True
      if not holds:
        continue
      key = f"{path.relative_to(SRC)}:{cls.name}"
      holders.add(key)
      has_rebind = any(isinstance(n, ast.FunctionDef) and n.name == "rebind"
                       for n in cls.body)
      if not has_rebind and key not in TRANSIENT_HOLDERS:
        missing.append(key)
  assert missing == [], f"holds the world and cannot be rebound: {missing}"
  assert TRANSIENT_HOLDERS <= holders, TRANSIENT_HOLDERS - holders


# ---- 4. a pair is one world (issue #315) --------------------------------------


def test_a_pair_is_compiled_from_a_spec_both_lifecycles_keep():
  """The deployed world is a pair, and `build_pair` once threw the spec
  away after `compile()` -- so `can_reshape` refused every build with "this
  world was compiled without its spec" before it reached any other rule
  (issue #315: 433 specified, 433 refused). One spec, one rack: two
  inventories would diverge the moment either robot hung anything. On the
  served world the refusal is the rail's."""
  from pluggybot.pair import build_pair
  a, b = build_pair(WORLD, errands=("none", "none"))
  assert a.spec is not None and a.spec is b.spec
  assert a.rack_inventory is b.rack_inventory
  with pytest.raises(SeamRefused, match="no built-tool rack"):
    a.can_reshape(0)


def test_the_seam_waits_for_the_other_robot_and_says_whose_errand_it_is():
  """A pair shares one world, so a recompile pulls it out from under BOTH
  routines. Every robot has to be between errands with its fork empty,
  and the refusal names the busy one: "wait" and "never" are different
  answers."""
  a, b = _stub_pair()
  b.state = "SWAP_RETURN"
  assert a.seam_busy().startswith("Rowan is busy (it is mid-errand): ")
  b.state = "DECIDE"
  b.body.holding = "module_pen"              # ...and a module on the OTHER fork
  assert a.seam_busy().startswith("Rowan is busy (its fork holds module_pen): ")
  b.body.holding = None
  assert a.seam_busy() == ""


def test_a_tool_the_other_robot_built_is_not_this_robots_to_take_or_retire():
  """One rail, two minds (issue #315): "the bay you name is taken" is a
  rule about a robot's OWN tools. The other robot's module is refused with
  whose it is, and the world is not recompiled behind the refusal."""
  a, b = _stub_pair()
  scoop = validate.check(SCOOP)
  a.rack_inventory[scoop.body] = coupling.built_bay_index(0)   # as `hang_tool`
  a.built[scoop.body] = scoop                                   # leaves them
  world = b.model
  with pytest.raises(SeamRefused, match="holds the scoop, which is not yours"):
    _railed(b).hang_tool(validate.check(SCOOP2), 0)
  with pytest.raises(SeamRefused, match="not yours to retire"):
    b.retire_tool(scoop.body)
  assert b.model is world and scoop.body in b.rack_inventory
