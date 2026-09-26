"""The recompile seam (issue #168, slice C): a tool appears in a running
world, and every holder of the old world follows it.

What these hold down:

  1. THE SPEC PATH IS THE OLD PATH. A world compiled from its `MjSpec` is
     trajectory-identical to `MjModel.from_xml_path`, so keeping the spec
     costs nothing (both worlds, hashed after 500 steps).
  2. THE SEAM. `hang_tool` hangs the built tool in a bay of the BUILT-TOOL
     RAIL (issue #277) -- retiring a built tool already there, never one of
     the five originals -- recompiles with the state carried across,
     rebinds every holder, registers the verbs and emits `scene_changed`;
     measured on a bare lifecycle, not flown. The module then HANGS: it
     settles into the rail's trays and `module_state` reads it hung.
  3. NO STALE HOLDER, two ways. At run time: every object reachable from
     the lifecycle that holds an `MjModel` or `MjData` holds the NEW one.
     Statically: every class in `src/` that assigns `self.model` or
     `self.data` defines `rebind`, or is on the roster below with a reason.
     The static fence is what catches a holder added next month; the
     runtime walk is what catches one the lifecycle forgot to call.
  4. BETWEEN ERRANDS ONLY -- FOR EVERY ROBOT IN THE WORLD. Mid-errand,
     with a module on the fork, while the OTHER robot of a pair is
     mid-errand, or in a world compiled without the rail, the seam refuses
     out loud -- and a hand-built module is refused by name, wherever asked.
  5. A PAIR HANGS A TOOL (issue #315). Two lifecycles over one world: the
     robot that did not build is rebound with the one that did, the shared
     physics loop steps the NEW world, and the rail is the world's -- a bay
     the other robot's tool hangs in is refused with whose it is.
"""

import ast
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot import tick
from pluggybot.lifecycle import HubLifecycle, world_config
from pluggybot.procedure import axes
from pluggybot.rack import coupling
from pluggybot.rack.coupling import BUILT_STATION_YS, HUB_STATION_YS, RACK_HANG_X
from pluggybot.robot import world_spec
from pluggybot.workshop import seam, validate
from pluggybot.workshop.seam import SeamRefused
from test_workshop import SCOOP

#: A second built tool, for the tests that need two on the rail.
SCOOP2 = {**SCOOP, "name": "scoop2"}

SRC = Path(__file__).parent.parent / "src" / "pluggybot"

#: Classes that assign `self.model` / `self.data` and do NOT rebind, each
#: with the reason. Adding one here is a decision made in the test.
TRANSIENT_HOLDERS = {
  # built per errand inside a routine and never held by the lifecycle: a
  # recompile is refused mid-errand, so one never outlives its world
  "tools/drawing.py:PenPlotter", "tools/gripper.py:ClawTool",
  "tools/dispenser.py:SeedDispenser",
  # `self.model` is the LLM's id, not a MuJoCo model
  "mind/overseer.py:Overseer",
}


def _hash(model) -> str:
  data = mujoco.MjData(model)
  for _ in range(500):
    mujoco.mj_step(model, data)
  return hashlib.sha256(data.qpos.tobytes() + data.qvel.tobytes()).hexdigest()


@pytest.mark.parametrize("world", ["room_hub", "home"])
def test_compiling_from_the_spec_is_the_old_path(world):
  cfg = world_config(world)
  assert _hash(world_spec(cfg["model"]).compile()) == \
      _hash(mujoco.MjModel.from_xml_path(cfg["model"]))


def _life(world: str = "room_hub") -> HubLifecycle:
  cfg = world_config(world)
  spec = world_spec(cfg["model"])
  model = spec.compile()
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, realtime=False, world=world,
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      spec=spec, errand=False)
  for _ in range(300):
    mujoco.mj_step(model, data)
  return life


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


def test_the_seam_hangs_a_tool_and_rebinds_every_holder():
  life = _life()
  old_model, old_data = life.model, life.data
  lcd = old_model.body("module_lcd").id
  mujoco.mj_forward(old_model, old_data)   # xpos lags qpos by a step after mj_step
  lcd_before = old_data.xpos[lcd].copy()
  t_before = float(old_data.time)
  events = []
  life.on_event.append(events.append)
  tool = validate.check(SCOOP)

  rec = life.hang_tool(tool, 2)

  # nothing retired: the rail's bay was empty, and no original can be
  assert rec["retired"] is None and rec["verbs"] == ["scoop.tilt"]
  assert rec["recompileMs"] < 200
  # the world: a new model, the five originals still there, the scoop in
  # the rail's bay C, state carried
  assert life.model is not old_model
  names = {mujoco.mj_id2name(life.model, mujoco.mjtObj.mjOBJ_BODY, i)
           for i in range(life.model.nbody)}
  assert "module_scoop" in names and "module_pen" in names
  assert float(life.data.time) == pytest.approx(t_before)
  assert np.allclose(life.data.xpos[life.model.body("module_lcd").id], lcd_before)
  assert life.rack_inventory == {"module_lcd": 0, "module_plug": 1, "module_pen": 2,
                                 "module_claw": 3, "module_seed": 4,
                                 "module_scoop": len(HUB_STATION_YS) + 2}
  assert "scoop.tilt" in axes.AXES and "scoop.tilt" in axes.SENSORS
  # ...at the rail's station, in the world (the rack frame -> world map
  # the swap and the standoff both use)
  prior = life.mission.rack_prior
  hx, hy = prior.to_world(RACK_HANG_X, BUILT_STATION_YS[2])
  scoop = life.data.xpos[life.model.body("module_scoop").id]
  assert abs(float(scoop[0]) - hx) < 0.01 and abs(float(scoop[1]) - hy) < 0.01
  # the wire
  changed = [e for e in events if e.get("type") == "scene_changed"]
  assert len(changed) == 1
  assert changed[0]["module"] == "module_scoop" and changed[0]["retired"] is None
  assert any(b["name"] == "module_scoop" for b in changed[0]["scene"]["bodies"])
  # ...and the world still steps, the module SETTLES INTO THE TRAYS, and
  # the swap reads it hung there (the rail's trays hold a peg like the
  # first rack's; `module_state` knows the rail's stations)
  for _ in range(1000):
    mujoco.mj_step(life.model, life.data)
  st = life.mission.swap.module_state("module_scoop")
  assert st["on_fork"] is False and st["hung"] is True, st
  # THE RUNTIME FENCE: nothing reachable from the lifecycle holds the old
  # world. Shown to fail by commenting out one `rebind` call in
  # `HubLifecycle.rebind`.
  stale = [(p, o) for p, o in _holders(life)
           if o is old_model or o is old_data]
  assert stale == [], [p for p, _ in stale]


def test_the_seam_refuses_mid_errand_on_the_fork_and_without_a_rail():
  life = _life()
  tool = validate.check(SCOOP)
  life.state = "USE_TOOL"
  with pytest.raises(SeamRefused, match="between errands"):
    life.hang_tool(tool, 2)
  life.state = "DECIDE"
  with pytest.raises(SeamRefused, match="no bay 3; the built-tool rail has 3"):
    life.hang_tool(tool, 3)
  bare = HubLifecycle(life.model, life.data, realtime=False, world="room_hub",
                      errand=False)
  with pytest.raises(SeamRefused, match="without its spec"):
    bare.hang_tool(tool, 2)
  # A WORLD WITHOUT THE RAIL (issue #277): the bare spike world's shape,
  # made here by deleting the rail from room_hub's spec. Read off the
  # compiled model, so the refusal is what the world is, not a flag.
  cfg = world_config("room_hub")
  spec = world_spec(cfg["model"])
  spec.delete(spec.body(coupling.BUILT_RACK_BODY))
  model = spec.compile()
  norail = HubLifecycle(model, mujoco.MjData(model), realtime=False,
                        world="room_hub", spec=spec, errand=False)
  assert norail.has_built_rack is False and life.has_built_rack is True
  with pytest.raises(SeamRefused, match="no built-tool rack"):
    norail.hang_tool(tool, 0)


@pytest.mark.parametrize("module", sorted(coupling.MODULE_TAG_IDS))
def test_a_hand_built_module_is_never_retired(module):
  """The five originals are permanent (issue #277): the seam refuses them
  by name at the spec, and the lifecycle refuses before it touches the
  spec -- what every offered job is written against cannot be deleted by
  the robot it is offered to. Shown to fail by dropping `HAND_BUILT` from
  `seam.retire`: the pen, its carriage and its actuator all go."""
  cfg = world_config("room_hub")
  spec = world_spec(cfg["model"])
  with pytest.raises(SeamRefused, match="original modules"):
    seam.retire(spec, module)
  assert spec.body(module) is not None
  life = _life()
  with pytest.raises(SeamRefused, match="original modules"):
    life.retire_tool(module)
  assert module in life.rack_inventory


def test_retiring_a_built_tool_takes_its_actuators_and_empties_its_bay():
  """A built tool leaves with its actuators and its verbs; a bay named
  again while a built tool hangs there retires that tool first."""
  life = _life()
  events = []
  life.on_event.append(events.append)
  life.hang_tool(validate.check(SCOOP), 0)
  assert life.model.actuator("module_scoop_tilt").id >= 0
  rec = life.retire_tool("module_scoop")
  assert rec["bay"] == 0 and rec["retiredWhat"]["actuators"] == ["module_scoop_tilt"]
  assert "module_scoop" not in life.rack_inventory and "scoop.tilt" not in axes.AXES
  names = {mujoco.mj_id2name(life.model, mujoco.mjtObj.mjOBJ_BODY, i)
           for i in range(life.model.nbody)}
  assert "module_scoop" not in names and "module_pen" in names
  # ...and naming the bay another built tool holds retires that one
  life.hang_tool(validate.check(SCOOP), 1)
  rec = life.hang_tool(validate.check(SCOOP2), 1)
  assert rec["retired"] == "module_scoop"
  assert life.rack_inventory.get("module_scoop2") == len(HUB_STATION_YS) + 1
  assert "module_scoop" not in life.rack_inventory
  with pytest.raises(SeamRefused, match="not on the rack"):
    life.retire_tool("module_scoop")


def test_ids_shift_on_a_retire_and_the_state_still_follows_by_name():
  """MEASURED: deleting a built tool moves every body after it in the
  tree (a second built tool here; it was the claw, 34 -> 31, while an
  original could be retired) and the moved body is exactly where it was.
  This is why every rebind resolves by name."""
  cfg = world_config("room_hub")
  spec = world_spec(cfg["model"])
  from pluggybot.rack.localize import RackPose
  prior = cfg["rack"] or RackPose.prior()
  seam.attach(spec, validate.check(SCOOP), 0, (prior.x, prior.y),
              np.degrees(prior.yaw))
  seam.attach(spec, validate.check(SCOOP2), 1, (prior.x, prior.y),
              np.degrees(prior.yaw))
  model = spec.compile()
  data = mujoco.MjData(model)
  for _ in range(500):
    mujoco.mj_step(model, data)
  second = model.body("module_scoop2").id
  mujoco.mj_forward(model, data)
  pose = data.xpos[second].copy()
  seam.retire(spec, "module_scoop")
  model2, data2 = seam.recompile(spec, model, data)
  assert model2.body("module_scoop2").id < second
  assert np.allclose(data2.xpos[model2.body("module_scoop2").id], pose)


def test_every_bay_on_the_rail_pairs_with_its_own_tag():
  """The rail's three stations index `STATION_YS` after the first rack's
  five, each with its own bay tag in the rack-fixed layout the dock camera
  fits to (issue #277) -- so a built tool's approach ranges off ITS bay's
  marker like any other, and a fix at the far bay is a fit over the rail's
  tags, not one tag's coin-flip yaw."""
  from pluggybot.rack.tags import BAY_TAG_IDS
  first = len(HUB_STATION_YS)
  assert coupling.STATION_YS[first:] == BUILT_STATION_YS
  for k, y in enumerate(BUILT_STATION_YS):
    assert coupling.built_bay_index(k) == first + k
    assert coupling.is_built_bay(first + k) and not coupling.is_built_bay(k)
    assert coupling.bay_tag_id(y) == BAY_TAG_IDS[first + k]
    assert coupling.RACK_TAG_FACES[BAY_TAG_IDS[first + k]] == (coupling.BAY_TAG_FACE_X, y)
  with pytest.raises(ValueError):
    coupling.built_bay_index(len(BUILT_STATION_YS))


def test_the_tag_png_is_written_once(tmp_path):
  p1 = seam.write_tag_png(seam.tag_id_for_bay(2), tmp_path)
  stamp = p1.stat().st_mtime_ns
  p2 = seam.write_tag_png(seam.tag_id_for_bay(2), tmp_path)
  assert p1 == p2 and p2.stat().st_mtime_ns == stamp
  assert not list(tmp_path.glob("tmp*"))


def test_every_holder_of_the_world_can_be_rebound():
  """THE STATIC FENCE. A class that stores `self.model` or `self.data` is a
  class that can go stale; it defines `rebind`, or it is on the roster
  above with a reason. A grep would pass on a comment; the syntax tree
  does not."""
  missing = []
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
      has_rebind = any(isinstance(n, ast.FunctionDef) and n.name == "rebind"
                       for n in cls.body)
      if not has_rebind and key not in TRANSIENT_HOLDERS:
        missing.append(key)
  assert missing == [], f"holds the world and cannot be rebound: {missing}"


# ---- 5. a pair hangs a tool (issue #315) ------------------------------------

def _pair():
  """Two lifecycles over one world, settled. ⚠ The steps are load-bearing
  and `_life` steps for the same reason: before the first one `xpos` is
  all zeros, so `module_state` reads every module as on the fork. These
  tests call `hang_tool` directly; the REAL path reaches it through
  `begin()`, which forwards the kinematics itself (and did not, which is
  why nothing built ever survived a restart --
  `test_a_built_tool_comes_back_after_a_restart_on_the_real_path`)."""
  from pluggybot.pair import build_pair
  lives = build_pair("room_hub", pack="hosting", errands=("none", "none"),
                     realtime=False)
  for _ in range(100):
    mujoco.mj_step(lives[0].model, lives[0].data)
  for life in lives:
    life.state = "DECIDE"
  return lives


def test_a_pair_is_compiled_from_a_spec_both_lifecycles_keep():
  """The deployed world is a pair, and `build_pair` threw the spec away
  after `compile()` -- so `can_reshape` refused every build with "this
  world was compiled without its spec" before it ever reached the pair
  rule (issue #315: 433 specified, 433 refused). One spec, one rack: the
  rail is the WORLD's, and two inventories would diverge the moment
  either robot hung anything."""
  a, b = _pair()
  assert a.spec is not None and a.spec is b.spec
  assert a.rack_inventory is b.rack_inventory
  assert a.has_built_rack and b.has_built_rack
  a.can_reshape(0)                                  # no refusal at all


def test_a_tool_built_by_one_robot_rebinds_the_other_and_the_shared_loop():
  """The whole of #315's blocker 1. Robot A hangs a tool while robot B is
  driving, both under ONE `tick.run_many` loop:

    - every lifecycle is rebound, not just the builder's (`_recompile`);
    - the loop's own `mj_step` reads the world off the swap each step, so
      it steps the NEW one -- captured once at loop start, `data.time`
      simply stops advancing while every rebound holder reads the new
      world (that is the failure this asserts against);
    - robot B's pose is carried across by name;
    - nothing reachable from EITHER lifecycle holds the old world.
  """
  a, b = _pair()
  old_model, old_data = a.model, a.data
  r2 = old_model.body("r2_pluggybot").id
  mujoco.mj_forward(old_model, old_data)
  b_before = old_data.xpos[r2].copy()
  tool = validate.check(SCOOP)
  events = []
  a.on_event.append(events.append)

  def builder(n):
    yield from ((0.0, 0.0) for _ in range(n))
    builder.rec = a.hang_tool(tool, 0)              # between two steps
    yield from ((0.0, 0.0) for _ in range(n))

  def driver(n):
    yield from ((0.05, 0.0) for _ in range(2 * n))

  n = 100
  t0 = float(a.data.time)
  tick.run_many([(a.mission.swap, builder(n)), (b.mission.swap, driver(n))],
                name="pair-build")

  assert builder.rec["module"] == "module_scoop" and builder.rec["retired"] is None
  # THE LOOP STEPPED THE NEW WORLD: 2n steps of sim time, not n
  assert a.model is not old_model and a.data is not old_data
  assert float(a.data.time) - t0 == pytest.approx(2 * n * a.model.opt.timestep)
  # ...and the robot that did NOT build followed it, down to its swap
  assert b.model is a.model and b.data is a.data
  assert b.mission.swap.model is a.model and b.mission.swap.data is a.data
  # ...with its pose carried across by name (its ids moved: a body was added)
  # (it drove ~2 cm in 200 steps at 0.05 m/s: the claim is that the pose
  # CARRIED, not that it held still, and a lost `qpos` reads as the origin)
  mujoco.mj_forward(a.model, a.data)
  moved = a.data.xpos[a.model.body("r2_pluggybot").id]
  assert np.linalg.norm(moved[:2] - b_before[:2]) < 0.25 and moved[2] > 0.0
  # one rack, and B can see and fetch what A built
  assert b.rack_inventory["module_scoop"] == len(HUB_STATION_YS)
  from pluggybot.lifecycle import world_facts
  assert "module_scoop" in world_facts(b.world, rack=b.rack_inventory).tools
  # the wire said so once
  assert [e["type"] for e in events if e.get("type") == "scene_changed"] == \
      ["scene_changed"]
  # THE RUNTIME FENCE, on BOTH lives
  for life in (a, b):
    stale = [p for p, o in _holders(life) if o is old_model or o is old_data]
    assert stale == [], (life.root, [p for p, _ in stale])


def test_the_seam_waits_for_the_other_robot_and_says_whose_errand_it_is(monkeypatch):
  """A pair shares one world, so a recompile pulls it out from under BOTH
  routines -- and a tool controller (the pen, the claw, the dispenser) is
  built per errand and is never rebound. Every robot has to be between
  errands with its fork empty, and the refusal names the busy one: "wait"
  and "never" are different answers."""
  from pluggybot.procedure import steps
  a, b = _pair()
  tool = validate.check(SCOOP)
  b.state = "SWAP_RETURN"
  with pytest.raises(SeamRefused, match=f"{b.robot_name} is busy"):
    a.hang_tool(tool, 0)
  b.state = "DECIDE"
  # ...and so does a module on the OTHER robot's fork
  monkeypatch.setattr(steps, "_carried",
                      lambda life: "module_pen" if life is b else None)
  with pytest.raises(SeamRefused, match=f"{b.robot_name} is busy"):
    a.hang_tool(tool, 0)
  monkeypatch.undo()
  a.hang_tool(tool, 0)                              # both idle: it hangs


def test_a_bay_the_other_robots_tool_hangs_in_is_not_this_robots_to_take():
  """One rail, two minds (issue #315). "The bay you name is taken" is a
  rule about a robot's OWN tools; the other robot's module is refused with
  whose it is, on `bay_index`'s terms for the five originals, and the
  world is not recompiled behind the refusal."""
  a, b = _pair()
  a.hang_tool(validate.check(SCOOP), 0)
  world = b.model
  with pytest.raises(SeamRefused, match="holds the scoop, which is not yours"):
    b.hang_tool(validate.check(SCOOP2), 0)
  with pytest.raises(SeamRefused, match="not yours to retire"):
    b.retire_tool("module_scoop")
  assert b.model is world and "module_scoop" in b.rack_inventory
  # ...and its OWN bay is still free to it
  b.hang_tool(validate.check(SCOOP2), 1)
  assert b.rack_inventory["module_scoop2"] == len(HUB_STATION_YS) + 1
  # ⚠ AND THE RULE STILL LETS A ROBOT REPLACE ITS OWN. "The bay you name
  # is taken" is the prompt's promise and the ownership test must not eat
  # it: `built` is what THIS lifecycle hung, so A retires A's scoop and
  # takes bay A back. Nothing pinned this before, and an ownership check
  # written off the shared inventory instead would have broken it silently.
  rec = a.hang_tool(validate.check({**SCOOP, "name": "scoop3"}), 0)
  assert rec["retired"] == "module_scoop"
  assert "module_scoop" not in a.rack_inventory
  assert a.rack_inventory["module_scoop3"] == len(HUB_STATION_YS)


def test_the_changed_scene_is_the_fixtures_shape_sidecar_and_pair_name():
  """The site REPLACES its whole scene graph with `scene_changed.scene`
  (rooftop #250), and protocol/README.md promises "exactly the shape the
  scene fixture has". It was not: no `meta`, so every `visual` hint, the
  zones, the spawns and the plate glyphs were absent, and a pair was named
  the single-robot world. Dormant until #315 -- a pair could not hang a
  tool at all -- and on the deployed home world it would have repainted
  the house as grey primitives the moment the robot built anything."""
  import json
  from pathlib import Path

  from pluggybot.robot import pair_model_name
  from pluggybot.telemetry.scene import scene_dict
  a, b = _pair()
  events = []
  a.on_event.append(events.append)
  a.hang_tool(validate.check(SCOOP), 0)
  scene = next(e for e in events if e.get("type") == "scene_changed")["scene"]
  cfg = world_config("room_hub")
  assert scene["model"] == pair_model_name(cfg["model_name"])
  assert any(body["name"] == "module_scoop" for body in scene["bodies"])
  assert any(body["name"] == "r2_pluggybot" for body in scene["bodies"])
  # ...and on a world WITH a sidecar, every field the fixture carries
  life = _life("home")
  events = []
  life.on_event.append(events.append)
  life.hang_tool(validate.check(SCOOP), 0)
  scene = next(e for e in events if e.get("type") == "scene_changed")["scene"]
  cfg = world_config("home")
  meta = json.loads(Path(cfg["meta"]).read_text())
  fixture = scene_dict(life.model, cfg["model_name"], meta=meta)
  assert scene["model"] == cfg["model_name"] == "home_world"
  assert set(scene) == set(fixture)
  assert scene["zones"] and scene["zones"] == fixture["zones"]
  assert scene["plates"] and scene["plates"] == fixture["plates"]
  assert any(body["visual"] for body in scene["bodies"])
