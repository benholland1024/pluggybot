"""The recompile seam (issue #168, slice C): a tool appears in a running
world, and every holder of the old world follows it.

What these hold down:

  1. THE SPEC PATH IS THE OLD PATH. A world compiled from its `MjSpec` is
     trajectory-identical to `MjModel.from_xml_path`, so keeping the spec
     costs nothing (both worlds, hashed after 500 steps).
  2. THE SEAM. `hang_tool` retires the module in a bay, hangs the built
     tool there, recompiles with the state carried across, rebinds every
     holder, registers the verbs and emits `scene_changed` -- measured on a
     bare lifecycle, not flown.
  3. NO STALE HOLDER, two ways. At run time: every object reachable from
     the lifecycle that holds an `MjModel` or `MjData` holds the NEW one.
     Statically: every class in `src/` that assigns `self.model` or
     `self.data` defines `rebind`, or is on the roster below with a reason.
     The static fence is what catches a holder added next month; the
     runtime walk is what catches one the lifecycle forgot to call.
  4. BETWEEN ERRANDS ONLY. Mid-errand, with a module on the fork, or on a
     pair, the seam refuses out loud.
"""

import ast
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.lifecycle import HubLifecycle, world_config
from pluggybot.procedure import axes
from pluggybot.robot import world_spec
from pluggybot.workshop import seam, validate
from pluggybot.workshop.seam import SeamRefused
from test_workshop import SCOOP

SRC = Path(__file__).parent.parent / "src" / "pluggybot"

#: Classes that assign `self.model` / `self.data` and do NOT rebind, each
#: with the reason. Adding one here is a decision made in the test.
TRANSIENT_HOLDERS = {
  # built per errand inside a routine and never held by the lifecycle: a
  # recompile is refused mid-errand, so one never outlives its world
  "tools/drawing.py:PenPlotter", "tools/gripper.py:ClawTool",
  "tools/dispenser.py:SeedDispenser",
  # the plug-era RL environment owns its own world and is never recompiled
  "envs/dock_env.py:DockEnv",
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

  assert rec["retired"] == "module_pen" and rec["verbs"] == ["scoop.tilt"]
  assert rec["recompileMs"] < 200
  # the world: a new model, the pen gone, the scoop in bay C, state carried
  assert life.model is not old_model
  names = {mujoco.mj_id2name(life.model, mujoco.mjtObj.mjOBJ_BODY, i)
           for i in range(life.model.nbody)}
  assert "module_scoop" in names and "module_pen" not in names
  assert float(life.data.time) == pytest.approx(t_before)
  assert np.allclose(life.data.xpos[life.model.body("module_lcd").id], lcd_before)
  assert life.rack_inventory == {"module_lcd": 0, "module_plug": 1, "module_claw": 3,
                                 "module_seed": 4, "module_scoop": 2}
  assert "scoop.tilt" in axes.AXES and "scoop.tilt" in axes.SENSORS
  # the wire
  changed = [e for e in events if e.get("type") == "scene_changed"]
  assert len(changed) == 1
  assert changed[0]["module"] == "module_scoop" and changed[0]["retired"] == "module_pen"
  assert any(b["name"] == "module_scoop" for b in changed[0]["scene"]["bodies"])
  # ...and the world still steps, with the swap reading the new module
  for _ in range(200):
    mujoco.mj_step(life.model, life.data)
  assert life.mission.swap.module_state("module_scoop")["on_fork"] is False
  # THE RUNTIME FENCE: nothing reachable from the lifecycle holds the old
  # world. Shown to fail by commenting out one `rebind` call in
  # `HubLifecycle.rebind`.
  stale = [(p, o) for p, o in _holders(life)
           if o is old_model or o is old_data]
  assert stale == [], [p for p, _ in stale]


def test_the_seam_refuses_mid_errand_on_the_fork_and_on_a_pair():
  life = _life()
  tool = validate.check(SCOOP)
  life.state = "USE_TOOL"
  with pytest.raises(SeamRefused, match="between errands"):
    life.hang_tool(tool, 2)
  life.state = "DECIDE"
  life.peers = [object()]
  with pytest.raises(SeamRefused, match="pair"):
    life.hang_tool(tool, 2)
  life.peers = []
  with pytest.raises(SeamRefused, match="no bay 7"):
    life.hang_tool(tool, 7)
  bare = HubLifecycle(life.model, life.data, realtime=False, world="room_hub",
                      errand=False)
  with pytest.raises(SeamRefused, match="without its spec"):
    bare.hang_tool(tool, 2)


def test_retiring_takes_the_payload_and_the_actuators_with_it():
  """The dispenser leaves with its three seeds and its gate; the claw's
  jaw bodies go with the claw; what stays is what was not the module's."""
  cfg = world_config("room_hub")
  spec = world_spec(cfg["model"])
  gone = seam.retire(spec, "module_seed")
  assert gone["payload"] == ["seed_0", "seed_1", "seed_2"]
  assert gone["actuators"] == ["seed_gate"]
  gone = seam.retire(spec, "module_claw")
  assert set(gone["bodies"]) == {"module_claw", "module_claw_jaw_l", "module_claw_jaw_r"}
  assert set(gone["actuators"]) == {"claw_l", "claw_r"}
  model = spec.compile()
  assert model.body("module_pen").id > 0
  assert model.nu == 8 - 3
  with pytest.raises(SeamRefused):
    seam.retire(spec, "module_claw")


def test_ids_shift_on_a_retire_and_the_state_still_follows_by_name():
  """MEASURED: deleting the pen moves the claw from body 34 to 31 and the
  claw is exactly where it was. This is why every rebind resolves by name."""
  cfg = world_config("room_hub")
  spec = world_spec(cfg["model"])
  model = spec.compile()
  data = mujoco.MjData(model)
  for _ in range(500):
    mujoco.mj_step(model, data)
  claw = model.body("module_claw").id
  mujoco.mj_forward(model, data)
  pose = data.xpos[claw].copy()
  seam.retire(spec, "module_pen")
  model2, data2 = seam.recompile(spec, model, data)
  assert model2.body("module_claw").id < claw
  assert np.allclose(data2.xpos[model2.body("module_claw").id], pose)


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
