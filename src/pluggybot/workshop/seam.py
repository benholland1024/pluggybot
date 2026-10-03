"""The recompile seam (issue #168, slice C): a tool appears in a running
world.

The model is compiled ONCE per mission and MuJoCo cannot add a body to a
compiled model -- so a tool built mid-run means compiling a NEW model from
the spec the world was compiled from, with the state carried across. This
module is the SPEC SURGERY and the recompile; `HubLifecycle.hang_tool` is
the seam that owns the consequences (every holder of the old model re-
pointed at the new one, the registries, the wire).

  MEASURED (the spike, 2026-09-13, a rover world + a scoop): `MjSpec.recompile
  (model, data)` takes ~13 ms, preserves `time` and `qpos`, keeps every
  existing element's id when a body is APPENDED (new ids come after the
  old), and returns NEW `MjModel` / `MjData` objects -- the Python binding
  does not update in place, and the old handles keep stepping a stale
  world. That last fact is the whole reason the lifecycle carries a rebind
  protocol rather than a pointer.

  ⚠ A BUILT TOOL HANGS ON THE BUILT-TOOL RAIL, and the hand-built modules
  are permanent (issue #277; ToolPattern.md). On legs (#407) the rail is
  three more bays on the rack's own board (`legs.rack.BUILT`); a bay here is
  the rail's own index, the only bays a build may name, and naming one that
  a built tool already hangs in RETIRES that tool: its body (and subtree)
  and its actuators are deleted from the spec before the new module is
  attached there. `retire` refuses a hand-built module by name. ⚠ Deleting
  a body SHIFTS the ids of everything after it in the tree, so a rebind
  re-resolves every id by name and trusts none; a built module is attached
  LAST, after the robots, so only another built module's ids ever move.

  ⚠ NO IDENTITY TAG: on legs a tool is known by its bay -- the bay's pair
  of tags, which the rail carries, and its presence switch -- never by a
  tag of its own (the rover's built modules carried one).
"""

from __future__ import annotations

import math

import mujoco

from pluggybot.rack.tags import MODULE_TAG_IDS
from pluggybot.workshop import build
from pluggybot.workshop.spec import Tool

#: The hand-built modules, permanent since #277: `retire` refuses them.
#: Read off the shipped inventory rather than listed twice.
HAND_BUILT = tuple(MODULE_TAG_IDS)


class SeamRefused(ValueError):
  pass


def bay_of(rack: dict[str, int], module: str) -> int:
  """Which bay a module hangs in, off the lifecycle's inventory."""
  try:
    return rack[module]
  except KeyError:
    raise SeamRefused(f"{module} is not on the rack") from None


def _subtree_names(spec: mujoco.MjSpec, body) -> tuple[set[str], set[str]]:
  """Every body and joint name under (and including) a spec body."""
  bodies, joints = set(), set()
  stack = [body]
  while stack:
    b = stack.pop()
    bodies.add(b.name)
    for j in b.joints:
      joints.add(j.name)
    stack.extend(b.bodies)
  return bodies, joints


def retire(spec: mujoco.MjSpec, module: str) -> dict:
  """Delete a BUILT module from the spec: its body and subtree, the
  actuators on its joints. Returns what went, for the record. A hand-built
  module is refused by name (issue #277): the pen, the claw and the LCD
  are what every offered job is written against."""
  if module in HAND_BUILT:
    raise SeamRefused(f"the {module.removeprefix('module_')} is one of the "
                      "original modules and stays on the rack")
  body = spec.body(module)
  if body is None:
    raise SeamRefused(f"no body {module!r} in the spec")
  bodies, joints = _subtree_names(spec, body)
  actuators = [a.name for a in spec.actuators if a.target in joints]
  for a in list(spec.actuators):
    if a.name in actuators:
      spec.delete(a)
  spec.delete(body)
  return {"module": module, "bodies": sorted(bodies), "actuators": actuators}


def rack_pose(model) -> tuple[float, float, float]:
  """The rack's commissioned frame in the world, (x, y, yaw rad), off its
  compiled body: where the rail's bays are."""
  from pluggybot.legs import rack as rk
  b = model.body(rk.RACK_BODY).id
  w, _, _, z = (float(v) for v in model.body_quat[b])
  return float(model.body_pos[b][0]), float(model.body_pos[b][1]), 2.0 * math.atan2(z, w)


def attach(spec: mujoco.MjSpec, tool: Tool, bay: int,
           rack: tuple[float, float, float]) -> dict:
  """The tool's module hung at the rail's bay `bay` (`legs.rack.BUILT`), with
  its default class and its actuators, into the spec, in the rack's frame
  `rack` (`rack_pose`). Returns the names the lifecycle records."""
  from pluggybot.legs import rack as rk
  if tool.body in {b.name for b in spec.bodies}:
    raise SeamRefused(f"the world already has a {tool.body}")
  x, y, yaw = rack
  default, module = build.module_for(tool, rk.bay_peg(rk.BUILT, bay, pos=(x, y), yaw=yaw),
                                     yaw=yaw)
  child = mujoco.MjSpec.from_string(
    f'<mujoco><compiler angle="radian"/><default>{default}</default>'
    f"<worldbody>{module}</worldbody>"
    f"<actuator>{build.actuator_xml(tool, tool.body)}</actuator></mujoco>")
  frame = spec.worldbody.add_frame()
  spec.attach(child, prefix="", frame=frame)
  return {"module": tool.body, "bay": bay,
          "actuators": [build.actuator_name(tool.body, p.axis.verb) for p in tool.axes]}


def recompile(spec: mujoco.MjSpec, model, data):
  """The new (model, data), state carried across. NEW objects: the caller
  must re-point every holder (`HubLifecycle.rebind`)."""
  new_model, new_data = spec.recompile(model, data)
  mujoco.mj_forward(new_model, new_data)
  return new_model, new_data
