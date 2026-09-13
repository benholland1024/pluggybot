"""The recompile seam (issue #168, slice C): a tool appears in a running
world.

The model is compiled ONCE per mission and MuJoCo cannot add a body to a
compiled model -- so a tool built mid-run means compiling a NEW model from
the spec the world was compiled from, with the state carried across. This
module is the SPEC SURGERY and the recompile; `HubLifecycle.hang_tool` is
the seam that owns the consequences (every holder of the old model re-
pointed at the new one, the registries, the wire).

  MEASURED (the spike, 2026-09-13, room_hub + the scoop): `MjSpec.recompile
  (model, data)` takes ~13 ms, preserves `time` and `qpos`, keeps every
  existing element's id when a body is APPENDED (new ids come after the
  old), and returns NEW `MjModel` / `MjData` objects -- the Python binding
  does not update in place, and the old handles keep stepping a stale
  world. That last fact is the whole reason the lifecycle carries a rebind
  protocol rather than a pointer.

  ⚠ A BAY IS A REPLACED MODULE (ToolPattern.md §6, route 3). The rack has
  five bays and a sixth needs the rail to grow through room_hub's west
  wall, so a built tool takes a bay by RETIRING the module in it: that
  module's body (and its subtree: carriage, jaws, shuttle), its actuators,
  and any free bodies that were its payload (the dispenser's seeds) are
  deleted from the spec before the new module is attached at the same
  station. ⚠ Deleting a body SHIFTS the ids of everything after it in the
  tree, so a rebind re-resolves every id by name and trusts none.

  ⚠ THE TAG IS WRITTEN ONCE. A built module's identity tag gets the next
  id past the hand-built modules' (`BUILT_TAG_BASE`); its PNG goes to
  `models/tags/` through a temp file and an atomic rename, because a
  parallel test suite writing the same PNG twice is a known race.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mujoco

from pluggybot.rack import coupling
from pluggybot.rack.coupling import (
  HUB_PEG_Z, HUB_STATION_YS, RACK_HANG_X, SMALL_PLATE_HALF, TOOL_HALF_X,
  rack_frame_to_world,
)
from pluggybot.rack.tags import TAG_DIR, tag_image
from pluggybot.workshop import build
from pluggybot.workshop.spec import Tool

#: Tag ids for built modules start here; the hand-built ones end at 14
#: (`rack/tags.py`) and bay tags stop at 6. One per BAY, not per tool: a
#: retired tool's tag id is reused by the next tool in that bay, so the
#: dock camera reads at most five module tags however many tools a day
#: builds and retires.
BUILT_TAG_BASE = 15

#: Free bodies that are a hand-built module's PAYLOAD and leave with it.
RETIRE_EXTRAS = {
  "module_seed": tuple(f"seed_{i}" for i in range(coupling.SEED_COUNT)),
}


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
  """Delete a module from the spec: its body and subtree, the actuators on
  its joints, its payload bodies. Returns what went, for the record."""
  body = spec.body(module)
  if body is None:
    raise SeamRefused(f"no body {module!r} in the spec")
  bodies, joints = _subtree_names(spec, body)
  actuators = [a.name for a in spec.actuators if a.target in joints]
  for a in list(spec.actuators):
    if a.name in actuators:
      spec.delete(a)
  extras = []
  for name in RETIRE_EXTRAS.get(module, ()):
    extra = spec.body(name)
    if extra is not None:
      spec.delete(extra)
      extras.append(name)
  spec.delete(body)
  return {"module": module, "bodies": sorted(bodies), "actuators": actuators,
          "payload": extras}


def tag_id_for_bay(bay: int) -> int:
  return BUILT_TAG_BASE + bay


def write_tag_png(tag_id: int, directory: Path = TAG_DIR) -> Path:
  """The tag's PNG on disk, written once and atomically."""
  from PIL import Image
  directory.mkdir(parents=True, exist_ok=True)
  path = directory / f"tag{tag_id}.png"
  if path.exists():
    return path
  fd, tmp = tempfile.mkstemp(dir=directory, suffix=".png")
  os.close(fd)
  Image.fromarray(tag_image(tag_id)).save(tmp)
  os.replace(tmp, path)
  return path


def write_built_tag_pngs(directory: Path = TAG_DIR) -> list[int]:
  """The five built-module tags, committed beside the hand-built ones so
  the website can vendor them once (`python -m pluggybot.rack.coupling`
  writes them with the rest)."""
  return [write_tag_png(tag_id_for_bay(b), directory) and tag_id_for_bay(b)
          for b in range(len(HUB_STATION_YS))]


def tag_face_xml(body: str, tag_id: int) -> str:
  """The identity tag on the +x face, as `_module_faces` gives the
  hand-built modules theirs."""
  return (f'\n      <geom name="{body}_tag" type="box" '
          f'size="0.002 {SMALL_PLATE_HALF:.4f} {SMALL_PLATE_HALF:.4f}" '
          f'pos="{TOOL_HALF_X:.4f} 0 0" contype="0" conaffinity="0" '
          f'material="tagmat{tag_id}"/>')


def attach(spec: mujoco.MjSpec, tool: Tool, bay: int, rack_pos, rack_yaw: float,
           model_dir: Path | None = None) -> dict:
  """The tool's module at a bay's station, plus its actuators and tag
  material, into the spec. Returns the names the lifecycle records."""
  if tool.body in {b.name for b in spec.bodies}:
    raise SeamRefused(f"the world already has a {tool.body}")
  x, y = rack_frame_to_world(RACK_HANG_X, HUB_STATION_YS[bay], rack_pos, rack_yaw)
  tag_id = tag_id_for_bay(bay)
  # the texture file is relative to the model's directory, as the hand-
  # built tags are (`asset_xml`: file="tags/tagN.png")
  write_tag_png(tag_id, (model_dir or Path("models")) / "tags")
  have = {t.name for t in spec.textures}
  if f"tagtex{tag_id}" not in have:
    tex = spec.add_texture()
    tex.name = f"tagtex{tag_id}"
    tex.type = mujoco.mjtTexture.mjTEXTURE_CUBE
    tex.file = f"tags/tag{tag_id}.png"
    mat = spec.add_material()
    mat.name = f"tagmat{tag_id}"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{tag_id}"
    mat.specular, mat.shininess, mat.reflectance = 0.05, 0.05, 0.0
  module = build.module_for(tool, x, y, HUB_PEG_Z, yaw_deg=rack_yaw,
                            tag=tag_face_xml(tool.body, tag_id))
  child = mujoco.MjSpec.from_string(
    f"<mujoco><worldbody>{module}</worldbody>"
    f"<actuator>{build.actuator_xml(tool, tool.body)}</actuator></mujoco>")
  frame = spec.worldbody.add_frame()
  spec.attach(child, prefix="", frame=frame)
  return {"module": tool.body, "bay": bay, "tagId": tag_id,
          "actuators": [build.actuator_name(tool.body, p.axis.verb) for p in tool.axes]}


def recompile(spec: mujoco.MjSpec, model, data):
  """The new (model, data), state carried across. NEW objects: the caller
  must re-point every holder (`HubLifecycle.rebind`)."""
  new_model, new_data = spec.recompile(model, data)
  mujoco.mj_forward(new_model, new_data)
  return new_model, new_data
