"""MJCF -> JSON scene transpiler: what the browser builds the world from.

The browser never parses MJCF. It fetches one JSON scene description --
transpiled here from the COMPILED MjModel, not the XML, so includes,
defaults and generated files have all been resolved by MuJoCo itself --
and instances a ThreeJS primitive per geom. room_hub compiles to boxes,
cylinders, capsules, a sphere and a plane; there are no meshes, and this
transpiler refuses them on purpose (the primitives-only world is a design
decision, so a mesh sneaking in should fail loudly, not ship quietly).

Conversions are applied HERE, unit-tested, so the client does none:
  - sizes are FULL extents in meters. MuJoCo stores box size as
    HALF-extents and cylinder/capsule size as (radius, HALF-length);
    ThreeJS constructors want the full figure.
  - cylinder/capsule axis: MuJoCo's runs along local +Z, ThreeJS's along
    local +Y. Each such geom's local quat is composed with a fixed +90 deg
    rotation about X so a Y-axis ThreeJS primitive lands exactly on the
    MuJoCo geometry. (A plane needs no fix: both sides agree its normal
    is local +Z.)
  - NOT converted: the world frame. Body poses here and streamed poses in
    telemetry both stay in MuJoCo's Z-up world; the client applies one
    Z-up -> Y-up rotation at the scene root. Converting per-frame in the
    recorder would tax every frame forever to save one line of client code.

Geoms with rgba alpha 0 are skipped: those are invisible collision layers
(the schuko socket wells alone are 99 geoms in room_hub) that the robot
needs and no viewer ever sees.

Regenerate the committed fixture after changing any world geometry:
  uv run python -m pluggybot.telemetry.scene
"""

import argparse
import json
import math
from pathlib import Path

import mujoco

from pluggybot.telemetry.protocol import (
  PROTOCOL_VERSION, ROBOT_ROOT, VISUAL_HINTS, dynamic_flags, robot_body_ids,
)

GEOM_TYPE_NAMES = {
  int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
  int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
  int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
  int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
  int(mujoco.mjtGeom.mjGEOM_BOX): "box",
}

_S = math.sqrt(0.5)
AXIS_FIX = (_S, _S, 0.0, 0.0)   # +90 deg about X: maps local +Y onto local +Z
NDIGITS = 6                     # micrometers -- the pogo pins are 1 mm parts


def quat_mul(a, b) -> tuple[float, float, float, float]:
  """Hamilton product, (w, x, y, z) order -- MuJoCo's convention."""
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return (aw * bw - ax * bx - ay * by - az * bz,
          aw * bx + ax * bw + ay * bz - az * by,
          aw * by - ax * bz + ay * bw + az * bx,
          aw * bz + ax * by - ay * bx + az * bw)


def geom_size(gtype: int, size) -> list[float]:
  """Type-specific dimensions in FULL extents (meters).

    box       [x, y, z]
    cylinder  [radius, length]
    capsule   [radius, length]   (length of the cylindrical part)
    sphere    [radius]
    plane     [x, y]             (0 means infinite)
  """
  s = [float(v) for v in size]
  name = GEOM_TYPE_NAMES.get(int(gtype))
  if name == "box":
    return [2 * s[0], 2 * s[1], 2 * s[2]]
  if name in ("cylinder", "capsule"):
    return [s[0], 2 * s[1]]
  if name == "sphere":
    return [s[0]]
  if name == "plane":
    return [2 * s[0], 2 * s[1]]
  raise ValueError(f"unsupported geom type {gtype} -- the world is "
                   "primitives-only by design (see docs/pluggyworld.md)")


def geom_quat(gtype: int, quat) -> list[float]:
  """The geom's body-local quat, with the ThreeJS axis fix baked in for
  cylinders and capsules (see module docstring)."""
  q = tuple(float(v) for v in quat)
  if GEOM_TYPE_NAMES.get(int(gtype)) in ("cylinder", "capsule"):
    q = quat_mul(q, AXIS_FIX)
  return list(q)


def _round(values, ndigits: int = NDIGITS) -> list[float]:
  return [round(float(v), ndigits) for v in values]


def _geom_appearance(model, gid: int) -> tuple[list[float], str | None]:
  """(rgba, texture name) for one geom. A material overrides the geom's
  own rgba and may carry an RGB-role texture (the AprilTags)."""
  mid = int(model.geom_matid[gid])
  if mid < 0:
    return _round(model.geom_rgba[gid], 4), None
  role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
  tid = int(model.mat_texid[mid][role])
  tex = model.tex(tid).name if tid >= 0 else None
  return _round(model.mat_rgba[mid], 4), tex


#: Geoms whose name ends with this are DISPLAY PANELS -- the LCD module's
#: face (issue #13). Named by convention rather than listed in a sidecar
#: because the modules are generated (rack/coupling.py) and live in three
#: worlds, only one of which has a sidecar at all.
SCREEN_SUFFIX = "_screen"


def _quat_rotate(q, v) -> list[float]:
  """Rotate v by quaternion q, both (w,x,y,z) / (x,y,z)."""
  w, x, y, z = q
  # v' = v + 2 * cross(q_vec, cross(q_vec, v) + w * v)
  tx = 2 * (y * v[2] - z * v[1])
  ty = 2 * (z * v[0] - x * v[2])
  tz = 2 * (x * v[1] - y * v[0])
  return [v[0] + w * tx + (y * tz - z * ty),
          v[1] + w * ty + (z * tx - x * tz),
          v[2] + w * tz + (x * ty - y * tx)]


def screen_map(model) -> dict:
  """{body: where its display panel is} for every screen geom in the world.

  The mapping a face needs, and the exact sibling of `boards`: telemetry says
  `"screens": {"module_lcd": {"mode": "face", ...}}` and the geom carrying
  that face is called `module_lcd_screen`. Shipping the pairing here means a
  client never derives one name from the other, so renaming a geom is a
  regenerate rather than a silently blank screen.

  `normal` is the panel's OUTWARD direction in body-local coordinates -- the
  thin axis of the slab, signed to point away from the body origin. A client
  paints its canvas on that face; without the sign it has a 50 % chance of
  drawing the face on the inside of the module.
  """
  screens: dict = {}
  for g in range(model.ngeom):
    name = model.geom(g).name
    if not name or not name.endswith(SCREEN_SUFFIX):
      continue
    size = geom_size(int(model.geom_type[g]), model.geom_size[g])
    pos = [float(v) for v in model.geom_pos[g]]
    quat = [float(v) for v in model.geom_quat[g]]
    thin = size.index(min(size))
    axis = [0.0, 0.0, 0.0]
    axis[thin] = 1.0
    normal = _quat_rotate(quat, axis)
    if sum(n * p for n, p in zip(normal, pos)) < 0:
      normal = [-n for n in normal]
    normal = [n + 0.0 if n else 0.0 for n in normal]   # never ship -0.0
    body = model.body(int(model.geom_bodyid[g])).name
    screens[body] = {
      "geom": name,
      "body": body,
      "size": _round(size),
      "pos": _round(pos),
      "quat": _round(quat),
      "normal": _round(normal),
    }
  return screens


def scene_dict(model, model_name: str, meta: dict | None = None) -> dict:
  """The full scene description for one compiled model.

  Body poses are WORLD-frame rest poses (mj_forward at qpos0), so a client
  can render the scene with no kinematic-tree math at all: static bodies
  keep this pose forever, dynamic bodies get theirs overwritten by
  telemetry frames (also world-frame). `parent` is still shipped for
  grouping.

  `meta` is a generator sidecar (models/<name>.meta.json, issue #6): its
  `visualHints` fill the per-body `visual` slots, and `zones`/`spawns`
  pass through as ADDITIVE optional top-level fields -- consumers that
  predate them ignore them, so no version bump. Hints ride in the sidecar
  and never in geom colors: the robot's cameras render rgba, and
  color-as-encoding would couple the tag detector to the website's art
  direction.
  """
  hints: dict = (meta or {}).get("visualHints", {})
  unknown = set(hints.values()) - set(VISUAL_HINTS)
  if unknown:
    raise ValueError(f"unknown visual hints {sorted(unknown)} -- the "
                     f"vocabulary is {VISUAL_HINTS} (bump it deliberately, "
                     "in both repos, or fix the sidecar)")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  dyn = dynamic_flags(model)
  rob = robot_body_ids(model)

  bodies = []
  used_textures: set[str] = set()
  for b in range(model.nbody):
    geoms = []
    for g in range(model.ngeom):
      if int(model.geom_bodyid[g]) != b:
        continue
      if float(model.geom_rgba[g][3]) == 0.0:
        continue                       # invisible collision layer
      rgba, tex = _geom_appearance(model, g)
      if tex is not None:
        used_textures.add(tex)
      gtype = int(model.geom_type[g])
      geoms.append({
        "name": model.geom(g).name or None,
        "type": GEOM_TYPE_NAMES[gtype],   # KeyError on a mesh, by design
        "size": _round(geom_size(gtype, model.geom_size[g])),
        "pos": _round(model.geom_pos[g]),
        "quat": _round(geom_quat(gtype, model.geom_quat[g])),
        "rgba": rgba,
        "texture": tex,
      })
    name = model.body(b).name or f"body_{b}"
    bodies.append({
      "name": name,
      "parent": None if b == 0 else model.body(int(model.body_parentid[b])).name,
      "dynamic": dyn[b],
      "robot": ROBOT_ROOT if b in rob else None,
      "visual": hints.get(name),
      "pos": _round(data.xpos[b]),
      "quat": _round(data.xquat[b]),
      "geoms": geoms,
    })

  textures = [{
    "name": model.tex(i).name,
    "file": f"{model.tex(i).name}.png",
    "width": int(model.tex_width[i]),
    "height": int(model.tex_height[i]),
  } for i in range(model.ntex) if model.tex(i).name in used_textures]

  scene: dict = {
    "protocolVersion": PROTOCOL_VERSION,
    "model": model_name,
    "upAxis": "z",                     # the client rotates the root, not us
    "bodies": bodies,
    "textures": textures,
  }
  if meta and "zones" in meta:
    scene["zones"] = meta["zones"]
  if meta and "spawns" in meta:
    scene["spawns"] = meta["spawns"]
  if meta and "boards" in meta:
    # The drawing surfaces, by the name the telemetry uses (issue #12). The
    # geoms are already in `bodies`; what this adds is the MAPPING -- a `draw`
    # event says `"board": "whiteboard_a"` and gives points in that board's
    # own frame, and without this the client has a polyline it cannot place:
    # the board's telemetry name is the generator's, while the geom is called
    # `board_b`. `half` is (depth, width, height) in the board's frame and
    # `heading` is the outward normal the robot squares up to, so a canvas is
    # width x height and the polyline's +lat runs left across it.
    scene["boards"] = meta["boards"]
  screens = screen_map(model)
  if screens:
    # Read off the COMPILED model rather than a sidecar: the display modules
    # are generated into every world, and only home_world has a sidecar.
    scene["screens"] = screens
  return scene


def export_textures(model, out_dir: Path) -> list[Path]:
  """Write every texture as a PNG next to the scene JSON, decoded from the
  compiled model's own pixel data -- no filename archaeology against the
  XML, and the fixture directory stays self-contained for vendoring."""
  from PIL import Image
  out_dir.mkdir(parents=True, exist_ok=True)
  written = []
  for i in range(model.ntex):
    w, h, nc = (int(model.tex_width[i]), int(model.tex_height[i]),
                int(model.tex_nchannel[i]))
    adr = int(model.tex_adr[i])
    pixels = model.tex_data[adr:adr + w * h * nc].reshape(h, w, nc)
    path = out_dir / f"{model.tex(i).name}.png"
    Image.fromarray(pixels.squeeze()).save(path)
    written.append(path)
  return written


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("model", nargs="?", default="models/room_hub.xml")
  parser.add_argument("-o", "--out", default=None,
                      help="output JSON (default protocol/scene.<name>.json)")
  parser.add_argument("--textures", default=None,
                      help="texture PNG directory (default <out dir>/textures)")
  parser.add_argument("--meta", default=None,
                      help="generator sidecar with visualHints/zones/spawns "
                           "(default: models/<name>.meta.json when it exists)")
  args = parser.parse_args()

  name = Path(args.model).stem
  model = mujoco.MjModel.from_xml_path(args.model)
  meta_path = (Path(args.meta) if args.meta
               else Path(args.model).with_suffix(".meta.json"))
  meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
  out = Path(args.out) if args.out else Path("protocol") / f"scene.{name}.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  scene = scene_dict(model, name, meta=meta)
  out.write_text(json.dumps(scene, indent=1) + "\n")
  tex_dir = Path(args.textures) if args.textures else out.parent / "textures"
  written = export_textures(model, tex_dir)

  n_geoms = sum(len(b["geoms"]) for b in scene["bodies"])
  print(f"{out}: {len(scene['bodies'])} bodies, {n_geoms} visible geoms "
        f"({model.ngeom - n_geoms} invisible skipped), "
        f"{len(written)} textures -> {tex_dir}/")


if __name__ == "__main__":
  main()
