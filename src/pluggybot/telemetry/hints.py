"""The visual-hint conformance fixture (issue #66, M13's freeze point).

`VISUAL_HINTS` is a tuple of strings, and a tuple of strings is not enough to
build art against. It says a body is a `couch`; it does not say whether the
marker is one box or five, which body-local axis is its height, or whether the
robot will plan around exactly that volume. Two lanes of M13 run in parallel
against those answers -- an artist-shaped session in the website and a
generator-shaped session here -- so the answers have to be written down once,
in a form a builder can read, before either starts guessing.

`protocol/hints.json` is that form. Per hint: ONE synthetic body in exactly
`scene_dict`'s body shape, plus a machine-readable rule.

  ⚠ THE BODY IS EMITTED THROUGH `scene_dict`, NOT HAND-WRITTEN. That is the
  whole reason the fixture is worth having: a hand-written example of a body
  shape drifts from the real one silently and is then worse than nothing,
  because it is a document that looks authoritative and lies. This module
  compiles a real MJCF, runs the real transpiler over it and keeps what comes
  out -- so if `scene_dict`'s body record ever grows a field, the fixture
  grows it on the next regeneration and `tests/test_hints.py` fails until it
  is regenerated.

  ⚠ THE RULES ARE BODY-LOCAL. A scene ships a body's world pose and its geoms'
  body-local `pos`/`quat`, and the browser puts the primitives in a group it
  then places -- so "which axis is the height" is a question about the body's
  own frame and is orientation-independent. `axes` maps a role to where to
  find it in the geom's `size`, and a builder that respects it works for a
  couch against any wall.

WHAT `collides` MEANS, and why it is not decoration: it is whether the robot
will plan around exactly this volume. MEASURED off the home world, where it is
already not constant -- `wall`, `fence`, `rack` and `whiteboard` collide;
`floor`, `ground` and **`plant`** do not. A plant is mapped and driven
through, which is a genuine and slightly surprising fact about this world, and
a builder that drew a solid-looking shrub would be describing an obstacle the
robot ignores. Where it is true, art outside the marker must either sit above
the robot or read as non-solid; a tree's canopy is the case this exists for.
"""

import argparse
import json
from pathlib import Path

import mujoco

from .protocol import PROTOCOL_VERSION, ROBOT_ROOT, VISUAL_HINTS
from .scene import scene_dict

#: Roles an axis can carry, so a builder can switch on a closed set rather
#: than parse prose. `size0`/`size1`/`size2` index the geom's `size` array as
#: the scene ships it (FULL extents, body-local).
AXIS_SLOTS = ("size0", "size1", "size2")

#: `build` says what the builder does with the primitives the scene shipped.
#: `replace` -- draw the asset instead of them (all scenery: the marker is a
#: stand-in nobody is meant to look at). `reskin` -- keep their silhouette and
#: dress it (`rack`, `robot`), because those two are shapes the physics used
#: and flattering them would misdescribe the sim.
BUILD_MODES = ("replace", "reskin")


class Marker:
  """One hint's conformance body: what to compile, and what it promises."""

  def __init__(self, hint: str, geoms: list[tuple], collides: bool,
               axes: dict[str, str], build: str, note: str,
               body: str | None = None, free: bool = False) -> None:
    self.hint, self.geoms, self.collides = hint, geoms, collides
    self.axes, self.build, self.note = axes, build, note
    # `robot` overrides both: its conformance body IS `ROBOT_ROOT`, with a
    # free joint, so the fixture carries a real robot body record --
    # `"robot": "pluggybot"` and `"dynamic": true` -- rather than a
    # scenery-shaped stand-in that would quietly under-describe the one
    # hinted body whose pose a frame overwrites.
    self.body = body or f"hint_{hint}"
    self.free = free
    assert build in BUILD_MODES, build
    assert set(axes.values()) <= set(AXIS_SLOTS), axes

  def mjcf(self, y: float) -> str:
    """One body, spread along +y so the fixture is also a viewable scene."""
    con = 'contype="1" conaffinity="1"' if self.collides \
        else 'contype="0" conaffinity="0"'
    out = [f'    <body name="{self.body}" pos="0 {y:.1f} 0.2">'
           if self.free else f'    <body name="{self.body}" pos="0 {y:.1f} 0">']
    if self.free:
      out.append('      <freejoint/>')
    for i, (gtype, size, pos) in enumerate(self.geoms):
      out.append(f'      <geom name="{self.body}_{i}" type="{gtype}" '
                 f'size="{size}" pos="{pos}" rgba="0.6 0.6 0.6 1" {con}/>')
    out.append("    </body>")
    return "\n".join(out)


#: ⚠ SIZES HERE ARE MuJoCo HALF-EXTENTS; the fixture carries FULL extents,
#: because `scene_dict` converts and this file goes through it like any world.
#: The v1 five are copied from the home world's real geometry rather than
#: invented, so a builder tested against the fixture is tested against the
#: house.
MARKERS = [
  Marker("wall", [("box", "0.02 4.0 0.6", "0 0 0.6")], collides=True,
         axes={"thickness": "size0", "length": "size1", "height": "size2"},
         build="replace",
         note="A slab standing on the floor. The two horizontal extents are "
              "not interchangeable: the SHORT one is the thickness, and a "
              "wall may run along either world axis -- read them body-local "
              "and let the body quat place the result."),
  Marker("fence", [("box", "2.5 0.02 0.45", "0 0 0.45")], collides=True,
         axes={"length": "size0", "thickness": "size1", "height": "size2"},
         build="replace",
         note="Like a wall but see-through: the browser draws posts and "
              "rails within the slab, which is why the marker is solid and "
              "the art is not."),
  Marker("floor", [("box", "3.5 4.0 0.001", "0 0 -0.001")], collides=False,
         axes={"extentX": "size0", "extentY": "size1", "thickness": "size2"},
         build="replace",
         note="Indoors underfoot. Does NOT collide -- the robot is held up "
              "by the world plane, and this is the surface a visitor sees."),
  Marker("ground", [("box", "2.5 4.0 0.001", "0 0 -0.001")], collides=False,
         axes={"extentX": "size0", "extentY": "size1", "thickness": "size2"},
         build="replace",
         note="Outdoors underfoot, same shape as `floor` and a different "
              "material. The pair is why the vocabulary has two names for "
              "one primitive."),
  Marker("whiteboard", [("box", "0.01 0.16 0.13", "0 0 0")], collides=True,
         axes={"depth": "size0", "width": "size1", "height": "size2"},
         build="replace",
         note="The DRAWN face is +x in the body frame, and its extent is "
              "size1 x size2 -- the same rectangle a `draw` event's polyline "
              "is in. `scene.boards` gives the same numbers by board name."),
  Marker("rack", [("box", "0.024 0.024 0.4", "-0.9 0 0.4"),
                  ("box", "0.024 0.024 0.4", "0.9 0 0.4"),
                  ("box", "0.9 0.024 0.024", "0 0 0.78")], collides=True,
         axes={}, build="reskin",
         note="MANY primitives, and the only v1 hint that is already a "
              "reskin: the bays are where the robot puts tools, so their "
              "positions are load-bearing and an asset that replaced them "
              "would put the tools somewhere else. Use `primitives()`."),
  Marker("plant", [("cylinder", "0.04 0.15", "0 0 0.15")], collides=False,
         axes={"radius": "size0", "height": "size1"},
         build="replace",
         note="A cylinder standing on the ground, and NOT solid: the robot "
              "maps it (the census counts it off the occupancy grid) and "
              "drives through it. Do not draw it as an obstacle."),
  # ---- v2 (issue #66) ------------------------------------------------------
  Marker("tree", [("cylinder", "0.06 0.9", "0 0 0.9")], collides=True,
         axes={"radius": "size0", "height": "size1"},
         build="replace",
         note="THE TRUNK, not the tree. Unlike `plant` it is solid, because "
              "a tree the robot may drive through is a tree it will park "
              "in. The canopy is art and may exceed the marker -- but only "
              "ABOVE the robot, because nothing outside this cylinder is "
              "something the robot will avoid."),
  Marker("hill", [("box", "1.0 1.0 0.15", "0 0 0.15")], collides=True,
         axes={"extentX": "size0", "extentY": "size1", "height": "size2"},
         build="replace",
         note="A mound INSIDE the fence, where the robot maps it -- the "
              "distant hills of the horizon are the browser's and are not "
              "hinted at all. The marker is the footprint and the peak "
              "height; the shape between them is the builder's."),
  Marker("couch", [("box", "0.9 0.4 0.35", "0 0 0.35")], collides=True,
         axes={"width": "size0", "depth": "size1", "height": "size2"},
         build="replace",
         note="EXACTLY ONE BOX -- see the note below on furniture. Width is "
              "the seating direction, depth is front-to-back."),
  Marker("bed", [("box", "1.0 0.7 0.25", "0 0 0.25")], collides=True,
         axes={"length": "size0", "width": "size1", "height": "size2"},
         build="replace",
         note="EXACTLY ONE BOX. Length is head-to-foot."),
  Marker("table", [("box", "0.6 0.4 0.37", "0 0 0.37")], collides=True,
         axes={"width": "size0", "depth": "size1", "height": "size2"},
         build="replace",
         note="EXACTLY ONE BOX, and the height is the TOP -- a builder draws "
              "legs and a top within it rather than a solid block."),
  Marker("robot", [("box", "0.105 0.08 0.045", "0 0 0"),
                   ("box", "0.05 0.04 0.06", "0 -0.08 0.135"),
                   ("cylinder", "0.045 0.012", "-0.105 0 0"),
                   ("cylinder", "0.045 0.012", "0.105 0 0")], collides=True,
         axes={}, build="reskin", body=ROBOT_ROOT, free=True,
         note="MANY primitives, and a RESKIN for the same reason as `rack` "
              "and a stronger one: this is the body a visitor watches, and "
              "its silhouette is the shape the physics used. A visitor "
              "watching it squeeze through a doorway is watching that "
              "shape, so dress the primitives -- never replace them with a "
              "prettier outline. It is also the only DYNAMIC hinted body: "
              "its pose is overwritten by every telemetry frame. ⚠ The hint "
              "goes on the ROOT body only; the head, wheels, carriage, arm "
              "and fork are bodies of their own with their own poses. They "
              "are already identifiable without a hint -- every body in the "
              "subtree carries `robot: \"pluggybot\"` -- so the hint names "
              "where to ANCHOR, and `robot` names what to dress."),
]

#: A furniture marker is EXACTLY ONE BOX (issue #66's second recorded
#: decision), and this is pinned rather than merely intended: the alternative
#: -- a builder keying on the marker's proportions to tell a couch from a bed
#: -- is what made three names better than one `furniture`, and it only works
#: while there is one box to take proportions of. `largestBox` selects by
#: VOLUME, so a couch modelled as a frame plus cushions could hand a builder
#: the wrong one. If the expanded house wants furniture that cannot be one
#: box, that is a bug to report on issue #66, not to work around in a builder.
ONE_BOX_HINTS = ("couch", "bed", "table")

_XML = """<mujoco model="hints">
  <worldbody>
{bodies}
  </worldbody>
</mujoco>
"""


def build() -> dict:
  """Compile the markers and transpile them exactly as a world is."""
  bodies = "\n".join(m.mjcf(y=i * 3.0) for i, m in enumerate(MARKERS))
  model = mujoco.MjModel.from_xml_string(_XML.format(bodies=bodies))
  meta = {"visualHints": {m.body: m.hint for m in MARKERS}}
  scene = scene_dict(model, "hints", meta=meta)
  by_name = {b["name"]: b for b in scene["bodies"]}
  return {
    "protocolVersion": PROTOCOL_VERSION,
    "model": "hints",
    "upAxis": scene["upAxis"],
    "vocabulary": list(VISUAL_HINTS),
    "oneBoxHints": list(ONE_BOX_HINTS),
    "hints": {
      m.hint: {
        "marker": ("many" if len(m.geoms) > 1
                   else by_name[m.body]["geoms"][0]["type"]),
        "markers": len(m.geoms),
        "collides": m.collides,
        "build": m.build,
        "axes": m.axes,
        "note": m.note,
        # Straight out of `scene_dict`: the body a scene would actually ship.
        "body": by_name[m.body],
      } for m in MARKERS
    },
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("-o", "--out", default="protocol/hints.json")
  args = parser.parse_args()
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  fixture = build()
  out.write_text(json.dumps(fixture, indent=1) + "\n")
  missing = set(VISUAL_HINTS) - set(fixture["hints"])
  print(f"{out}: {len(fixture['hints'])} of {len(VISUAL_HINTS)} hints"
        + (f" -- MISSING {sorted(missing)}" if missing else ""))


if __name__ == "__main__":
  main()
