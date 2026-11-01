"""The dressing rules (issue #71): scenery the sensors must ignore.

Issue #71 arrived mostly done: #68 shipped every furniture hint it names
(`couch`, `bed`, `table`, `counter`, and `stairs` besides), so what was left
was the one genuine decision and the guards that make the rules survivable.

⚠ THE DECISION: NO IN-WORLD TREES OR HILLS, the issue's own escape hatch
("if the expanded house does not actually need one, ship no in-world trees
and let the horizon carry the look" -- rooftop#127 owns the horizon). It is
also the only self-consistent outcome: the frozen conformance rules
(protocol/hints.json, #66) say a `tree` and a `hill` COLLIDE -- "a tree the
robot may drive through is a tree it will park in" -- while #71 requires
every decorative body non-colliding and THE MAP UNCHANGED. Nothing can be
all three of collidable, decorative and invisible to the map, so an in-world
tree is a real obstacle or it is a lie, and this world ships neither. The
day the house wants a real tree, it arrives as a mapped obstacle under the
#66 rules and `test_no_tree_or_hill_ships_without_meeting_its_own_rules`
below is the tollbooth.

⚠ WHY "non-colliding" IS NOT ENOUGH, measured: the LIDAR's raycast ignores
contype entirely -- the plants are contype 0 and are MAPPED. So decorative
safety has a second, independent requirement: stay out of the band the map
is laid from (the humanoid's LIDAR's points laid flat, `lidar.BAND_M` over
the ground, #581's stage 2), unless it lies on something solid or is meant
to be seen. `test_dressing_in_the_band_reaches_the_map` proves the failure
is real by planting a hostile mound and watching the map change.
"""

import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest


ROOT = Path(__file__).parent.parent

#: How near a decorative geom must lie to a solid one to be mapped as part
#: of it, m: a tag on a wall, a bulb on its pole.
ON_SOLID_M = 0.01


@pytest.fixture(scope="module")
def trio():
  import json
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  meta = json.loads((ROOT / "models" / "home_world.meta.json").read_text())
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data, meta["visualHints"]


def _body_geoms(model, bid):
  return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == bid]


def _root_name(model, b):
  while int(model.body_parentid[b]) != 0:
    b = int(model.body_parentid[b])
  return model.body(b).name


def _collides(model, g):
  return int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0


def test_no_tree_or_hill_ships_without_meeting_its_own_rules(trio):
  """The recorded decision, as a tollbooth rather than prose. If a future
  layout adds one, this fails until whoever adds it has read the module
  docstring's contradiction and decided which side wins: a REAL tree
  (collidable per hints.json, mapped, priced into the reserve's routes and
  the camera-safety survey) -- in which case extend this test to assert
  exactly that -- or no tree. There is no third, invisible-solid option."""
  model, _, hints = trio
  offenders = [b for b, h in hints.items() if h in ("tree", "hill")]
  assert not offenders, (
    f"{offenders} arrived hinted tree/hill: decide against the rules in "
    f"tests/test_dressing.py's docstring before shipping them")


def test_furniture_is_collidable_and_decor_is_not(trio):
  """Off the COMPILED model, both directions of #71's collision rule: the
  furniture the robot must plan around collides, and every fully
  non-colliding body is one of the known decorative kinds."""
  model, _, hints = trio
  for name, hint in hints.items():
    geoms = _body_geoms(model, model.body(name).id)
    any_solid = any(_collides(model, g) for g in geoms)
    if hint in ("couch", "bed", "table", "counter", "stairs", "wall",
                "fence", "whiteboard", "rack", "cage", "mouse", "plate"):
      assert any_solid, f"{name} ({hint}) stopped colliding"
    elif hint in ("floor", "ground", "sidewalk", "street", "plant"):
      assert not any_solid, f"{name} ({hint}) grew a collision"


def test_decor_stays_out_of_the_band_the_map_is_laid_from(trio):
  """The rule that actually keeps the map clean, asserted geometrically:
  every fully non-colliding body keeps its geoms clear of the band the
  map is laid from (the LIDAR's points laid flat, `lidar.BAND_M` over the
  ground) -- or lies on a solid geom, mapped as part of it (a tag on a
  wall, the garden light's bulb on its pole), or is a plant, which the
  robot is meant to see (`protocol/hints.json`'s note; the depth camera's
  layer planned round them before the band did). Collision flags cannot do
  this job -- the raycast ignores them. The house as served: its tags,
  the station and the counter go in at load (`humanoid.world`)."""
  from pluggybot.humanoid import world as hw
  from pluggybot.perception.lidar import BAND_M
  _, _, hints = trio
  model = hw.home_spec().compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  robots = {model.body(b).name for b in range(model.nbody)
            if model.body(b).name.endswith("pluggybot")}
  solid = [g for g in range(model.ngeom) if _collides(model, g)
           and int(model.geom_type[g]) != mujoco.mjtGeom.mjGEOM_PLANE]
  bad = []
  for b in range(1, model.nbody):
    name = model.body(b).name
    geoms = _body_geoms(model, b)
    if (not geoms or any(_collides(model, g) for g in geoms) or hints.get(name) == "plant"
        or name in robots or _root_name(model, b) in robots):
      continue                       # solid things are the MAP's business, robots their own
    for g in geoms:
      # The real oriented z-extent (|R| @ half, the world-budget lesson):
      # the first cut used max(geom_size) as a radius, and a 2 mm floor slab
      # "spanned" +-4 m because its HORIZONTAL half-extent is 4 m.
      t, size = int(model.geom_type[g]), model.geom_size[g]
      if t == mujoco.mjtGeom.mjGEOM_BOX:
        half = np.array(size, dtype=float)
      elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        half = np.array([size[0]] * 3)
      elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        half = np.array([size[0], size[0], size[1]])
      elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        half = np.array([size[0], size[0], size[1] + size[0]])
      else:
        half = np.array([float(max(size))] * 3)
      rz = float((np.abs(data.geom_xmat[g].reshape(3, 3)) @ half)[2])
      z = float(data.geom_xpos[g][2])
      if z + rz < BAND_M[0] or z - rz > BAND_M[1]:
        continue
      near = [h for h in solid if np.linalg.norm(data.geom_xpos[g] - data.geom_xpos[h])
              <= model.geom_rbound[g] + model.geom_rbound[h] + ON_SOLID_M]
      if any(mujoco.mj_geomDistance(model, data, g, h, ON_SOLID_M, None) <= ON_SOLID_M
             for h in near):
        continue
      bad.append(f"{name}/{model.geom(g).name} spans z [{z - rz:.3f},{z + rz:.3f}] in the "
                 f"band, on nothing solid")
  assert not bad, "\n".join(bad)


def _scan_grid(xml_path):
  """One scan's occupancy grid from a fixed pose -- the shortest honest map:
  the LIDAR sees all round, so the scan's seam is called once, standing."""
  from pluggybot.humanoid import world as hw
  from pluggybot.humanoid.body import HumanoidBody
  from pluggybot.lifecycle import HUMANOID_HOME, world_config

  model = hw.home_spec(path=str(xml_path)).compile()
  body = HumanoidBody(model, mujoco.MjData(model), realtime=False,
                      grid_bounds=world_config(HUMANOID_HOME)["grid_bounds"])
  try:
    body.start_at(7.5, -4.0, 0.0)    # garden_south, where the probe plants
    body.mission._scan_step()
    return np.asarray(body.mission.grid.grid).copy()
  finally:
    body.close()


def test_dressing_in_the_band_reaches_the_map():
  """The failure #71's placement rules exist to prevent, demonstrated: a
  non-colliding 1.6 m mound in the garden -- exactly a hill a dresser might
  think 'safe' because contype is 0 -- changes the occupancy grid, because
  the LIDAR raycast never looked at contype. In the real world it would be
  a phantom the planner routes around; here it is the proof that
  `test_decor_stays_out_of_the_band_the_map_is_laid_from` is about
  something.

  The clean scan is taken twice first: determinism is what makes the
  hostile diff meaningful rather than noise.
  """
  scratch = Path(tempfile.mkdtemp(prefix="dressed_home_"))
  for entry in (ROOT / "models").iterdir():
    (scratch / entry.name).symlink_to(entry)
  xml = (ROOT / "models" / "home_world.xml").read_text()
  mound = '''
    <body name="hostile_mound" pos="8.5 -4.0 0.8">
      <geom name="hostile_mound_geom" type="box" size="0.4 0.4 0.8"
            contype="0" conaffinity="0" rgba="0.4 0.5 0.35 1"/>
    </body>
  </worldbody>'''
  hostile = scratch / "hostile.xml"
  hostile.write_text(xml.replace("  </worldbody>", mound, 1))

  clean_a = _scan_grid(ROOT / "models" / "home_world.xml")
  clean_b = _scan_grid(ROOT / "models" / "home_world.xml")
  assert np.array_equal(clean_a, clean_b), \
      "the scan is not deterministic; the hostile diff below means nothing"
  dressed = _scan_grid(hostile)
  changed = int(np.count_nonzero(clean_a != dressed))
  assert changed > 0, (
    "a non-colliding mound in the beam left no trace on the map -- the "
    "LIDAR started honouring contype, and the band's rule needs "
    "re-deriving")


def test_every_hinted_body_validates_against_its_conformance_rule(trio):
  """The loop #66 built, closed: the world's bodies checked against the
  same protocol/hints.json the website builds art against. `collides` and
  the one-box rule are the halves a generator edit can silently break."""
  import json
  fixture = json.loads((ROOT / "protocol" / "hints.json").read_text())
  model, _, hints = trio
  for name, hint in hints.items():
    rule = fixture["hints"][hint]
    geoms = _body_geoms(model, model.body(name).id)
    any_solid = any(_collides(model, g) for g in geoms)
    assert any_solid == rule["collides"], (
      f"{name} ({hint}): collides={any_solid} but the conformance rule "
      f"says {rule['collides']}")
    if hint in fixture["oneBoxHints"]:
      assert len(geoms) == 1, f"{name} ({hint}) is {len(geoms)} geoms, " \
                              f"not the ONE box its rule pins"
      assert model.geom(geoms[0]).type == mujoco.mjtGeom.mjGEOM_BOX, \
          f"{name} ({hint}) is not a box"
