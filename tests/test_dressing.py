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
contype entirely -- the plants are contype 0 and are MAPPED (the census
counts them off the occupancy grid; that is the task). So decorative safety
has a second, independent requirement: stay out of the beam plane
(z = 0.223 m). `test_dressing_in_the_beam_reaches_the_map` proves the
failure is real by planting a hostile mound and watching the map change.
"""

import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest


ROOT = Path(__file__).parent.parent

#: The beam's absolute height, with clearance either side for the sprung
#: chassis breathing. A decorative geom whose z-extent crosses this band is
#: in the map whether it collides or not.
BEAM_Z = 0.223
BEAM_CLEAR = 0.05

#: The one deliberate exception: plants LIVE in the beam. They are the
#: census's subject -- non-colliding so the mission is exactly as hard as
#: the walls make it, in the beam so the robot can count them.
BEAM_EXEMPT_HINTS = frozenset({"plant"})


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
                "fence", "whiteboard", "rack"):
      assert any_solid, f"{name} ({hint}) stopped colliding"
    elif hint in ("floor", "ground", "sidewalk", "street", "plant"):
      assert not any_solid, f"{name} ({hint}) grew a collision"


def test_decor_stays_out_of_the_lidar_beam(trio):
  """The rule that actually keeps the map clean, asserted geometrically:
  every fully non-colliding body (except the plants, see the module
  docstring) keeps all its geoms clear of the beam band. Collision flags
  cannot do this job -- the raycast ignores them."""
  model, data, hints = trio
  bad = []
  for b in range(1, model.nbody):
    name = model.body(b).name
    geoms = _body_geoms(model, b)
    if not geoms or any(_collides(model, g) for g in geoms):
      continue                       # solid things are the MAP's business
    if hints.get(name) in BEAM_EXEMPT_HINTS:
      continue
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
      if z - rz <= BEAM_Z + BEAM_CLEAR and z + rz >= BEAM_Z - BEAM_CLEAR:
        bad.append(f"{name}/{model.geom(g).name} spans "
                   f"z [{z - rz:.3f},{z + rz:.3f}] across the beam")
  assert not bad, "\n".join(bad)


def _scan_grid(xml_path):
  """One spin's occupancy grid from a fixed pose -- the shortest honest map."""
  from pluggybot.mission.mission import HubMission
  from pluggybot.lifecycle import world_config

  cfg = world_config("home")
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  m = HubMission(model, data, viewer=None, realtime=False,
                 rack=cfg["rack"], grid_bounds=cfg["grid_bounds"])
  try:
    m.start_at(7.5, -4.0, 0.0)       # garden_south, where the probe plants
    m._spin()
    return np.asarray(m.grid.grid).copy()
  finally:
    m.close()


def test_dressing_in_the_beam_reaches_the_map():
  """The failure #71's placement rules exist to prevent, demonstrated: a
  non-colliding 0.3 m mound in the garden -- exactly a hill a dresser might
  think 'safe' because contype is 0 -- changes the occupancy grid, because
  the LIDAR raycast never looked at contype. In the real world it would be
  a phantom the planner routes around or the census counts; here it is the
  proof that `test_decor_stays_out_of_the_lidar_beam` is about something.

  The clean scan is taken twice first: determinism is what makes the
  hostile diff meaningful rather than noise.
  """
  scratch = Path(tempfile.mkdtemp(prefix="dressed_home_"))
  for entry in (ROOT / "models").iterdir():
    (scratch / entry.name).symlink_to(entry)
  xml = (ROOT / "models" / "home_world.xml").read_text()
  mound = '''
    <body name="hostile_mound" pos="8.5 -4.0 0.15">
      <geom name="hostile_mound_geom" type="box" size="0.4 0.4 0.15"
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
    "LIDAR started honouring contype, and the beam-band rule (plus the "
    "plants' whole census) needs re-deriving")


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


def test_the_beam_constant_is_where_the_sensor_actually_rides(trio):
  """`BEAM_Z` above is a copy of where the scan really is; a copy that can
  drift is a rule about nothing. So it is pinned against the COMPILED
  robot: the world z of the `lidar` site the scanner itself ray-casts from
  (`perception.lidar` takes it by name), at rest pose. Move the sensor and
  this fails until the band -- and every judgement made with it -- follows."""
  model, data, _ = trio
  z = float(data.site_xpos[model.site("lidar").id][2])
  assert abs(z - BEAM_Z) < 0.01, \
      f"the lidar site rides at z={z:.3f}, BEAM_Z says {BEAM_Z}"
