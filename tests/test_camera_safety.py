"""Camera safety: the tag detector cannot be fooled by the scenery (issue #69).

"Camera-safe" is the whole qualifier on M13's dressing. The humanoid's
head camera (`nav_eye`) finds its places and its charging station off
AprilTag PnP poses -- the station's pair, a task area's tag, a plate's sign
-- and a false decode does not degrade gracefully: it steers the robot
into them.

The survey renders the home world through the robot's real camera at the
detector's real resolution, from poses covering every zone, and asserts
two things per decode: the id is one the world actually contains, and its
PnP range agrees with the ground-truth distance to that tag's own geom.
The second is what catches a COPY of a legal tag somewhere illegal -- the
"a picture of a tag is a tag" failure the hint freeze (#66) banished wall
pictures to the browser to avoid, demonstrated here by the adversarial
probe rather than assumed.

⚠ This file WRITES NOTHING into models/ (the tags PNG write race is a
documented -n auto flake in generator-calling tests): the hostile world is
the served world's spec with one body added, and every render is at a
fixed resolution through `perception.tags.TagDetector`'s own renderer.
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.home import world as home
from pluggybot.home.areas import area_ids
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import world as hw
from pluggybot.perception.tags import (
  BLOCK_TAG_IDS, BOARD_TAG_IDS, CHEST_TAG_IDS, COUNTER_TAG_IDS, CUBE_TAG_IDS, DOOR_TAG_IDS,
  DRAWER_TAG_IDS, MASS_TAG_IDS, MAT_TAG_IDS, PLATE_TAG_IDS, STATION_TAG_IDS, TagDetector,
)
from pluggybot.robot import FIRST

#: Every id the home world may legally decode: the lab's plate signs, the
#: whiteboards' pairs, the tower's blocks, the bench's masses, the tower's,
#: bench's and garden's area tags, the charging station's pair, and the
#: storeroom counter's row and its props' (#559).
LEGAL_IDS = frozenset({*PLATE_TAG_IDS, *(t for ids in BOARD_TAG_IDS.values() for t in ids),
                       *BLOCK_TAG_IDS, *MASS_TAG_IDS, *area_ids(), *STATION_TAG_IDS,
                       *COUNTER_TAG_IDS, *CHEST_TAG_IDS, *DRAWER_TAG_IDS, *DOOR_TAG_IDS,
                       *CUBE_TAG_IDS, *MAT_TAG_IDS})

#: PnP range vs ground-truth distance, worst case, clean world. Measured on
#: the full survey: the translation half of a tag pose is millimetre-true
#: (the AMBIGUOUS half is the yaw -- issue #88 -- which this file does not
#: lean on); 0.25 m is an order of magnitude of headroom over the measured
#: worst while staying far below the "copy of a tag on a wall" signature,
#: which mis-ranges by METRES because the copy is not where the original is.
RANGE_TOL_M = 0.25

#: One standing pose per zone, at the zone's centre -- every zone, which is
#: the acceptance -- plus this many evenly spaced yaws at each.
YAWS = 4


def _zone_poses():
  poses = []
  for z in home.ZONES:
    cx = (z["min"][0] + z["max"][0]) / 2.0
    cy = (z["min"][1] + z["max"][1]) / 2.0
    if z["name"] == "hall":
      cy = 1.0          # the centre of the hall's OPEN half; y=0 is fine
                        # too, but the stairs block y in [-6,-3] and a pose
                        # inside solid geometry surveys the inside of a box
    poses.append((z["name"], cx, cy))
  return poses


def _tag_truth(model, data):
  """id -> world positions of every geom textured with that tag."""
  truth: dict[int, list] = {}
  for g in range(model.ngeom):
    mid = int(model.geom_matid[g])
    if mid < 0:
      continue
    name = model.mat(mid).name or ""
    if name.startswith("tagmat"):
      truth.setdefault(int(name[6:]), []).append(
        np.array(data.geom_xpos[g], dtype=float))
  return truth


def _stand(model, data, x, y, yaw):
  """The humanoid upright at (x, y) facing `yaw`, as `HumanoidBody.start_at`
  stands it, and forwarded."""
  q = FIRST.qpos_adr(model)
  data.qpos[q:q + 3] = [x, y, hm.stand_height()]
  data.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  stand = hm.stand_qpos()
  for name in hm.qpos_order():
    data.qpos[model.jnt_qposadr[model.joint(FIRST.el(name)).id]] = stand.get(name, 0.0)
  mujoco.mj_forward(model, data)


def _survey(model, data, poses, yaws=YAWS):
  """Render every (pose, yaw) through nav_eye and collect violations."""
  det = TagDetector(model, "nav_eye")
  # ⚠ Forward BEFORE reading truth: a fresh MjData's geom_xpos is zeros, and
  # ground truth read from it puts every tag at the origin -- which made the
  # survey's first run flag every honest decode as a copy 1.48 m off. The
  # detector was right and the ground truth was uninitialized.
  mujoco.mj_forward(model, data)
  truth = _tag_truth(model, data)
  cam_id = model.camera("nav_eye").id
  violations = []
  try:
    for name, x, y in poses:
      for k in range(yaws):
        yaw = 2 * math.pi * k / yaws
        _stand(model, data, x, y, yaw)
        cam = np.array(data.cam_xpos[cam_id], dtype=float)
        for tid, d in det.detect(data).items():
          where = f"{name} ({x:.1f},{y:.1f}) yaw {math.degrees(yaw):.0f}"
          if tid not in LEGAL_IDS:
            violations.append(f"{where}: decoded UNKNOWN tag id {tid}")
            continue
          rng = float(np.linalg.norm(d["t"]))
          best = min(float(np.linalg.norm(p - cam)) for p in truth[tid])
          if abs(rng - best) > RANGE_TOL_M:
            violations.append(
              f"{where}: tag {tid} ranges {rng:.2f} m but its nearest real "
              f"geom is {best:.2f} m away -- a copy somewhere illegal?")
  finally:
    det.close()
  return violations


@pytest.fixture(scope="module")
def home_pair():
  model = hw.home_spec().compile()
  return model, mujoco.MjData(model)


def test_every_zone_decodes_only_what_is_there(home_pair):
  """The full survey: one pose per zone, four yaws each, through the real
  camera at the detector's real resolution. NOT marked slow: 96 renders
  plus decodes cost ~3 s, and the marker's own rule (expensive AND unable
  to catch a regression while iterating) refuses it."""
  model, data = home_pair
  bad = _survey(model, data, _zone_poses())
  assert not bad, "\n".join(bad)


# ---- the adversarial probe ---------------------------------------------------


#: The tag the hostile print copies: the charging station's first.
COPIED = STATION_TAG_IDS[0]


def _hostile_world():
  """The home world plus one wall picture that is a COPY of the charging
  station's first tag.

  Exactly the decoration the hint freeze forbade, built on purpose: a
  0.24 m framed print of the station's tag on the living room's west wall,
  at a head camera's height, where a visitor might hang art. Added to the
  served world's spec, so nothing under models/ is written or disturbed.
  """
  spec = hw.home_spec()
  art = spec.worldbody.add_body(name="wall_art", pos=[-1.955, -0.5, 1.2])
  art.add_geom(name="wall_art_print", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[0.005, 0.12, 0.12], material=f"tagmat{COPIED}",
               contype=0, conaffinity=0)
  return spec.compile()


def test_a_picture_of_a_tag_trips_the_survey():
  """A harness with nothing adversarial in it is decor, so this PASSES BY
  DEMONSTRATING CONFUSION: hang a print of the station's tag on the
  living-room wall and the survey must catch it.

  What the false decode actually is: the detector reports the station's tag
  -- which a charge would walk toward -- at a range that is no such tag's.
  The id is legal, which is why a subset check alone is not a harness; the
  RANGE against the tags of that id is what convicts the copy. This is the
  concrete failure that keeps wall pictures browser-only (#66): a picture
  the robot's cameras never render cannot do this.
  """
  model = _hostile_world()
  data = mujoco.MjData(model)
  bad = _survey(model, data, [("living", 1.5, 0.25)])
  assert bad, "the survey did not notice a copied tag hung as wall art"
  assert any(f"tag {COPIED}" in b for b in bad), "\n".join(bad)
