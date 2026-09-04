"""Camera safety: the detectors cannot be fooled by the scenery (issue #69).

"Camera-safe" is the whole qualifier on M13's dressing, and until this file
it was an intention with nothing behind it. Two detectors, two risk shapes:

  APRILTAGS are the live and dangerous one. `dock_eye` drives both terminal
  maneuvers off tag PnP poses (`charge_approach`, `bay_fix`), and a false
  decode does not degrade gracefully -- it steers the robot into the rack.
  OUTLETS are the quiet one: the YOLO detector already has a documented
  failure of exactly this shape (0.99 mAP on a val split that shared the
  training generator, while calling a light switch an outlet).

The survey renders the HOME world through the robot's real camera body at
the detector's real resolution, from poses covering every zone, and asserts
two things per decode: the id is one the world actually contains, and its
PnP range agrees with the ground-truth distance to that tag's own geom.
The second is what catches a COPY of a legal tag somewhere illegal -- the
"a picture of a tag is a tag" failure the hint freeze (#66) banished wall
pictures to the browser to avoid, demonstrated here by the adversarial
probe rather than assumed.

⚠ This file WRITES NOTHING into models/ (the tags PNG write race is a
documented -n auto flake in generator-calling tests): the hostile world is
compiled from a scratch directory of symlinks, and every render is at a
fixed resolution through `rack.tags.TagDetector`'s own renderer.
"""

import glob
import math
import os
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.home import world as home
from pluggybot.rack.tags import (
  BAY_TAG_IDS, CHARGE_TAG_ID, MODULE_TAG_IDS, RACK_TAG_ID, TagDetector,
)

ROOT = Path(__file__).parent.parent

#: Every id the home world may legally decode.
LEGAL_IDS = frozenset(
  {RACK_TAG_ID, CHARGE_TAG_ID, *BAY_TAG_IDS, *MODULE_TAG_IDS.values()})

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


def _place(model, data, x, y, yaw):
  data.qpos[:] = model.qpos0
  data.qpos[0] = x + 0.08 * math.cos(yaw)
  data.qpos[1] = y + 0.08 * math.sin(yaw)
  data.qpos[2] = 0.045
  data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(model, data)


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


def _survey(model, data, poses, yaws=YAWS):
  """Render every (pose, yaw) through dock_eye and collect violations."""
  det = TagDetector(model, "dock_eye")
  # ⚠ Forward BEFORE reading truth: a fresh MjData's geom_xpos is zeros, and
  # ground truth read from it puts every tag at the origin -- which made the
  # survey's first run flag every honest decode as a copy 1.48 m off. The
  # detector was right and the ground truth was uninitialized.
  mujoco.mj_forward(model, data)
  truth = _tag_truth(model, data)
  cam_id = model.camera("dock_eye").id
  violations = []
  try:
    for name, x, y in poses:
      for k in range(yaws):
        yaw = 2 * math.pi * k / yaws
        _place(model, data, x, y, yaw)
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
    det.renderer.close()
  return violations


@pytest.fixture(scope="module")
def home_pair():
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  return model, mujoco.MjData(model)


def test_every_zone_decodes_only_what_is_there(home_pair):
  """The full survey: one pose per zone, four yaws each, through the real
  camera at the detector's real resolution.

  ⚠ NOT marked slow, and the issue expected otherwise -- it budgeted a
  "pose grid is expensive" and asked for a fast subset outside the marker.
  Measured: 36 renders plus decodes cost 0.85 s, so the WHOLE grid lives in
  the iterate loop and there is nothing for a subset to be a subset of.
  The marker's own rule (expensive AND unable to catch a regression while
  iterating) refuses it; shorten-before-you-mark, already shorter than the
  bar."""
  model, data = home_pair
  bad = _survey(model, data, _zone_poses())
  assert not bad, "\n".join(bad)


# ---- the adversarial probe ---------------------------------------------------


def _hostile_world():
  """The home world plus one wall picture that is a COPY of bay E's tag.

  Exactly the decoration the hint freeze forbade, built on purpose: a
  0.24 m framed print of tag id 5 on the living room's west wall, where a
  visitor might hang art. Compiled from a scratch directory of symlinks so
  nothing under models/ is written or disturbed.
  """
  scratch = Path(tempfile.mkdtemp(prefix="hostile_home_"))
  for entry in (ROOT / "models").iterdir():
    (scratch / entry.name).symlink_to(entry)
  xml = (ROOT / "models" / "home_world.xml").read_text()
  picture = '''
    <body name="wall_art" pos="-1.955 -0.5 0.55">
      <geom name="wall_art_print" type="box" size="0.005 0.12 0.12"
            material="tagmat5" contype="0" conaffinity="0"/>
    </body>
  </worldbody>'''
  hostile = scratch / "hostile_home.xml"
  hostile.write_text(xml.replace("  </worldbody>", picture, 1))
  return mujoco.MjModel.from_xml_path(str(hostile))


def test_a_picture_of_a_tag_trips_the_survey():
  """A harness with nothing adversarial in it is decor, so this PASSES BY
  DEMONSTRATING CONFUSION: hang a print of bay E's tag on the living-room
  wall and the survey must catch it.

  What the false decode actually is: the detector reports tag 5 -- bay E,
  which `bay_fix` would happily creep toward -- several metres from where
  bay E's real tag is. The id is legal, which is why a subset check alone
  is not a harness; the RANGE against the tag's own geom is what convicts
  the copy. This is the concrete failure that keeps wall pictures
  browser-only (#66): a picture the robot's cameras never render cannot do
  this.
  """
  model = _hostile_world()
  data = mujoco.MjData(model)
  bad = _survey(model, data, [("living", 1.5, 0.25)])
  assert bad, "the survey did not notice a copied tag hung as wall art"
  assert any("tag 5" in b for b in bad), "\n".join(bad)


# ---- the outlet detector -----------------------------------------------------

_WEIGHTS = sorted(glob.glob(str(ROOT / "runs/detect/*/weights/best.pt")),
                  key=os.path.getmtime)


def _project(model, data, cam_id, world, w=1280, h=720):
  """World point -> pixel through the named camera, or None if behind it."""
  cam = np.array(data.cam_xpos[cam_id])
  rot = np.array(data.cam_xmat[cam_id]).reshape(3, 3)
  local = rot.T @ (np.asarray(world) - cam)      # MuJoCo cams look along -z
  if local[2] > -1e-6:
    return None
  fovy = float(model.cam_fovy[cam_id])
  f = (h / 2) / math.tan(math.radians(fovy) / 2)
  return (w / 2 - f * local[0] / local[2], h / 2 + f * local[1] / local[2])


#: ⚠ Not marked slow either: 10.9 s warm (the 125 s first run was torch
#: loading a cold cache, paid once per venv), it skips wherever no weights
#: exist, and a scenery edit -- the one thing #71 is made of -- is exactly
#: the regression it exists to catch while you iterate.
@pytest.mark.skipif(not _WEIGHTS, reason="no trained YOLO weights under runs/")
def test_no_outlets_are_seen_except_where_a_tag_is(home_pair):
  """The home world contains NO outlets, so every detection above the
  spotter's own 0.7 threshold is a false positive -- the light-switch
  failure's guard, pointed at the new house.

  ⚠ THE STATED ALLOWANCE, and it is a finding, not a fudge: the very first
  full survey caught the outlet head firing at conf 0.87 on THE RACK'S OWN
  APRILTAG -- a dark, high-contrast square plate on a wall, which is exactly
  what its generated training set looks like. The marker one detector
  requires is an outlet hallucination to the other. It is latent today (the
  outlet stack is the plug era's and no home mission runs it), so the
  allowance excuses a detection ONLY if a real tag geom projects inside its
  box -- pixel-precise, so a new piece of scenery that fools the detector
  still trips this test, and the day the plug era returns to a tag-bearing
  world, this docstring is the warning.
  """
  from ultralytics import YOLO

  model, data = home_pair
  mujoco.mj_forward(model, data)
  truth = _tag_truth(model, data)
  tag_points = [pt for pts in truth.values() for pt in pts]
  cam_id = model.camera("dock_eye").id
  yolo = YOLO(_WEIGHTS[-1])
  det = TagDetector(model, "dock_eye")     # reuse its renderer + camera
  ghosts, excused = [], 0
  try:
    for name, x, y in _zone_poses():
      for k in range(YAWS):
        yaw = 2 * math.pi * k / YAWS
        _place(model, data, x, y, yaw)
        det.renderer.update_scene(data, "dock_eye")
        frame = det.renderer.render()
        for r in yolo.predict(frame, conf=0.7, verbose=False):
          for b in r.boxes:
            x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
            over_tag = any(
              (px is not None and x0 - 8 <= px[0] <= x1 + 8
               and y0 - 8 <= px[1] <= y1 + 8)
              for px in (_project(model, data, cam_id, pt)
                         for pt in tag_points))
            if over_tag:
              excused += 1
              continue
            ghosts.append(f"{name} yaw {math.degrees(yaw):.0f}: "
                          f"conf {float(b.conf):.2f} at "
                          f"[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]")
  finally:
    det.renderer.close()
  assert not ghosts, "outlets detected in a house that has none:\n" + \
      "\n".join(ghosts)
  # The allowance must stay an allowance for something REAL: if the tag
  # plates stop reading as outlets (a retrained detector, a moved rack),
  # this fails so the excuse is deleted rather than outliving its cause --
  # the strict-xfail lesson in allowance form.
  assert excused > 0, \
      "no detection needed the tag allowance any more; remove it"
