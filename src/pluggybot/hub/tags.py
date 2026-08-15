"""AprilTags for the hub: real tag36h11 markers, generated and detected.

This replaces the colour-plate stand-in. What the real thing buys, in the
order it matters here:

  IDENTITY   every tag carries a decoded ID, so "which fiducial is this?"
             stops being a geometry puzzle. The stand-in had to guess by
             size and image position, and it guessed wrong once -- steering
             onto the charge bay's marker and dragging a module 22 cm
             toward the wrong bay. That class of bug is now impossible.
  POSE       one tag yields a full 6-DoF pose by PnP from its four corners,
             so range comes from the marker itself. No depth buffer, which
             also means no second render per look: on hardware there is no
             depth camera to consult, and now there needs to be none.
  ROBUSTNESS decoding is a thresholded, pattern-based pipeline with error
             correction (36h11 corrects up to 4 bit errors), rather than a
             colour key that any orange object could trip.

Tag sizes are a real design decision, not a detail: a tag36h11 must span
roughly 25-30 px before it decodes, so a marker's physical size sets the
range at which the robot can see it. The rack's marker is big because it is
read from across the room; the bay and module markers are small because
they are only ever read from arm's length.
"""

from pathlib import Path

import numpy as np

FAMILY = "tag36h11"
QUIET_CELLS = 1          # white margin around the tag, in tag cells -- the
                         # detector needs it to find the quad's outer edge
PIXELS_PER_CELL = 24     # render scale of the generated PNGs

# Tag identities. The whole point of real tags: these are decoded, not
# inferred from where a blob happened to sit in the frame.
RACK_TAG_ID = 0
BAY_TAG_IDS = (1, 2, 4, 5, 6)  # bays A-E -- same ORDER as HUB_STATION_YS,
                         # which is what pairs them (coupling.bay_tag_id). Not
                         # contiguous because 3 was already the charge bay and
                         # renumbering would invalidate every generated tag
                         # PNG and rack model for cosmetics. Bay E (id 6) came
                         # with the seed dispenser, appended for the same
                         # reason the station tuple is appended to.
CHARGE_TAG_ID = 3
MODULE_TAG_IDS = {"module_lcd": 10, "module_plug": 11, "module_pen": 12,
                  "module_claw": 13, "module_seed": 14}

# Physical marker sizes (m), edge of the BLACK tag -- what the detector is
# told, and what PnP scales its translation by. The plate carrying it is
# larger by the quiet zone.
RACK_TAG_SIZE = 0.120
SMALL_TAG_SIZE = 0.030

# Which physical size each id is, so one detection pass can serve markers of
# different sizes: PnP translation scales linearly with the assumed tag
# size, so a single decode can be rescaled per id exactly.
TAG_SIZES = {RACK_TAG_ID: RACK_TAG_SIZE, CHARGE_TAG_ID: SMALL_TAG_SIZE,
             **{i: SMALL_TAG_SIZE for i in BAY_TAG_IDS},
             **{i: SMALL_TAG_SIZE for i in MODULE_TAG_IDS.values()}}

TAG_DIR = Path("models/tags")


def plate_half_extent(tag_size: float) -> float:
  """Half-size of the plate that carries a tag of this size, including the
  quiet zone (the texture spans the whole plate face)."""
  cells = 8 + 2 * QUIET_CELLS               # tag36h11 is 8x8 with its border
  return tag_size * (cells / 8.0) / 2.0


def tag_image(tag_id: int) -> np.ndarray:
  """One tag36h11 bitmap with its white quiet zone, as a uint8 image."""
  from moms_apriltag import TagGenerator2
  tag = np.array(TagGenerator2(FAMILY).generate(tag_id), dtype=np.uint8)
  n = tag.shape[0]
  out = np.full((n + 2 * QUIET_CELLS, n + 2 * QUIET_CELLS), 255, dtype=np.uint8)
  out[QUIET_CELLS:QUIET_CELLS + n, QUIET_CELLS:QUIET_CELLS + n] = tag
  # Nearest-neighbour upscale: cell edges must stay hard. Interpolation here
  # is the segmentation-MSAA lesson in another costume -- a marker is data.
  return np.kron(out, np.ones((PIXELS_PER_CELL, PIXELS_PER_CELL), dtype=np.uint8))


def write_tag_pngs(directory: Path = TAG_DIR) -> list[int]:
  """Write every tag PNG the hub worlds reference. Returns the ids written."""
  from PIL import Image
  directory.mkdir(parents=True, exist_ok=True)
  ids = [RACK_TAG_ID, CHARGE_TAG_ID, *BAY_TAG_IDS, *MODULE_TAG_IDS.values()]
  for tag_id in ids:
    Image.fromarray(tag_image(tag_id)).save(directory / f"tag{tag_id}.png")
  return ids


def asset_xml(ids) -> str:
  """MJCF <asset> entries for the tag textures and their materials."""
  parts = []
  for tag_id in ids:
    # type="cube", not "2d": a 2d texture is mapped through geom texcoords,
    # which primitives do not carry -- a box painted that way renders flat
    # grey (measured: the plate looked blank and nothing ever decoded). Cube
    # mapping puts the same image squarely on every face, which is what a
    # printed marker glued to a plate looks like anyway.
    parts.append(
      f'<texture name="tagtex{tag_id}" type="cube" file="tags/tag{tag_id}.png"/>')
    parts.append(
      f'<material name="tagmat{tag_id}" texture="tagtex{tag_id}" '
      f'specular="0.05" shininess="0.05" reflectance="0"/>')
  return "\n    ".join(parts)


_DETECTOR = None


def _shared_detector():
  """One AprilTag detector for the whole process.

  Not an optimisation -- a crash fix. pupil-apriltags frees its tag family in
  `Detector.__del__` (apriltag_detector_destroy -> clear_families ->
  quick_decode_uninit), and a mission holds TWO detectors: one on the dock
  camera and one in the RackFinder. When the second was collected the family
  was freed twice and the process took SIGSEGV, inside `HubMission.close()`,
  killing whole test runs at random.

  Worth recording how nearly this was missed: the same crash had shown up
  earlier as a stack dump printed AFTER pytest reported "120 passed", which
  read like harmless interpreter-shutdown noise and got waved off as
  cosmetic. It was the same double-free landing at a different moment. A
  segfault is never cosmetic; a green summary line printed before one does
  not mean the run was clean.

  Detection here is synchronous and single-threaded, so one detector serves
  every camera, and it is deliberately never destroyed: nothing frees it
  twice if nothing frees it at all.
  """
  global _DETECTOR
  if _DETECTOR is None:
    from pupil_apriltags import Detector
    _DETECTOR = Detector(families=FAMILY, nthreads=2, quad_decimate=1.0)
  return _DETECTOR


class TagDetector:
  """Renders one camera and decodes the AprilTags in it.

  Returns detections keyed by ID, each with the tag's translation in the
  CAMERA frame (x right, y down, z forward) from PnP, plus its centre pixel.
  """

  def __init__(self, model, camera_name: str, width: int = 1280,
               height: int = 720, tag_size: float = SMALL_TAG_SIZE) -> None:
    import mujoco
    self.renderer = mujoco.Renderer(model, height, width)
    self.camera_name = camera_name
    self.width, self.height = width, height
    self.tag_size = tag_size
    fovy = float(model.camera(camera_name).fovy[0])
    f = (height / 2) / np.tan(np.radians(fovy) / 2)
    self.camera_params = (f, f, width / 2, height / 2)
    self.detector = _shared_detector()

  def detect(self, data) -> dict:
    """{tag_id: {"t": (x, y, z) in camera frame, "center": (u, v)}}.

    One render, one decode, all sizes: PnP translation is linear in the
    assumed tag size, so each id's pose is rescaled from the nominal size
    to its own. Cheaper than a pass per marker size, and exact.
    """
    self.renderer.update_scene(data, camera=self.camera_name)
    rgb = self.renderer.render()
    gray = np.ascontiguousarray(
      (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
       + 0.114 * rgb[:, :, 2]).astype(np.uint8))
    found = self.detector.detect(
      gray, estimate_tag_pose=True, camera_params=self.camera_params,
      tag_size=self.tag_size)
    out = {}
    for det in found:
      tag_id = int(det.tag_id)
      scale = TAG_SIZES.get(tag_id, self.tag_size) / self.tag_size
      out[tag_id] = {
        "t": tuple(float(v) * scale for v in np.asarray(det.pose_t).ravel()),
        "center": (float(det.center[0]), float(det.center[1])),
      }
    return out

  def close(self) -> None:
    self.renderer.close()
