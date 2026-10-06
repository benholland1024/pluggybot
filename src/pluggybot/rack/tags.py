"""AprilTags: real tag36h11 markers, generated and detected, and the one
registry of which id is which.

What real tags buy over the colour plates they replaced, in the order it
matters here:

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
range at which the robot can see it. A plate's sign is big because it is
read from across the lab; a rack's are read from the working pose.
"""

import io
import os
import threading
from pathlib import Path

import numpy as np

FAMILY = "tag36h11"
QUIET_CELLS = 1          # white margin around the tag, in tag cells -- the
                         # detector needs it to find the quad's outer edge
PIXELS_PER_CELL = 24     # render scale of the generated PNGs

# Tag identities. The whole point of real tags: these are decoded, not
# inferred from where a blob happened to sit in the frame. 0-9 were the
# rover's rack, charge bay and bays (#376 deleted them; not reused).
#: The five hand-built modules, by the ids they carried on the rover's rack
#: -- the workshop's permanent originals (`workshop.seam.HAND_BUILT`).
MODULE_TAG_IDS = {"module_lcd": 10, "module_plug": 11, "module_pen": 12,
                  "module_claw": 13, "module_seed": 14}
#: The tower's three blocks (issue #207; challenge/stack.py), after the
#: rover's built-tool ids 15-19 (a tool on legs carries no tag). A
#: block is a 26 mm cube with the tag on every face (cube mapping), so its
#: black edge is 8/10 of the cube: the "20 mm tag" whose decode range is a
#: measurement to make against the first attempt, not before it.
BLOCK_TAG_IDS = (20, 21, 22)
BLOCK_TAG_SIZE = 0.026 * 8 / 10
#: The bench's two masses (issue #215; challenge/bench.py): the same 26 mm
#: cube as a block, so the claw's one proven grasp finds them, with the
#: next two ids. Which one is which is a fact the offer states (#227): the
#: known mass is told, the unknown is what the job is.
MASS_TAG_IDS = (23, 24)
#: The quadruped's dock (issue #378; legs/dock.py): a board of four, two
#: rows of two, so a robot standing over the dock and one lying on it each
#: see a pair, and a pair's BASELINE gives the facing (issue #88).
DOCK_TAG_IDS = (25, 26, 27, 28)
DOCK_TAG_SIZE = 0.060
#: The quadruped's rack (issue #378; legs/rack.py): a pair a bay, three
#: bays, either side of each bay under its peg's ends, so the nose camera
#: at the working pose sees the bay's pair and fits the rack's facing to
#: their baseline.
LEGS_RACK_TAG_IDS = (29, 30, 31, 32, 33, 34)
LEGS_RACK_TAG_SIZE = 0.060
#: ...and its built-tool rail beside it on the same board (issue #407; the
#: workshop's tools hang there): the same pair a bay, its own ids.
LEGS_BUILT_TAG_IDS = (47, 48, 49, 50, 51, 52)
#: The lab's three pressure plates (issue #419): one tag a plate, on a sign
#: at its far edge facing the room (`activity/cage.py`), what a robot finds
#: the plate by. 120 mm, read from across the lab: MEASURED off the nose
#: camera, square-on past 5 m and 70 deg off its face to 3.5 m.
PLATE_TAG_IDS = (35, 36, 37)
PLATE_TAG_SIZE = 0.120
#: The two whiteboards (issue #406): a pair a board, on the wall either side
#: of it level with its middle (`tools/drawing.py`), what a robot finds the
#: board by and fits its facing to (the baseline, #88's rule). The plates'
#: 120 mm, read from across a room.
BOARD_TAG_IDS = {"whiteboard_a": (38, 39), "whiteboard_b": (40, 41)}
BOARD_TAG_SIZE = 0.120
#: The areas the claw's cubes are set out in (issue #407): a pair of tags
#: either side of each, the boards' size, on the workshop corner's wall
#: behind the tower's blocks and on the bench's front behind its masses --
#: what a robot finds the area by and fits its facing to; and one on the
#: garden's east fence, what the census finds the garden by.
TOWER_TAG_IDS = (42, 43)
BENCH_TAG_IDS = (44, 45)
GARDEN_TAG_IDS = (46,)
AREA_TAG_SIZE = BOARD_TAG_SIZE
#: The drop-handle chest's knob (issue #466; activity/chest.py), a demo's and
#: never the served world's: the claw's 26 mm cube with its tag on every
#: face, as the blocks' are, so the claw's walk-in can steer by it.
CHEST_TAG_IDS = (53,)

# Physical marker sizes (m), edge of the BLACK tag -- what the detector is
# told, and what PnP scales its translation by. The plate carrying it is
# larger by the quiet zone.
SMALL_TAG_SIZE = 0.030

# Which physical size each id is, so one detection pass can serve markers of
# different sizes: PnP translation scales linearly with the assumed tag
# size, so a single decode can be rescaled per id exactly.
TAG_SIZES = {**{i: SMALL_TAG_SIZE for i in MODULE_TAG_IDS.values()},
             **{i: BLOCK_TAG_SIZE for i in BLOCK_TAG_IDS},
             **{i: BLOCK_TAG_SIZE for i in MASS_TAG_IDS},
             **{i: BLOCK_TAG_SIZE for i in CHEST_TAG_IDS},
             **{i: DOCK_TAG_SIZE for i in DOCK_TAG_IDS},
             **{i: LEGS_RACK_TAG_SIZE for i in (*LEGS_RACK_TAG_IDS, *LEGS_BUILT_TAG_IDS)},
             **{i: PLATE_TAG_SIZE for i in PLATE_TAG_IDS},
             **{i: BOARD_TAG_SIZE for ids in BOARD_TAG_IDS.values() for i in ids},
             **{i: AREA_TAG_SIZE for i in (*TOWER_TAG_IDS, *BENCH_TAG_IDS, *GARDEN_TAG_IDS)}}

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


def write_tag_pngs(directory: Path = TAG_DIR, ids=None) -> list[int]:
  """Write the house's tag PNGs -- the tower's blocks and the bench's
  masses -- or just `ids`. Returns the ids: the home world declares
  exactly these as its textures (a scene test holds every declared texture
  referenced), and what is put in at load (the dock, the rack, the plates'
  signs, a built module) asks for its own by name.

  ⚠ A PNG ALREADY HOLDING THESE BYTES IS LEFT ALONE, AND ANY OTHER IS
  REPLACED WHOLE (issue #440): the suite runs the house's generator while
  other workers compile the house, and `Image.save` onto the file truncates
  it first -- they read an empty texture in 2 of 3 `-n 6` runs. Written
  beside it under a name no other writer shares (the workers all write the
  same files), then renamed over it."""
  from PIL import Image
  directory.mkdir(parents=True, exist_ok=True)
  ids = list(ids) if ids is not None else [*BLOCK_TAG_IDS, *MASS_TAG_IDS]
  for tag_id in ids:
    png = io.BytesIO()
    Image.fromarray(tag_image(tag_id)).save(png, format="PNG")
    path = directory / f"tag{tag_id}.png"
    if path.exists() and path.read_bytes() == png.getvalue():
      continue
    # Not `mkstemp`: its 0600 is renamed into models/, which the image copies
    # as root and reads as `pluggy` -- a texture the world cannot open.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(png.getvalue())
    os.replace(tmp, path)
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
  quick_decode_uninit), and a mission held TWO detectors (the rover's dock
  camera and its rack finder). When the second was collected the family
  was freed twice and the process took SIGSEGV, killing whole test runs at
  random.

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
    # ⚠ NO SHADOWS, NO REFLECTIONS (rooftop-media-2026 #296). MEASURED on
    # the deploy box (MUJOCO_GL=osmesa, llvmpipe, home world, 2026-09-19):
    # this frame costs 1113 ms with shadows and 32 ms without -- sixteen
    # lights each casting into a 4096^2 shadow map is sixteen 16-Mpixel
    # depth passes per look, on a software rasteriser, ~97 % of the render.
    # At 3-4 looks a sim-second per robot the served pair spent ~80 % of
    # all its CPU here and ran at 0.23x real time. A tag decode thresholds
    # gray levels and a real camera sees no shadow PASS; nothing hardware
    # has is lost. The viewer and the filmstrips keep their shadows: this
    # is the detector's scene alone, and `update_scene` keeps the flags.
    for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
      self.renderer.scene.flags[flag] = 0
    self.camera_name = camera_name
    self.width, self.height = width, height
    self.tag_size = tag_size
    fovy = float(model.camera(camera_name).fovy[0])
    f = (height / 2) / np.tan(np.radians(fovy) / 2)
    self.camera_params = (f, f, width / 2, height / 2)
    self.detector = _shared_detector()

  def detect(self, data) -> dict:
    """{tag_id: {"t": (x, y, z) in camera frame, "center": (u, v), "yaw": r,
    "normal": n}}; `n` is the tag's z axis in the camera frame, pointing
    INTO its face (away from a camera square-on), what "yaw" is read off.

    One render, one decode, all sizes: PnP translation is linear in the
    assumed tag size, so each id's pose is rescaled from the nominal size
    to its own. Cheaper than a pass per marker size, and exact.

    "yaw" is the tag PLANE's rotation about the camera's vertical, in
    radians: 0 when the tag faces the camera squarely, positive when the
    robot's heading is rotated positive (counter-clockwise) of the tag's
    normal. It comes from the same PnP pose as "t" (rotation is scale-free,
    so no per-size rescale). Measured convention check: a robot placed
    +10 deg off the charge approach heading reads +9.4 deg (issue #32).

    ⚠ Square-on it is a COIN FLIP (issue #88): a planar tag viewed along
    its normal has two mirrored PnP solutions, and the solver's pick swung
    a bay tag's yaw -7.5..+7 deg across 2 mm of robot pose while "t"
    held to a millimetre (the detector prints "more than one new minima"
    when it happens). Fit a facing to several tags' translations instead
    (`legs.dock.fit_dock`, `legs.rack.fit_rack`); read one tag's yaw only
    when it is the only one there is.
    """
    return {d["id"]: {k: v for k, v in d.items() if k != "id"}
            for d in self.detect_all(data)}

  def detect_all(self, data) -> list[dict]:
    """Every decode in one render, as `detect`'s entries with their `id`.
    ⚠ A CUBE SHOWS ITS TAG ON EVERY FACE (`challenge.stack.block_xml`), and
    keyed by id the last face decoded wins -- lying in front of one, that
    was its top seen edge-on, 16 mm over the face the robot faced (#407).
    Which face a decode is, is its `normal`."""
    self.renderer.update_scene(data, camera=self.camera_name)
    rgb = self.renderer.render()
    gray = np.ascontiguousarray(
      (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
       + 0.114 * rgb[:, :, 2]).astype(np.uint8))
    found = self.detector.detect(
      gray, estimate_tag_pose=True, camera_params=self.camera_params,
      tag_size=self.tag_size)
    out = []
    for det in found:
      tag_id = int(det.tag_id)
      scale = TAG_SIZES.get(tag_id, self.tag_size) / self.tag_size
      normal = np.asarray(det.pose_R) @ (0.0, 0.0, 1.0)
      out.append({
        "id": tag_id,
        "t": tuple(float(v) * scale for v in np.asarray(det.pose_t).ravel()),
        "center": (float(det.center[0]), float(det.center[1])),
        "yaw": float(np.arctan2(normal[0], normal[2])),
        "normal": tuple(float(v) for v in normal),
      })
    return out

  def close(self) -> None:
    self.renderer.close()
