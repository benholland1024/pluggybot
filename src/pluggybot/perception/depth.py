"""Near-field depth camera, modelled by batched ray casts (issue #34).

The robot's third ranging sensor, after the LIDAR (one horizontal plane at
0.223 m) and the navigation camera (AprilTags): a RealSense D435-class active
stereo unit on the mast top, pitched 40° at the floor ahead, so the robot can
find things ON the floor -- the one place the scan plane never looks. The
decision, the runner-up sensors and the mount are docs/Parts.md "near-field
depth camera"; the height map it feeds is `perception/heightmap.py`.

A DEPTH IMAGE, CAST AS RAYS. Each pixel of the D435's pinhole frustum is one
`mj_multiRay` ray from the `depth_eye` camera element, so the frame is
byte-deterministic (no GPU, no MSAA -- the lesson of issue #110), carries a
geom id per pixel (which is how the robot's own body is filtered), and costs
~0.6 µs a ray: 120 x 70 is ~5 ms, the LIDAR's 360-ray scan is ~3 ms.

Honest where the part is, and the tests pin each:

  AXIAL DEPTH, NOT RANGE. A stereo unit reports z along the optical axis,
  its limits and its noise are in z, and the point is recovered from z and
  the pixel's fixed direction -- exactly what a consumer of the real unit
  does. `MIN_Z` is the datasheet's min-Z at full resolution; it does not
  bind on this mount (nothing in the frame but the robot is inside 0.5 m).

  NOISE GROWS WITH z². Stereo depth error is quadratic in distance (a fixed
  sub-pixel disparity error, `NOISE_K` = 1/(f·B) x 0.08 px on the real
  optics): 0.9 mm at 0.5 m, 3.6 mm at 1 m, 14 mm at 2 m, 32 mm at `MAX_Z`.

  A NEAR OBJECT SHADOWS THE BACKGROUND ON ITS IMAGE-LEFT. Depth is computed
  in the left imager's frame, and background the right imager cannot see
  (hidden behind a nearer surface) has no disparity. The band is on the
  image-left of every near edge, f·B·(1/z_near - 1/z_far) pixels wide, and
  it is the sensor's signature: the floor right beside a dropped tool reads
  UNKNOWN, never "floor". Computed as the right-image ordering test (a
  pixel is shadowed when some pixel to its right lands at or left of it in
  the right image), one reverse cumulative minimum per row.

  NO RETURN IS NOT A READING -- the LIDAR's rule inverted. A LIDAR ray that
  hits nothing reports max range and the grid clears free space along it; a
  stereo pixel with no disparity (out of range, shadowed, textureless) is
  INVALID and says nothing. `MAX_Z` is where a pixel becomes NaN, never a
  point at 3 m. The height map inherits this: a cell nobody has measured is
  unseen, never floor.

  THE ROBOT SEES ITSELF. The lower rows are the chassis deck, the LIDAR body
  and the carriage; those pixels are dropped (`self_fraction` measures them)
  and they still shadow what stands behind them.

  THE FRAME IS IN THE ROBOT'S FRAME. Points come out relative to the axle
  midpoint, z above the floor, through the mount's NOMINAL transform (the
  chassis assumed level, as a robot without a tilt estimate assumes) -- the
  true pitch is in the ray, the nominal pitch is in the reconstruction, and
  the difference is the honest error. Where the robot IS is the map's
  business, and the map takes the believed pose (`HeightMap.update`).
"""

from dataclasses import dataclass, field
import math

import mujoco
import numpy as np

from pluggybot import control
from pluggybot.perception.lidar import robot_geoms
from pluggybot.robot import FIRST, RobotHandle

CAMERA = "depth_eye"

#: Sim resolution. The D435's depth stream is 848 x 480 at 87° x 58°; this is
#: its aspect at 1/7 (120/70 = 1.714 = tan 43.5° / tan 29°), 8400 rays, ~5 ms
#: a frame (`scripts/nearfield_spike.py --cost` is the measurement). A 60 x 35
#: frame is ~1.3 ms and 2.4 cm a pixel at 1 m; the height map's 2 cm cell is
#: what sets the floor at 120 wide.
WIDTH, HEIGHT = 120, 70
PERIOD = 0.1              # s between frames: 10 Hz. The part streams 30 Hz; the
                          # map integrates over motion and 10 Hz is the LIDAR's
                          # rate, so one seam ticks both.

MIN_Z = 0.28              # m, the datasheet's min-Z at 1280 x 720 (lower at lower
                          # resolutions; the conservative number, like the LIDAR's
                          # 8 m). Does not bind on the mast-top mount.
MAX_Z = 3.0               # m: the part ranges to 10 m, but σ is 32 mm here and the
                          # map is near-field. Also the ray cutoff, so the cost.
BASELINE = 0.050          # m, the D435's imager baseline; sets the shadow width
                          # and, with the sub-pixel error, the noise
NOISE_K = 0.0036          # 1/m: σ_z = NOISE_K · z². (0.08 px sub-pixel error on
                          # the real 447 px focal length and 50 mm baseline.)
DROPOUT = 0.01            # returns lost to dark or specular surfaces; the active
                          # projector makes this small, not zero
FLOOR_TOL = 0.02          # m: a point under this is floor, for the mount metric

#: One empty cloud, shared: a frame with no peer in it allocates nothing.
_NO_POINTS = np.zeros((0, 3))


@dataclass
class DepthFrame:
  """One frame: `z` is HEIGHT x WIDTH axial depth in metres, NaN where the
  pixel is invalid (out of range, shadowed, dropped, or a robot -- this one
  or another); `points` is the (k, 3) cloud in the robot frame -- x ahead of
  the axle midpoint, y to its left, z above the floor -- one row per valid
  pixel; `self_fraction` is the share of the frame the robot's own body
  filled.

  `peers` is the SAME cloud for the pixels that landed on ANOTHER ROBOT
  (issue #328), empty where there is none. It is kept apart for the reason
  the LIDAR keeps its peer returns apart (`Lidar.scan_split`, issue #316):
  the height map must not contain a robot that will have driven off by the
  time anything reads the map, and the thing that must not drive into it
  must. Nothing is inferred here -- these are the same casts, sorted by
  what they hit."""
  z: np.ndarray
  points: np.ndarray
  self_fraction: float
  peers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


class DepthCamera:
  """D435-class depth camera over MuJoCo ray casts, from a camera element."""

  def __init__(self, model, handle: RobotHandle = FIRST,
               camera_name: str = CAMERA, width: int = WIDTH,
               height: int = HEIGHT, min_z: float = MIN_Z,
               max_z: float = MAX_Z, noise_k: float = NOISE_K,
               dropout: float = DROPOUT, baseline: float = BASELINE,
               seed: int = 0) -> None:
    self.handle = handle
    self.camera_name = handle.el(camera_name)
    self.width, self.height = width, height
    self.min_z, self.max_z = min_z, max_z
    self.noise_k, self.dropout, self.baseline = noise_k, dropout, baseline
    self.rng = np.random.default_rng(seed)
    #: ...and the stream the PEER channel draws on (issue #328, the lesson
    #: of #316's `Lidar.peer_rng`): a peer pixel is as noisy as any other,
    #: and taking its noise off the map's stream would move every reading
    #: the height map takes the moment a second robot walked into frame.
    self.peer_rng = np.random.default_rng(seed + 1)
    self._other_roots: list[str] = []
    n = width * height
    self._geomid = np.zeros(n, dtype=np.int32)
    self._dist = np.zeros(n, dtype=np.float64)
    self.rebind(model)
    # The pinhole grid in the CAMERA frame (MuJoCo: x right, y up, looks
    # along -z), pixel centres, square pixels: fovy sets f, the aspect sets
    # the horizontal field.
    fovy = math.radians(float(model.cam_fovy[self.cam_id]))
    self.focal_px = (height / 2) / math.tan(fovy / 2)
    u = np.arange(width) - width / 2 + 0.5
    v = np.arange(height) - height / 2 + 0.5
    uu, vv = np.meshgrid(u, v)
    dirs = np.stack([uu / self.focal_px, -vv / self.focal_px,
                     -np.ones_like(uu)], axis=-1).reshape(n, 3)
    self._axial = 1.0 / np.linalg.norm(dirs, axis=1)   # cos(angle to axis)
    self._dirs_cam = dirs * self._axial[:, None]
    self._u = uu.reshape(n)
    # The NOMINAL mount: the camera's rest pose on the robot's root body,
    # re-expressed relative to the axle midpoint on the floor. Points are
    # reconstructed through this, never through the live pose.
    root = model.body(handle.root).id
    if int(model.cam_bodyid[self.cam_id]) != root:
      raise ValueError(f"{self.camera_name} must sit on {handle.root} itself")
    axle = np.mean([model.body(handle.el(w)).pos
                    for w in ("left_wheel", "right_wheel")], axis=0)
    self.origin_robot = (np.array(model.cam_pos[self.cam_id]) - axle
                         + np.array([0.0, 0.0, control.WHEEL_RADIUS]))
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.array(model.cam_quat[self.cam_id],
                                      dtype=np.float64))
    self._dirs_robot = self._dirs_cam @ mat.reshape(3, 3).T

  def rebind(self, model) -> None:
    """A recompiled world: the camera and the geom sets by name again
    (issue #168 slice C). Excluded OTHER robots are re-resolved too."""
    self.model = model
    self.cam_id = model.camera(self.camera_name).id
    self._self_geoms = robot_geoms(model, self.handle.root)
    self._other_geoms: set = set()
    for root in self._other_roots:
      self._other_geoms |= robot_geoms(model, root)
    self._filtered = np.fromiter(self._self_geoms | self._other_geoms,
                                 dtype=np.int32)
    #: ...and the two halves of it apart, because the frame now answers
    #: them apart (issue #328). Both stay out of `points`.
    self._mine = np.fromiter(self._self_geoms, dtype=np.int32)
    self._theirs = np.fromiter(self._other_geoms, dtype=np.int32)

  def exclude_robot(self, root_name: str) -> None:
    """Keep another robot's body out of the MAP's cloud, as the LIDAR keeps
    it out of the scan -- and, since issue #328, answer it on `peers`
    instead of throwing it away. One name, both halves, because a caller
    that asked for the exclusion is exactly the caller that needs the
    channel: it has a second robot in its world."""
    self._other_roots.append(root_name)
    self.rebind(self.model)

  def frame(self, data) -> DepthFrame:
    n = self.width * self.height
    pos = np.array(data.cam_xpos[self.cam_id], dtype=np.float64)
    mat = np.array(data.cam_xmat[self.cam_id], dtype=np.float64).reshape(3, 3)
    world_dirs = np.ascontiguousarray((self._dirs_cam @ mat.T).reshape(-1))
    mujoco.mj_multiRay(self.model, data, pos, world_dirs, None, 1, -1,
                       self._geomid, self._dist, None, n, self.max_z)
    hit = self._dist >= 0.0
    z = np.where(hit, self._dist * self._axial, np.nan)
    robot = hit & np.isin(self._geomid, self._filtered)
    # The shadow is cast on the true geometry, with the robot's own body
    # still standing in the scene as an occluder.
    shadow = self._shadow(z)
    # ANOTHER ROBOT'S PIXELS, BEFORE THEY ARE THROWN AWAY (issue #328).
    # They leave on their own channel and the map never sees them, which is
    # the same trade `Lidar.scan_split` makes one sensor along: the map must
    # not hold a body that moves, and the reflex must see it. Skipped
    # entirely when no peer is in frame, so a single robot pays nothing.
    peers = _NO_POINTS
    if self._theirs.size:
      theirs = hit & np.isin(self._geomid, self._theirs) & ~shadow
      if theirs.any():
        peers = self._cloud(np.where(theirs, z, np.nan),
                             self.peer_rng)[1]
    z, points = self._cloud(np.where(robot | shadow, np.nan, z), self.rng)
    return DepthFrame(z=z.reshape(self.height, self.width), points=points,
                      self_fraction=float(robot.mean()), peers=peers)

  def _cloud(self, z: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Range gate, noise, dropout and the pinhole reconstruction: `(z,
    points)`.

    ⚠ ONE SENSOR MODEL, TWO CHANNELS. The room and the peers go through
    this same function on their own streams, because a peer return that
    was gated or noised differently would be a different instrument
    reporting on the same casts -- and the difference would be invisible
    until something decided off it.
    """
    z = np.where((z < self.min_z) | (z > self.max_z), np.nan, z)
    z = z + rng.normal(0.0, 1.0, z.size) * self.noise_k * np.square(
      np.nan_to_num(z))
    z = np.where(rng.random(z.size) < self.dropout, np.nan, z)
    valid = ~np.isnan(z)
    depth = z[valid] / self._axial[valid]
    return z, self.origin_robot + self._dirs_robot[valid] * depth[:, None]

  def _shadow(self, z: np.ndarray) -> np.ndarray:
    """Pixels the RIGHT imager cannot see (the module docstring). A NaN
    pixel occludes nothing and is not itself marked."""
    nan = np.isnan(z)
    d = np.where(nan, 0.0, self.focal_px * self.baseline / np.where(nan, 1.0, z))
    u_right = np.where(nan, np.inf, self._u - d).reshape(self.height, self.width)
    # the minimum over every pixel to the RIGHT of this one, exclusive
    rev = np.minimum.accumulate(u_right[:, ::-1], axis=1)[:, ::-1]
    right_min = np.full_like(u_right, np.inf)
    right_min[:, :-1] = rev[:, 1:]
    return ((right_min <= u_right) & ~nan.reshape(u_right.shape)).reshape(-1)

  def floor_band(self, data) -> tuple[float, float]:
    """The floor the CENTRE column sees, as (nearest, farthest) x ahead of
    the axle -- the mount metric (`scripts/nearfield_spike.py --mount`)."""
    f = self.frame(data)
    valid = ~np.isnan(f.z.reshape(-1))
    centre = (np.abs(self._u) < 1.0)[valid]
    floor = centre & (f.points[:, 2] < FLOOR_TOL)
    if not floor.any():
      return math.nan, math.nan
    return float(f.points[floor, 0].min()), float(f.points[floor, 0].max())
