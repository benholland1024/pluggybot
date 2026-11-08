"""Near-field depth camera, modelled by batched ray casts (issue #34).

The robot's third ranging sensor, after the LIDAR (one horizontal plane, on
the quadruped 0.51 m up) and the navigation camera (AprilTags): a RealSense
D435-class active stereo unit on the body, pitched at the floor ahead, so
the robot can find things ON and near the floor -- where the scan plane
never looks. The decision and the runner-up sensors are docs/Parts.md
"near-field depth camera"; the height map it feeds is
`perception/heightmap.py`, and the humanoid's planner reads its low points
(`HumanoidMission._planning_grid`).

A DEPTH IMAGE, CAST AS RAYS. Each pixel of the D435's pinhole frustum is one
`mj_multiRay` ray from the `depth_eye` camera element, so the frame is
byte-deterministic (no GPU, no MSAA -- the lesson of issue #110), carries a
geom id per pixel (which is how the robot's own body is filtered), and costs
~0.6 µs a ray: 120 x 70 is ~5 ms, the LIDAR's 360-ray scan is ~3 ms.

Honest where the part is, and the tests pin each:

  AXIAL DEPTH, NOT RANGE. A stereo unit reports z along the optical axis,
  its limits and its noise are in z, and the point is recovered from z and
  the pixel's fixed direction -- exactly what a consumer of the real unit
  does. `MIN_Z` is the datasheet's min-Z at full resolution.

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
  A legged body (`mount="body"`, #388) gets its points in the root body's
  own frame: its torso pitches 33° on a flight, so no level can be
  assumed, and the caller applies its attitude estimate (`humanoid/scan.py`).
  ⚠ A JOINT ON THE MOUNT'S WAY IS TURNED AS ITS DRIVER READS IT (#560's
  stage (c3): the humanoid's head rides its waist): the frame takes the
  reading (`frame(joints=)`) and refuses to be laid without it
  (`perception.mount`).
"""

from dataclasses import dataclass, field
import functools
import math

import mujoco
import numpy as np

from pluggybot.power import DEPTH_CAMERA_W
from pluggybot.perception.cull import StaticCull
from pluggybot.perception.lidar import robot_geoms
from pluggybot.perception.mount import camera_mount
from pluggybot.robot import FIRST, RobotHandle

CAMERA = "depth_eye"

#: Sim resolution. The D435's depth stream is 848 x 480 at 87° x 58°; this is
#: its aspect at 1/7 (120/70 = 1.714 = tan 43.5° / tan 29°), 8400 rays, ~5 ms
#: a frame (measured, #34). A 60 x 35
#: frame is ~1.3 ms and 2.4 cm a pixel at 1 m; the height map's 2 cm cell is
#: what sets the floor at 120 wide.
WIDTH, HEIGHT = 120, 70
PERIOD = 0.1              # s between frames: 10 Hz. The part streams 30 Hz; the
                          # map integrates over motion and 10 Hz is the LIDAR's
                          # rate, so one seam ticks both.

MIN_Z = 0.28              # m, the datasheet's min-Z at 1280 x 720 (lower at lower
                          # resolutions; the conservative number, like the LIDAR's
                          # 8 m).
MAX_Z = 3.0               # m: the part ranges to 10 m, but σ is 32 mm here and the
                          # map is near-field. Also the ray cutoff, so the cost.
#: A cast that culls nothing (`reach`): MuJoCo's ray cast skips a geom whose
#: bounding sphere lies past its cutoff, and a PLANE's sphere is its anchor
#: point, so cast to `MAX_Z` a camera 2.75 m from where the floor plane is
#: anchored sees no floor (the bare floor's: a seeing walker walked on a
#: frozen map from there, #507). Unculled, a frame cost the same.
UNCULLED = mujoco.mjMAXVAL
BASELINE = 0.050          # m, the D435's imager baseline; sets the shadow width
                          # and, with the sub-pixel error, the noise
NOISE_K = 0.0036          # 1/m: σ_z = NOISE_K · z². (0.08 px sub-pixel error on
                          # the real 447 px focal length and 50 mm baseline.)
DROPOUT = 0.01            # returns lost to dark or specular surfaces; the active
                          # projector makes this small, not zero

#: One empty cloud, shared: a frame with no peer in it allocates nothing.
_NO_POINTS = np.zeros((0, 3))
_NO_GEOMS = np.zeros(0, dtype=np.int32)


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
  what they hit. `peer_geoms` is the geom each of those rows hit, so a
  consumer can say WHICH robot it saw (issue #365: one lying on the floor
  is not held for, `HubLifecycle._near_field_step`)."""
  z: np.ndarray
  points: np.ndarray
  self_fraction: float
  peers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
  peer_geoms: np.ndarray = field(
    default_factory=lambda: np.zeros(0, dtype=np.int32))


@dataclass
class DepthPicture:
  """One frame of the depth camera laid in the believed world frame, the
  picture beside its points (`Body.depth_pictures`, issue #636): a cloud
  says where points are, and nothing of where the look at them stopped.

  `z` is its `DepthFrame`'s; `points` the (k, 3) cloud in the believed world
  frame, one row per valid pixel of `z` in reading order (`pixels`); `eye`
  where the camera is and `turn` its frame (camera to world, its columns the
  camera's axes); `focal` its focal length, px; `near` and `reach` the
  nearest and farthest it reads, m (its range gate), and `noise_k` its
  noise, sigma_z = noise_k * z^2, along each ray."""
  z: np.ndarray
  points: np.ndarray
  eye: np.ndarray
  turn: np.ndarray
  focal: float
  near: float
  reach: float
  noise_k: float

  def pixels(self) -> tuple[np.ndarray, np.ndarray]:
    """`points`' pixels: their rows and their columns."""
    return np.nonzero(~np.isnan(self.z))

  @functools.cached_property
  def world(self) -> np.ndarray:
    """Each pixel's point, H x W x 3, the believed world frame: NaN where it
    read nothing (built once a picture: `measure` walks twelve times)."""
    out = np.full((*self.z.shape, 3), np.nan)
    out[~np.isnan(self.z)] = self.points
    return out

  def rises(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """What a metre of axial depth does to the height along each of these
    pixels' rays, m: the part's noise is along the ray, so this is its share
    of a pixel's height."""
    height, width = self.z.shape
    u, v = (cols - width / 2 + 0.5) / self.focal, (rows - height / 2 + 0.5) / self.focal
    return self.turn[2, 0] * u - self.turn[2, 1] * v - self.turn[2, 2]

  def project(self, points: np.ndarray) -> np.ndarray:
    """Where `points` (k, 3, the believed world frame) fall in the picture:
    (k, 2), row and column, in pixels (a pixel's centre on the integer),
    as the camera's pinhole lays them (`DepthCamera.__init__`)."""
    cam = (np.asarray(points, dtype=float) - self.eye) @ self.turn
    depth = -cam[:, 2]
    height, width = self.z.shape
    return np.column_stack([-self.focal * cam[:, 1] / depth + height / 2 - 0.5,
                            self.focal * cam[:, 0] / depth + width / 2 - 0.5])


class DepthCamera:
  """D435-class depth camera over MuJoCo ray casts, from a camera element."""

  #: What it draws while it streams, W (the loop adds it to the pack's load
  #: whenever the camera is on).
  draw_w = DEPTH_CAMERA_W

  def __init__(self, model, handle: RobotHandle = FIRST,
               camera_name: str = CAMERA, width: int = WIDTH,
               height: int = HEIGHT, min_z: float = MIN_Z,
               max_z: float = MAX_Z, noise_k: float = NOISE_K,
               dropout: float = DROPOUT, baseline: float = BASELINE,
               seed: int = 0, mount: str = "body", reach: float | None = None) -> None:
    self.handle = handle
    self.camera_name = handle.el(camera_name)
    self.width, self.height = width, height
    self.min_z, self.max_z = min_z, max_z
    #: How far the rays are cast, m: `max_z` unless told (`UNCULLED`); the
    #: range gate is `max_z` either way.
    self.reach = max_z if reach is None else reach
    self.noise_k, self.dropout, self.baseline = noise_k, dropout, baseline
    self.rng = np.random.default_rng(seed)
    #: ...and the stream the PEER channel draws on (issue #328, the lesson
    #: of #316's `Lidar.peer_rng`): a peer pixel is as noisy as any other,
    #: and taking its noise off the map's stream would move every reading
    #: the height map takes the moment a second robot walked into frame.
    self.peer_rng = np.random.default_rng(seed + 1)
    self._other_roots: list[str] = []
    #: A tool riding this robot's fork (issue #405): its own for the cloud.
    self._carried: str | None = None
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
    # The NOMINAL mount: the camera's pose on the robot's root body, each
    # joint on its way as its driver reads it (`frame(joints=)`). Points
    # are reconstructed through this, never through the live pose.
    if mount != "body":
      raise ValueError(f"unknown mount {mount!r}")
    # A legged body (#388): the root body's own frame, the torso's pitch
    # left to the caller's attitude estimate -- on a flight it is 33°.
    self.mount = camera_mount(model, self.cam_id, handle.root)
    if not self.mount.joints:
      self.origin_robot, mat = self.mount.pose()
      self._dirs_robot = self._dirs_cam @ mat.T

  def rebind(self, model) -> None:
    """The camera and the geom sets, by name: built, and built again when
    what it carries or which robots it excludes changes."""
    self.model = model
    self.cam_id = model.camera(self.camera_name).id
    #: What a cast leaves out past what the gate can pass (`perception.cull`)
    self._cull = StaticCull(model)
    self._self_geoms = robot_geoms(model, self.handle.root)
    if self._carried is not None:
      self._self_geoms |= robot_geoms(model, self._carried)
    self._other_geoms: set = set()
    for root in self._other_roots:
      self._other_geoms |= robot_geoms(model, root)
    self._filtered = np.fromiter(self._self_geoms | self._other_geoms,
                                 dtype=np.int32)
    #: ...and the two halves of it apart, because the frame now answers
    #: them apart (issue #328). Both stay out of `points`.
    self._mine = np.fromiter(self._self_geoms, dtype=np.int32)
    self._theirs = np.fromiter(self._other_geoms, dtype=np.int32)

  def carry(self, body: str | None) -> None:
    """A body riding this robot's fork is its own to the cloud, or None."""
    self._carried = body
    self.rebind(self.model)

  def exclude_robot(self, root_name: str) -> None:
    """Keep another robot's body out of the MAP's cloud, as the LIDAR keeps
    it out of the scan -- and, since issue #328, answer it on `peers`
    instead of throwing it away. One name, both halves, because a caller
    that asked for the exclusion is exactly the caller that needs the
    channel: it has a second robot in its world."""
    self._other_roots.append(root_name)
    self.rebind(self.model)

  def _mounted(self, joints: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """The camera's place and each pixel's direction in the root body's
    frame, its mount's joints at `joints` (`perception.mount.Mount.pose`)."""
    if not self.mount.joints:
      return self.origin_robot, self._dirs_robot
    origin, mat = self.mount.pose(joints)
    return origin, self._dirs_cam @ mat.T

  def frame(self, data, joints: dict | None = None) -> DepthFrame:
    """One frame, its points laid through the mount -- each joint on its
    way at `joints[name]`, rad, as the robot's own driver reads it."""
    origin, dirs = self._mounted(joints)
    n = self.width * self.height
    pos = np.array(data.cam_xpos[self.cam_id], dtype=np.float64)
    mat = np.array(data.cam_xmat[self.cam_id], dtype=np.float64).reshape(3, 3)
    world_dirs = np.ascontiguousarray((self._dirs_cam @ mat.T).reshape(-1))
    # ...every static geom left out that lies past what the range gate can
    # pass at the image's widest angle (#560): its pixel would read beyond
    # `max_z`, NaN as a pixel that met nothing is; a farther pixel shadows
    # only farther ones (`_shadow`); and noise is drawn for every pixel --
    # the same frame, bit for bit (`test_a_cast_leaves_out_nothing_it_could_reach`)
    self._cull.multi_ray(data, pos, world_dirs, -1, self._geomid, self._dist, n, self.reach,
                         self.max_z / float(self._axial.min()))
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
    peers, peer_geoms = _NO_POINTS, _NO_GEOMS
    if self._theirs.size:
      theirs = hit & np.isin(self._geomid, self._theirs) & ~shadow
      if theirs.any():
        pz, peers = self._cloud(np.where(theirs, z, np.nan), self.peer_rng, origin, dirs)
        # `peers`' rows, in order; a mask index copies the reused ray buffer
        peer_geoms = self._geomid[~np.isnan(pz)]
    z, points = self._cloud(np.where(robot | shadow, np.nan, z), self.rng, origin, dirs)
    return DepthFrame(z=z.reshape(self.height, self.width), points=points,
                      self_fraction=float(robot.mean()), peers=peers,
                      peer_geoms=peer_geoms)

  def _cloud(self, z: np.ndarray, rng, origin: np.ndarray, dirs: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Range gate, noise, dropout and the pinhole reconstruction through
    the mount (`origin`, `dirs`, `_mounted`): `(z, points)`.

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
    return z, origin + dirs[valid] * depth[:, None]

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
