"""Finding the rack by looking at it, instead of being told where it is.

The rack carries a large fiducial plate high on its mast. This module turns
sightings of that plate into a believed RACK POSE, which every bay standoff
and terminal approach is then computed from -- so the hardcoded room
placement becomes a FALLBACK (the pose the robot boots believing, e.g.
because it started docked) rather than the source of truth.

Two jobs, deliberately kept apart, because they have different accuracy
budgets and different failure modes:

  discovery     where is the rack, roughly? Needs ~10 cm: enough to drive to
                the neighborhood. Done here, from across the room, by
                decoding the rack's AprilTag through the head camera --
                otherwise exactly like outlet landmarks in milestone 5 (same
                projection, same LandmarkStore, same sighting-count
                confirmation, because a marker seen once is still a rumor).
  fine approach the last few millimeters. Needs ~4 mm, which no stored pose
                can promise. Done by hub/mission.py's servo, looking at the
                bay's own tag as it closes in.

Measured detection range for the rack's 120 mm marker at 1280x720: it
decodes past 4.5 m, with PnP range good to a few mm at working distances.
The bay and module markers (30 mm) decode to ~2.5 m, which is far more than
they are ever read from.
"""

import math
from dataclasses import dataclass

from pluggybot.hub.coupling import (
  RACK_BRACKET_X, RACK_HANG_X, RACK_RAIL_Z, RACK_ROOM_POS, RACK_ROOM_YAW,
)
from pluggybot.hub.tags import RACK_TAG_ID, RACK_TAG_SIZE, TagDetector
from pluggybot.mapping.landmarks import LandmarkStore, wall_normal_conf
from pluggybot.perception.outlet_spotter import pixel_to_world

# Where the rack tag sits in the rack's own frame (generator constants).
TAG_LOCAL_X = RACK_BRACKET_X + 0.014
TAG_Z = RACK_RAIL_Z + 0.075

# No height gate, no size gate, no hoping: the tag's decoded ID is the
# filter. (The stand-in needed "only things above 0.42 m count" to keep the
# room's pale plates out; that reasoning is simply gone.)
MAX_RANGE = 5.0
MIN_SIGHTINGS = 3        # the milestone-5 lesson: one sighting is a rumor
#: How well conditioned the free-space sum must be before its direction is
#: believed as the rack's FACING (landmarks.wall_normal_conf).
#:
#: ⚠ MEASURED, and it exists because more map can make this estimate WORSE.
#: The rack stands against a wall, so early in a mission the free cells round
#: it all lie on one side and the sum is unambiguous. Drive BEHIND it -- which
#: the trip to the charge bay does -- and it becomes the free-standing
#: partition `wall_normal` warns about: the two sides nearly cancel, and the
#: direction that survives is leftover noise. Measured on the home world, a
#: charge trip moved the believed yaw by ~20 deg, which at the 0.63 m
#: standoff radius throws the hand-off pose ~0.2 m and points the approach
#: heading into the rack's flank.
MIN_FACING_CONF = 0.35


@dataclass(frozen=True)
class RackPose:
  """Believed pose of the rack: origin in world, plus its outward normal."""

  x: float
  y: float
  yaw: float               # direction of the rack's local +x, in world

  @classmethod
  def prior(cls) -> "RackPose":
    """What the robot believes before it ever looks -- from having been
    docked here at boot, or from a teach-by-driving step on a real robot."""
    return cls(RACK_ROOM_POS[0], RACK_ROOM_POS[1], math.radians(RACK_ROOM_YAW))

  @classmethod
  def from_tag(cls, tag_x: float, tag_y: float, yaw: float) -> "RackPose":
    """Back out the rack origin from a tag sighting and the facing."""
    return cls(tag_x - TAG_LOCAL_X * math.cos(yaw),
               tag_y - TAG_LOCAL_X * math.sin(yaw), yaw)

  def to_world(self, lx: float, ly: float) -> tuple[float, float]:
    c, s = math.cos(self.yaw), math.sin(self.yaw)
    return self.x + lx * c - ly * s, self.y + lx * s + ly * c

  @property
  def heading(self) -> float:
    """The heading a robot faces the rack with (into its outward normal)."""
    return math.atan2(-math.sin(self.yaw), -math.cos(self.yaw))

  def bay_standoff(self, station_y: float, standoff: float,
                   plug_lateral: float) -> tuple[float, float, float]:
    """(axle_x, axle_y, heading) of the hand-off pose for one bay."""
    bx, by = self.to_world(RACK_HANG_X, station_y)
    hd = self.heading
    nx, ny = math.cos(self.yaw), math.sin(self.yaw)
    sx, sy = bx + nx * standoff, by + ny * standoff
    right = (math.sin(hd), -math.cos(hd))
    return sx - plug_lateral * right[0], sy - plug_lateral * right[1], hd

  def error_against(self, other: "RackPose") -> tuple[float, float]:
    """(position error in m, yaw error in rad) -- for reports and tests."""
    return (math.hypot(self.x - other.x, self.y - other.y),
            abs(math.atan2(math.sin(self.yaw - other.yaw),
                           math.cos(self.yaw - other.yaw))))


class RackSpotter:
  """Sees the rack's AprilTag through the head camera, in world coordinates.

  The tag's ID is what makes this trustworthy: the room is full of pale
  plates and none of them decode as tag 0, so there is nothing to filter by
  height or size or hope. Range comes from the marker's own PnP pose, so no
  depth buffer is consulted -- on hardware there is no depth camera here.
  """

  def __init__(self, model, camera_name: str = "left_eye") -> None:
    self.detector = TagDetector(model, camera_name, tag_size=RACK_TAG_SIZE)
    self.fovy_deg = float(model.camera(camera_name).fovy[0])

  def spot(self, data, pose: tuple[float, float, float],
           ) -> list[tuple[float, float, float, int]]:
    """One look: [(x, y, z, 1), ...] -- at most one entry, the rack tag."""
    found = self.detector.detect(data)
    if RACK_TAG_ID not in found:
      return []
    det = found[RACK_TAG_ID]
    u, v = det["center"]
    depth = det["t"][2]
    if depth > MAX_RANGE:
      return []
    x, y, z = pixel_to_world(u, v, depth, pose,
                             width=self.detector.width,
                             height=self.detector.height,
                             fovy_deg=self.fovy_deg)
    return [(x, y, z, 1)]

  def close(self) -> None:
    self.detector.close()


class RackFinder:
  """Accumulates tag sightings into a confirmed rack pose."""

  def __init__(self, model, camera_name: str = "left_eye") -> None:
    self.spotter = RackSpotter(model, camera_name)
    self.landmarks = LandmarkStore()
    #: the last facing that came off a well-conditioned free-space sum. The
    #: rack does not turn round, so a good answer stays good -- and keeping
    #: it is the whole fix for a later, worse look (see MIN_FACING_CONF).
    self.facing: float | None = None

  def look(self, data, pose: tuple[float, float, float]) -> int:
    """One look; returns how many tag sightings it added."""
    n = 0
    for x, y, z, _px in self.spotter.spot(data, pose):
      self.landmarks.add_sighting(x, y, z, seen_from=(pose[0], pose[1]))
      n += 1
    return n

  def estimate(self, grid) -> RackPose | None:
    """Best confirmed rack pose, or None while still unconfirmed.

    Facing comes off the occupancy grid (landmarks.wall_normal) exactly as
    it does for outlets: the rack stands against a wall, so the free-space
    sum points out of it. A real AprilTag gives this directly from its pose
    -- and better -- but the grid version needs no new perception.

    ⚠ AND A FACING IS KEPT ONCE IT IS KNOWN. The position keeps improving
    with every sighting, which is what more looks are for; the FACING does
    not, because its input is the shape of the mapped free space rather than
    the number of times the tag was seen. Once the robot has driven behind
    the rack -- which is what going to the charge bay does -- the sum has
    free cells on both sides, nearly cancels, and returns noise. A rack does
    not turn round, so the honest response to a badly conditioned look is to
    keep the answer from the well conditioned one.

    Found the hard way: the SECOND tool fetch of a mission failed, because
    the charge trip in between rotated the believed rack by ~20 deg and the
    bay standoff computed from it sat 0.2 m off the approach axis and a
    metre from the bay. Nothing else exercised the sequence -- the first
    fetch of a mission happens before the robot has been round the back.
    """
    best = None
    for lm in self.landmarks.landmarks:
      if lm.n_sightings < MIN_SIGHTINGS:
        continue
      if best is None or lm.n_sightings > best.n_sightings:
        best = lm
    if best is None:
      return None
    nx, ny, conf = wall_normal_conf(grid, best.x, best.y,
                                    fallback=(best.seen_from_x - best.x,
                                              best.seen_from_y - best.y))
    if conf >= MIN_FACING_CONF or self.facing is None:
      self.facing = math.atan2(ny, nx)
    return RackPose.from_tag(best.x, best.y, self.facing)

  def close(self) -> None:
    self.spotter.close()
