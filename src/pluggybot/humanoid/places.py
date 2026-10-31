"""Finding a place on the humanoid (issue #559, stage 3): the quadruped's
search (`legs.places.PlaceWalk`, #419), unchanged -- where it should be,
the address, then viewpoints round it -- with the humanoid's own LOOK: one
decode of its NAVIGATION camera, the head's Camera Module 3 pitched for the
tags on the walls (`humanoid.model`'s `nav_pitch_deg`), each place tag in
it placed through its model of itself as its odometry last posed it (its
legs' encoders and its IMU, every tick) and through its stance into its
map. A mixin: `HumanoidMission` is the rest of the body.

⚠ NO PRESS: a plate is pressed by the quadruped's walk in, and the
humanoid's places are only found (`find`). ⚠ But it keeps off every pad
it knows, as the quadruped does (`keep_out`, `HumanoidMission.
_planning_grid`): kept off none, a humanoid exploring the lab walked onto
its plates and pressed the shock plate (#559's stage 4).

MEASURED (`home.storeroom`'s row from the storeroom's door, 5.5 m off, and
from in front of the counter, 0.8 m): the 120 mm tags decode across the
room and from where a robot works at the counter.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from pluggybot.legs.places import PlaceWalk, plate_pads, world_places
from pluggybot.mapping.places import CLOSE_M, OBLIQUE_RAD

#: The camera a place is looked for by: the head's, pitched for the walls.
NAV_CAMERA = "nav_eye"


def place_tags(model) -> tuple[list[int], tuple]:
  """The place tags this world puts up, and their fixtures: the house's
  (`legs.places.world_places`) and the storeroom's counter's row (#559)."""
  from pluggybot.home import storeroom
  ids, fixtures = world_places(model)
  if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                       f"store_tag{storeroom.TAGS[storeroom.PROPS[0]]}") >= 0:
    ids, fixtures = [*ids, *storeroom.TAGS.values()], (*fixtures, storeroom.fixture())
  return ids, fixtures


class HumanoidPlaces(PlaceWalk):
  """The places' search on `HumanoidMission` (the module docstring)."""

  def _rebind_places(self, model) -> None:
    """The plates' pads, which it keeps off once it knows them, and no
    feet to press one with."""
    self._pads, self._foot_gids = plate_pads(model), []

  def _init_places(self, model) -> None:
    super()._init_places(model)
    self._nav_tags = None

  def _world_places(self, model) -> tuple[list[int], tuple]:
    return place_tags(model)

  def _place_detector(self):
    if self._nav_tags is None:
      from pluggybot.rack.tags import TagDetector
      self._nav_tags = TagDetector(self.model, self.handle.el(NAV_CAMERA))
    return self._nav_tags

  def close_places(self) -> None:
    if self._nav_tags is not None:
      self._nav_tags.close()
      self._nav_tags = None

  # ---- looking ------------------------------------------------------------------

  def _stance_frame(self) -> tuple[float, float, float]:
    """Its stance in the map, (x, y, heading): its pelvis where it believes
    it, less where its odometry's model of itself puts the pelvis over its
    feet (`HumanoidBody.stance`'s reading)."""
    x, y, yaw = self.pose
    sm = self.odo.sm
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, sm.d.qpos[3:7])
    r = rot.reshape(3, 3)
    syaw = yaw - math.atan2(r[1, 0], r[0, 0])
    c, s = math.cos(syaw), math.sin(syaw)
    px, py = float(sm.d.qpos[0]), float(sm.d.qpos[1])
    return x - (c * px - s * py), y - (s * px + c * py), syaw

  def see_places(self, dets: dict) -> list[int]:
    """Every place tag in one decode into the memory, in the map frame:
    the camera placed through its model of itself as its odometry last
    posed it (the stance frame), and the stance into the map -- where the
    tag is, where it was seen from and, off a look close and oblique enough
    (`places.OBLIQUE_RAD`), which way its face points."""
    ids = [i for i in dets if i in self.places.ids]
    if not ids:
      return []
    sm = self.odo.sm
    mujoco.mj_camlight(sm.m, sm.d)
    cam = sm.m.camera(NAV_CAMERA).id
    origin, rot = sm.d.cam_xpos[cam].copy(), sm.d.cam_xmat[cam].reshape(3, 3).copy()
    sx, sy, syaw = self._stance_frame()
    c, s = math.cos(syaw), math.sin(syaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    eye = rz @ origin + [sx, sy, 0.0]
    t = float(self.data.time)
    for i in ids:
      d = dets[i]
      tx, ty, tz = d["t"]
      # the detector's camera is x right, y down, z ahead; MuJoCo's looks
      # along its -z with y up
      p = rz @ (origin + rot @ np.array([tx, -ty, -tz])) + [sx, sy, 0.0]
      facing = None
      if abs(d["yaw"]) >= OBLIQUE_RAD and math.hypot(p[0] - eye[0], p[1] - eye[1]) <= CLOSE_M:
        nx, ny, nz = d["normal"]
        n = rz @ (rot @ np.array([nx, -ny, -nz]))           # into the face
        facing = math.atan2(-n[1], -n[0])                  # ...and out of it
      self.places.see(i, float(p[0]), float(p[1]), t,
                      math.atan2(eye[1] - p[1], eye[0] - p[0]), facing)
    return ids

  def look_for_places(self) -> list[int]:
    """One look now: the place tags it decoded."""
    self._place_look = (float(self.data.time), self.pose)
    return self.see_places(self._place_detector().detect(self.data))

  def _place_step(self) -> None:
    """The walking look (`PlaceWalk._place_step`'s rule) on its feet and
    level, never while the get-up policy has the body."""
    if not self.places.ids or self.legs == "getup" or not self.level():
      return
    t = float(self.data.time)
    last_t, last = self._place_look
    from pluggybot.legs.places import LOOK_EVERY_S, LOOK_MOVED_M, LOOK_TURNED_RAD
    if t - last_t < LOOK_EVERY_S:
      return
    x, y, th = self.pose
    if last is not None and math.hypot(x - last[0], y - last[1]) < LOOK_MOVED_M \
        and abs(math.remainder(th - last[2], 2 * math.pi)) < LOOK_TURNED_RAD:
      return
    self.look_for_places()

  # ---- what the quadruped's search asks of its body --------------------------------

  def stand_routine(self):
    """On its feet already: it rests standing."""
    return
    yield

  def _turn_by_routine(self, degrees: float):
    """A turn where it stands by `degrees`, the look-around's step."""
    yield from self.face_routine(math.remainder(self.pose[2] + math.radians(degrees),
                                                2 * math.pi))
