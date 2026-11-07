"""Leaving out of a ray cast what it cannot reach (issue #560).

Every geom in the model costs every ray of an `mj_multiRay`, whether a ray
could reach it or not -- 27 us a LIDAR frame each in the house -- and its
cutoff leaves them in (`perception.lidar`'s cast found one saved nothing).
The grown world's 115 more geoms, nearly all of them past any one sensor's
reach, cost the served pair a sixth of its speed. A geom whose every point
lies past what a sensor can read could only come back as what that sensor
already counts as nothing, so a cast without it is the same reading, bit for
bit -- each sensor states why at its own call.
"""

import mujoco
import numpy as np

#: A geom group no world puts anything in: a cast leaves out what is moved
#: there for the call, and casts every other group.
CULL_GROUP = 5
#: ...and where what is PUT AWAY waits (#560: the restaurant's dishware
#: between its jobs, `activity.dishware.park`): no cast reads it, as no
#: camera draws it (MuJoCo draws groups 0-2). Before it, only the chest's
#: hidden lump was here, which a ray never meets (its alpha is 0).
PARKED_GROUP = 3
CAST_GROUPS = np.array([group not in (CULL_GROUP, PARKED_GROUP) for group in range(6)],
                       dtype=np.uint8)
#: A cast with the cull off: everything but what is put away.
UNCULLED_GROUPS = np.array([group != PARKED_GROUP for group in range(6)], dtype=np.uint8)


class StaticCull:
  """The geoms welded to the world, but a plane or a height field (no far
  side), each judged afresh at every cast -- an activity moves and resizes
  its geoms, and a mocap body moves -- against where it is and how big as
  the cast is about to meet it. Off where some geom is in `CULL_GROUP`
  already."""

  def __init__(self, model) -> None:
    self.model = m = model
    self.static = np.flatnonzero(
      (m.body_weldid[m.geom_bodyid] == 0)
      & ~np.isin(m.geom_type, (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)))
    t = m.geom_type[self.static]
    self._sphere = t == mujoco.mjtGeom.mjGEOM_SPHERE
    self._capsule = t == mujoco.mjtGeom.mjGEOM_CAPSULE
    self._cylinder = t == mujoco.mjtGeom.mjGEOM_CYLINDER
    self._mesh = ~np.isin(t, (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                              mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE,
                              mujoco.mjtGeom.mjGEOM_CYLINDER))
    self.on = not np.any(m.geom_group == CULL_GROUP)

  def far(self, data, pos, reach: float) -> np.ndarray:
    """The static geoms every point of which lies past `reach` from `pos`:
    each one's box in its own frame (`geom_xpos`, `geom_xmat`, `geom_size`;
    a mesh's compiled box, as nothing resizes one)."""
    m, g = self.model, self.static
    size = m.geom_size[g]
    half, centre = size.copy(), np.zeros((len(g), 3))
    half[self._sphere] = size[self._sphere, :1]
    cap, cyl = self._capsule, self._cylinder
    half[cap] = np.column_stack((size[cap, 0], size[cap, 0], size[cap, 0] + size[cap, 1]))
    half[cyl] = np.column_stack((size[cyl, 0], size[cyl, 0], size[cyl, 1]))
    centre[self._mesh], half[self._mesh] = m.geom_aabb[g[self._mesh], :3], m.geom_aabb[g[self._mesh], 3:]
    xmat = data.geom_xmat[g].reshape(-1, 3, 3)
    local = np.einsum("nji,nj->ni", xmat, pos - data.geom_xpos[g]) - centre
    gap = np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1)
    return g[gap > reach]

  def multi_ray(self, data, pos, dirs, bodyexclude: int, geomid, dist, n: int,
                cutoff: float, reach: float) -> None:
    """`mj_multiRay` from `pos` with every static geom past `reach` left
    out, the model's groups as they were after."""
    far = self.far(data, pos, reach) if self.on else self.static[:0]
    groups = self.model.geom_group[far]
    self.model.geom_group[far] = CULL_GROUP
    try:
      mujoco.mj_multiRay(self.model, data, pos, dirs,
                         CAST_GROUPS if self.on else UNCULLED_GROUPS, 1,
                         bodyexclude, geomid, dist, None, n, cutoff)
    finally:
      self.model.geom_group[far] = groups
