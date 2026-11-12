"""What a ray cast leaves out: what it cannot reach (issue #560), and nothing
it can (issue #657).

Every geom in the model costs every ray of an `mj_multiRay`, whether a ray
could reach it or not -- 27 us a LIDAR frame each in the house -- and its
cutoff leaves them in (`perception.lidar`'s cast found one saved nothing).
The grown world's 115 more geoms, nearly all of them past any one sensor's
reach, cost the served pair a sixth of its speed. A geom whose every point
lies past what a sensor can read could only come back as what that sensor
already counts as nothing, so a cast without it is the same reading, bit for
bit -- each sensor states why at its own call.

`static_inertials` is the other half: what keeps MuJoCo's own culling of a
whole body from leaving out a body a ray does meet (SimNotes, "A ray through
a body MuJoCo's cull left out").
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
#: How far off its body a cull sphere may stand and be left alone, m: rounding.
SPHERE_OFF_M = 1e-6
#: A box's corners, each half-size's sign.
_CORNERS = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])


def static_inertials(model) -> dict[str, tuple[np.ndarray, float, np.ndarray]]:
  """The bodies welded to the world whose cull sphere `mj_multiRay` would
  misplace, each with the inertial frame it needs: `{name: (pos, mass,
  inertia)}`, the frame unturned in the body's, `pos` the middle of its box,
  its mass and inertia as compiled.

  ⚠ MuJoCo 3.10 leaves a body out of an `mj_multiRay` ray that misses its
  bounding sphere, and centres that sphere at `xipos` plus the box's offset
  UNTURNED (`mju_singleRay`; `mj_ray` tests the box itself). The compiler
  turns a body's inertial frame to its principal axes, and 6 % of the depth
  camera's rays on a dining table's top went on through it to the floor
  (#657). Welded to the world a body's inertia moves nothing and its frame
  may stand anywhere, so it stands where there is no offset to misplace.
  Unturned alone does not do: a body turned in the world turns the offset
  itself. A body that moves keeps its true inertia, and this bug."""
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  out = {}
  for b in range(1, model.nbody):
    adr = model.body_bvhadr[b]
    if adr < 0 or model.body_weldid[b] or model.body_mocapid[b] >= 0:
      continue
    offset = model.bvh_aabb[adr, :3]
    if np.linalg.norm(data.ximat[b].reshape(3, 3) @ offset - offset) < SPHERE_OFF_M:
      continue
    corners = []
    for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]):
      mat = np.empty(9)
      mujoco.mju_quat2Mat(mat, model.geom_quat[g])
      corners.append(model.geom_pos[g] + (model.geom_aabb[g, :3] + _CORNERS * model.geom_aabb[g, 3:])
                     @ mat.reshape(3, 3).T)
    points = np.concatenate(corners)
    out[model.body(b).name] = ((points.min(axis=0) + points.max(axis=0)) / 2,
                               float(model.body_mass[b]), model.body_inertia[b].copy())
  return out


def fix_static_inertials(spec) -> None:
  """`static_inertials` set on a world's `spec`, for what a world puts in at
  load: the chest, the drawers, the stations, the signs. (The house's own
  are written into its file, `home.world`.)"""
  for name, (pos, mass, inertia) in static_inertials(spec.compile()).items():
    body = spec.body(name)
    body.explicitinertial = True
    body.ipos, body.iquat, body.mass, body.inertia = pos, [1.0, 0.0, 0.0, 0.0], mass, inertia


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
