"""#762: the self-filter round a held grip, sized by what a job carries
(`mission.HELD_DROP_M`, the LIDAR's; `HELD_DEPTH_M`, the depth camera's;
SimNotes, "The self-filter sized by what a job carries"). Each rule pinned
without a flight -- the bare body posed at the carry pose, its sensors cast at
it, a direct call (docs/Testing.md):

  1. every piece a job may carry (`CARRIED`: its drawing, and where and how
     its reference grips it), posed in the hand at the carry pose, lies
     inside the depth camera's radius whole and inside the LIDAR's wherever
     its rings can reach it -- each by the 3 cm the robot's model of itself
     may place its grip off the true one (`GRIP_OFF_M`) and two sigmas of
     that sensor's range noise at the piece;
  2. those grips are the references' own: a reference that moves its grip
     fails here before it carries a piece past the radius;
  3. a carried piece of #736's candle's size, gripped as that reference grips
     it, stops the walk in no scan of the LIDAR's at its radius, where at
     0.12 it did in a fifth of them;
  4. each sensor drops at its own radius: the band at the LIDAR's, the depth
     camera's frame at its own.

#736's merge adds its candle and its frame to `CARRIED`.
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.activity import cube as cb
from pluggybot.activity import dishware as dw
from pluggybot.body import body_for
from pluggybot.challenge import solutions as sol
from pluggybot.evaluation import counter as ct
from pluggybot.evaluation import hands_check as hc
from pluggybot.humanoid import mission as hmis
from pluggybot.humanoid import model as hm
from pluggybot.humanoid.arms import HandGoal
from pluggybot.perception import depth as pd
from pluggybot.perception import lidar as pl

#: How far the robot's model of itself may place its grip off the true one,
#: m: the placement pin's tolerance
#: (`test_carry.py::test_the_mission_places_a_held_grip_off_its_model_of_itself_and_drops_it_from_its_layer`;
#: in a carried walk it was 2-13 mm, #762's flights).
GRIP_OFF_M = 0.03

#: A tag's face over the face it is printed on, m (`dishware._film`).
_TAG = dw.FILM
#: The reaches' turns: fingers 45 deg down, jaws across (the cups'); fingers
#: 60 deg down, jaws along -y (the plate's and the fork's).
_CUP_TURN = ((0.707, 0.0, -0.707), (0.0, 1.0, 0.0))
_DOWN60 = ((0.5, 0.0, -0.866), (0.0, -1.0, 0.0))


def _piece(name):
  """A drawing as a body a test can pose (mocap, no joint)."""
  def add(spec, body_name):
    if name == "cube":
      b = spec.worldbody.add_body(name=body_name, mocap=True)
      b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[cb.CUBE] * 3, pos=[0.0, 0.0, cb.CUBE])
      return
    dw.ATTACH[name](spec, (0.0, 0.0), 0.0, 0.0, 0.3, 0.7, tags=False, name=body_name)
    b = spec.body(body_name)
    for j in list(b.joints):
      spec.delete(j)
    b.mocap = True
  return add


#: EVERY PIECE A JOB MAY CARRY, as its reference grips it: (what, its drawing,
#: the hand, the grip's place in the piece's frame as it is set out -- level,
#: unturned in the stance, z up from what it stands on -- the reach's fingers
#: and jaws there, and each reference's words that put it there, read in its
#: source by `test_the_grips_are_the_references_own`: the kitchen shelf's are
#: the pass's sources rewritten, its grips as the pass's). The pieces are set
#: out as their references find them: the fork with its head ahead of the hand
#: (`counter.PASS_SOURCES`: "its head in").
CARRIED = (
  ("the pass's and the kitchen shelf's cup", "cup", "right", (0.0, 0.0, dw.CUP_H + _TAG - 0.015),
   *_CUP_TURN, ((sol.PASS, 'cz = read("spot.z") - 0.015'), (sol.SHELF, 'cz = read("spot.z") - 0.015'))),
  ("the pass's and the kitchen shelf's plate", "plate", "left",
   (0.0, sol.PASS_PLATE_ACROSS, dw.PLATE_FLOOR_T + _TAG + 0.0384), *_DOWN60,
   ((sol.PASS, 'gz = read("spot.z") + 0.0384'), (sol.SHELF, 'gz = read("spot.z") + 0.0384'))),
  ("the pass's and the kitchen shelf's fork", "fork", "left",
   (0.0, 0.0, 2 * dw.HANDLE[2] + _TAG + 0.0264), *_DOWN60,
   ((sol.PASS, 'gz = read("spot.z") + 0.0264'), (sol.SHELF, 'gz = read("spot.z") + 0.0264'))),
  ("the table check's cup", "cup", "right", (0.0, 0.0, dw.CUP_H + _TAG - 0.03), *_CUP_TURN,
   ((hc.TABLE, 'cz = read("spot.z") - 0.03'),)),
  ("the table check's plate", "plate", "left", (0.0, 0.066, 0.7734 - ct.TABLE_H), *_DOWN60,
   ((hc.TABLE, "gy = py + 0.066 * (1 - i)\n    gz = 0.7734 - 0.005 * i"),)),
  ("the table check's fork", "fork", "left", (0.0, 0.0, 0.7734 - 0.005 - ct.TABLE_H), *_DOWN60,
   ((hc.TABLE, "gz = 0.7734 - 0.005 * i"),)),
  ("the cube check's cube", "cube", "left", (0.01, 0.0, cb.CUBE + 0.003), *_DOWN60[:1],
   (0.0, 1.0, 0.0), ((hc.PLACE, "b = 0.01"),)),
)


def _turn(fingers, jaws) -> np.ndarray:
  """A reach's grip turn off its fingers and jaws, as `steps._grip_turn`
  makes it: x the jaws across the fingers, z against the fingers."""
  a = np.asarray(fingers, dtype=float) / np.linalg.norm(fingers)
  c = np.asarray(jaws, dtype=float)
  x = c - (a @ c) * a
  x /= np.linalg.norm(x)
  return np.column_stack([x, np.cross(-a, x), -a])


def _surface(model, data, body: int, step: float = 0.004) -> np.ndarray:
  """Points over every face of a body's geoms (boxes and cylinders), world."""
  out = []
  for g in np.flatnonzero(model.geom_bodyid == body):
    pos, mat, size = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
    local = []
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
      for ax in range(3):
        u, v = [k for k in range(3) if k != ax]
        for su in np.linspace(-1, 1, max(2, int(2 * size[u] / step) + 1)):
          for sv in np.linspace(-1, 1, max(2, int(2 * size[v] / step) + 1)):
            for sa in (-1.0, 1.0):
              p = np.zeros(3)
              p[ax], p[u], p[v] = sa * size[ax], su * size[u], sv * size[v]
              local.append(p)
    elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
      r, h = size[0], size[1]
      for t in np.linspace(0, 2 * math.pi, max(12, int(2 * math.pi * r / step)), endpoint=False):
        for z in np.linspace(-h, h, max(2, int(2 * h / step) + 1)):
          local.append((r * math.cos(t), r * math.sin(t), z))
        for rr in np.linspace(0.0, r, max(2, int(r / step) + 1)):
          local += [(rr * math.cos(t), rr * math.sin(t), -h), (rr * math.cos(t), rr * math.sin(t), h)]
    out.append(pos + np.asarray(local) @ mat.T)
  return np.concatenate(out)


@pytest.fixture(scope="module")
def carry():
  """The bare body with every drawing as a body a test poses, and a pose:
  `pose(side, grip, fingers, jaws, drawing)` puts the arm at its carry pose
  for a hand turned so (`Reach.carry_pose`), the rest of the body at the
  stand, and the drawing in the hand as gripped -- the others far off --
  returning the body, the true grip site and the drawing's body id."""
  spec = mujoco.MjSpec.from_string(hm.body_xml())
  names = ("plate", "cup", "fork", "cube")
  for n in names:
    _piece(n)(spec, f"held_{n}")
  cyl = spec.worldbody.add_body(name="held_candle", mocap=True)
  cyl.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.035, 0.05, 0.0], pos=[0.0, 0.0, 0.05])
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)

  def pose(side, grip, fingers, jaws, drawing):
    m = body.mission
    turn = _turn(fingers, jaws)
    m.reach.cmd[side] = HandGoal(np.zeros(3), turn, 1.0)
    q = m.reach.carry_pose(side)
    for j, v in zip(hm.ARM, q):
      data.qpos[model.jnt_qposadr[model.joint(f"{side}_{j}").id]] = v
    mujoco.mj_forward(model, data)
    site = model.site(f"{side}_grip").id
    at, rot = data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()
    for n in (*names, "candle"):
      mid = model.body_mocapid[model.body(f"held_{n}").id]
      data.mocap_pos[mid] = (20.0, 20.0, -5.0)
    b = model.body(f"held_{drawing}").id
    mid = model.body_mocapid[b]
    data.mocap_pos[mid] = at + rot @ (turn.T @ -np.asarray(grip, dtype=float))
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (rot @ turn.T).reshape(-1))
    data.mocap_quat[mid] = quat
    mujoco.mj_forward(model, data)
    # ...and the robot's model of itself posed as its senses would say it
    qs = np.array([data.qpos[model.jnt_qposadr[model.joint(n).id]] for n in hm.JOINT_NAMES])
    m.odo.sm.pose(qs, np.array([0.0, 0.0, -1.0]))
    return body, at, b

  yield pose
  body.close()


# ---- 1: every carried piece inside both radii ------------------------------------


@pytest.fixture(scope="module")
def needs(carry):
  """What each of `CARRIED` needs of each radius, m: (the depth camera's, the
  LIDAR's -- None where no ring reaches it at the carry pose), each the
  farthest point of it that sensor can see from the true grip, `GRIP_OFF_M`
  and two sigmas of the sensor's noise there."""
  out = {}
  for what, drawing, side, grip, fingers, jaws, _ in CARRIED:
    body, at, b = carry(side, grip, fingers, jaws, drawing)
    model, data = body.model, body.data
    pts = _surface(model, data, b)
    far = np.linalg.norm(pts - at, axis=1)
    # the depth camera looks down on all of it, its noise along its axis
    dep = body.mission.depth
    cam, cmat = data.cam_xpos[dep.cam_id], data.cam_xmat[dep.cam_id].reshape(3, 3)
    z = -((pts - cam) @ cmat)[:, 2]
    depth = float((far + GRIP_OFF_M + 2 * pd.NOISE_K * z ** 2).max())
    # the LIDAR's rings reach what stands at or over its steepest, its noise
    # at that range
    lid = body.mission.lidar
    spos, smat = data.site_xpos[lid.site_id], data.site_xmat[lid.site_id].reshape(3, 3)
    v = (pts - spos) @ smat
    seen = np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1])) >= lid.elevations.min() - 1e-9
    lidar = (float((far[seen] + GRIP_OFF_M + 2 * pl._sigma(np.linalg.norm(v[seen], axis=1))).max())
             if seen.any() else None)
    out[what] = (depth, lidar)
  return out


@pytest.mark.parametrize("what", [c[0] for c in CARRIED])
def test_every_carried_piece_lies_inside_the_self_filter_of_each_sensor(needs, what):
  """#762: at 0.12 the candle fired the front stop on a fifth of its scans and
  every carried walk crawled (123-273 s for 28-39), and the pass's fork's own
  returns stood in the band on 62 % of them, walls a walk then planned
  round. The depth camera looks down on the whole piece -- its noise a few
  mm (`depth.NOISE_K` z^2) -- and the LIDAR's rings reach what stands at or
  over its steepest, its noise centimetres (`lidar.SIGMA`)."""
  depth, lidar = needs[what]
  assert depth <= hmis.HELD_DEPTH_M, (
    f"{what} needs {depth:.3f} m of the depth camera's {hmis.HELD_DEPTH_M}")
  assert lidar is None or lidar <= hmis.HELD_DROP_M, (
    f"{what} needs {lidar:.3f} m of the LIDAR's {hmis.HELD_DROP_M}")


def test_each_radius_is_what_its_rule_needs_to_the_centimetre_and_no_wider(needs):
  """What a radius hides near a held hand -- a counter's edge, a board -- the
  robot does not see (#762's call 1): each is the most any carried piece
  needs of it, rounded up to the cm -- the LIDAR's the pass's fork's, its
  head above the steepest ring; the depth camera's the plate's, its rim."""
  def cm(v):
    return math.ceil(v * 100 - 1e-6) / 100
  assert hmis.HELD_DEPTH_M == cm(max(d for d, _ in needs.values()))
  assert hmis.HELD_DROP_M == cm(max(r for _, r in needs.values() if r is not None))


# ---- 2: the grips are the references' own ----------------------------------------


@pytest.mark.parametrize("what,refs", [(c[0], c[6]) for c in CARRIED], ids=[c[0] for c in CARRIED])
def test_the_grips_are_the_references_own(what, refs):
  """A reference that moves its grip -- further along a fork, higher on a
  cup -- can carry a piece past the radius: it fails here first, and its
  grip is read again in `CARRIED`."""
  for source, words in refs:
    text = source if isinstance(source, str) else "\n".join(source)
    assert words in text, f"{what}: its reference no longer says {words!r}"


# ---- 3: a carried candle's front stop --------------------------------------------


def _fires(body, at, scans: int, radius: float) -> int:
  """Scans of the LIDAR's, its noise drawn (seeded), whose band laid flat with
  the left grip held fires the front stop, at `radius`."""
  m, model, data = body.mission, body.model, body.data
  m.lidar.rng = np.random.default_rng(762)
  root = model.body(hm.ROBOT_ROOT).id
  rot = data.xmat[root].reshape(3, 3)
  yaw = math.atan2(rot[1, 0], rot[0, 0])
  c, s = math.cos(yaw), math.sin(yaw)
  level = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ rot
  joints = {j: 0.0 for j in hm.WAIST}
  real = m.reach.grippers["left"].held
  m.reach.grippers["left"].held = lambda: True
  try:
    centres, _ = m._held_drop()
    assert np.linalg.norm(centres[0][:2]) > 0.2 and abs(centres[0][2] - at[2]) < GRIP_OFF_M
    n = 0
    for _ in range(scans):
      scan = m.lidar.band(data, level, float(data.xpos[root][2]), joints=joints,
                          drop=(centres, radius))
      n += bool(len(m.in_front(scan.angles, scan.ranges)))
    return n
  finally:
    m.reach.grippers["left"].held = real


def test_a_carried_candle_never_stops_the_walk_at_the_lidars_radius(carry):
  """#762: a candle 70 mm across and 100 mm tall -- #736's -- gripped 2 cm
  short of its axis and 1.5 cm over its middle, its fingers level, as #736's
  reference grips it, and held at the carry pose: the rings strike its top
  and its near side 0.5-0.6 m out, a range read 4 cm a sigma long or short,
  and at 0.12 its own returns stopped the walk; at the LIDAR's radius, never
  (2000 scans of #762's cast: none from 0.19 m with the grip true, from 0.22
  with it 3 cm off the worst way)."""
  body, at, _ = carry("left", (-0.02, 0.0, 0.065), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "candle")
  assert _fires(body, at, 30, 0.12) >= 3, "the premise: at 0.12 its own returns stop it"
  assert _fires(body, at, 30, hmis.HELD_DROP_M) == 0


# ---- 4: each sensor at its own radius --------------------------------------------


def test_the_band_drops_at_the_lidars_radius_and_the_depth_frame_at_its_own(carry, monkeypatch):
  """The LIDAR's noise is centimetres and the depth camera's millimetres: the
  band (the map, the front stop, what a step's room reads at the head's
  height) leaves out what lies within `HELD_DROP_M` of a held grip, the
  depth camera's frame (its low layer, its knees' and arms' memories) what
  lies within `HELD_DEPTH_M` -- the narrower, so a counter's face near a
  held handle stays in what a reach's swing is judged against."""
  body, at, _ = carry("left", (-0.02, 0.0, 0.065), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "candle")
  m = body.mission
  monkeypatch.setattr(m.reach.grippers["left"], "held", lambda: True)
  drops = {}
  monkeypatch.setattr(m.lidar, "band", lambda *a, **kw: drops.setdefault("band", kw["drop"]))
  m._scan()
  real = hmis._held_out

  def held_out(points, drop):
    drops["depth"] = drop
    return real(points, drop)
  monkeypatch.setattr(hmis, "_held_out", held_out)
  frame = m.latest_depth()
  m._frame_used_t, m._frame_level = -math.inf, True
  monkeypatch.setattr(m, "latest_depth", lambda: frame)
  monkeypatch.setattr(m, "level", lambda: True)
  monkeypatch.setattr(m, "_place_step", lambda: None)
  m._look_step()
  assert drops["band"][1] == hmis.HELD_DROP_M
  assert drops["depth"][1] == hmis.HELD_DEPTH_M
  assert np.allclose(drops["band"][0], drops["depth"][0]), "one grip, placed once"
